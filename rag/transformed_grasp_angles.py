import os
import random
import sys
import shutil
import numpy as np
from PIL import Image, ImageDraw
import cv2
import json
from argparse import ArgumentParser

from sapien.core import Pose

from rag.env import Env, ContactError
from rag.camera import Camera
from rag.robots import Robot


def save_draw_point(img_arr, point, file_path):
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    radius = 2
    y, x = point
    draw.ellipse((y - radius, x - radius, y + radius, x + radius), fill='red')
    img.save(file_path)


def save_draw_line(img_arr, p1, p2, file_path, fill='blue', width=2):
    """
    Draw a line on an image and save it to file.
    
    Draws a line between two points on the given image array, adds a red circle
    at the first point, and saves the result as an image file.
    
    Args:
        img_arr (np.ndarray): Input image array (values 0-1)
        p1 (tuple): Starting point coordinates (y, x)
        p2 (tuple): Ending point coordinates (y, x)
        file_path (str): Output file path to save the image
        fill (str): Line color (default: 'blue')
        width (int): Line width in pixels (default: 2)
    """
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.line([p1, p2], fill=fill, width=width)
    y, x = p1
    radius = 2
    draw.ellipse((y - radius, x - radius, y + radius, x + radius), fill='red')
    img.save(file_path)


def generate_aff(object_link_ids, env, cam, cam_XYZA_world):
    """
    Generate affordance maps for object parts based on their joint types.
    
    For each object part, generates an affordance map indicating valid interaction
    regions based on the joint type (prismatic or revolute). The affordance map
    shows areas where actions can be applied to manipulate the part.
    
    Args:
        object_link_ids (list): List of object link IDs to process
        env (Env): Environment object containing the scene
        cam (Camera): Camera object for rendering masks and coordinates
        cam_XYZA_world (np.ndarray): World coordinates for each pixel
        
    Returns:
        list: List of affordance maps (np.ndarray) for each object part
    """
    aff_gt_all = []
    for i in range(len(object_link_ids)):
        action_type, hinge_pose, joint = env.set_target_object_part_actor_id(object_link_ids[i])
        pose = joint.get_parent_link().pose * joint.get_pose_in_parent_frame()
        axis_direct = pose.to_transformation_matrix()[:3, :3] @ [1, 0, 0]
        axis_direct /= np.linalg.norm(axis_direct)
        hinge_point = hinge_pose.p

        part_movable_link_mask = cam.get_movable_link_mask([object_link_ids[i]])

        if str(action_type).split('.')[-1] == 'PRISMATIC':
            aff_gt = (part_movable_link_mask > 0).astype(np.uint8) * 255
            aff_gt_all.append(aff_gt)


        elif str(action_type).split('.')[-1] == 'REVOLUTE':
            indices_of_ones = np.where(part_movable_link_mask == 1)
            sampled_indices = np.random.choice(np.arange(len(indices_of_ones[0])), size=len(indices_of_ones[0]),
                                               replace=False)  # Shuffle the order of these pixels
            sampled_points = np.vstack(
                (indices_of_ones[0][sampled_indices], indices_of_ones[1][sampled_indices])).T  # Arrange points in (X, 2) array according to new order
            black_aff = np.zeros((336, 336))

            for index in range(len(sampled_indices)):
                cur_index = sampled_points[index]
                cur_point = cam_XYZA_world[cur_index[0], cur_index[1]]
                point_rotated, flow_norm = rotate_point_around_axis(cur_point[:3], hinge_point, axis_direct)
                black_aff[cur_index[0], cur_index[1]] = flow_norm

            non_zero_values = black_aff[black_aff != 0]
            if non_zero_values.size > 0:
                min_value = np.min(non_zero_values)
                max_value = np.max(non_zero_values)
                normalized_flow = (black_aff - min_value) / (max_value - min_value)
            else:
                normalized_flow = np.zeros_like(black_aff)

            normalized_flow *= (part_movable_link_mask > 0)

            aff_gt = (normalized_flow * 255).astype(np.uint8)

            aff_gt_all.append(aff_gt)
    return aff_gt_all


def rotate_point_around_axis(point, axis_point, axis_direct):
    v1, v2, v3 = point
    c1, c2, c3 = axis_point
    d1, d2, d3 = axis_direct

    # calculate the rotation angle
    theta = np.arccos(d1) if d2 == 0 else np.arccos(d2) if d1 == 0 else np.arccos(d3)
    if theta == 0:
        return point, 0

    # calculate the rotation matrix
    R_mat = np.array([[np.cos(theta) + d1 ** 2 * (1 - np.cos(theta)),
                       d1 * d2 * (1 - np.cos(theta)) - d3 * np.sin(theta),
                       d1 * d3 * (1 - np.cos(theta)) + d2 * np.sin(theta)],
                      [d2 * d1 * (1 - np.cos(theta)) + d3 * np.sin(theta),
                       np.cos(theta) + d2 ** 2 * (1 - np.cos(theta)),
                       d2 * d3 * (1 - np.cos(theta)) - d1 * np.sin(theta)],
                      [d3 * d1 * (1 - np.cos(theta)) - d2 * np.sin(theta),
                       d3 * d2 * (1 - np.cos(theta)) + d1 * np.sin(theta),
                       np.cos(theta) + d3 ** 2 * (1 - np.cos(theta))]])

    # calculate the translated vector
    T = np.array([c1, c2, c3])

    # calculate the rotated point
    P = np.array([v1, v2, v3])
    P_rot = np.dot(R_mat, P - T)
    P_rotated = P_rot + T

    # calculate the flow norm
    flow_norm = np.linalg.norm(P_rotated - P)

    return P_rotated, flow_norm


def point_camera3d_to_img2d(point_cam, cam):
    point_img = [-point_cam[1], -point_cam[2], point_cam[0]]
    point_img = np.dot((cam.get_metadata())["camera_matrix"][:3, :3], point_img)
    point_img = (point_img / point_img[2])[:2]
    point_img = (int(point_img[0]), int(point_img[1]))
    return point_img


def point_world3d_to_camera3d(point_world, cam):
    """
    Convert a 3D point from world coordinates to camera coordinates.
    
    Transforms a 3D point using the inverse of the camera's transformation matrix
    to convert from world coordinate system to camera coordinate system.
    
    Args:
        point_world (np.ndarray): 3D point in world coordinates [x, y, z]
        cam (Camera): Camera object containing transformation metadata
        
    Returns:
        np.ndarray: 3D point in camera coordinates [x, y, z]
    """
    point_cam = np.linalg.inv(cam.get_metadata()['mat44']) @ np.append(point_world, 1)
    point_cam = point_cam[:3]
    return point_cam


def load_data(json_path):
    """
    Load JSON data from a file.
    
    Simple utility function to load and parse JSON data from a file path.
    
    Args:
        json_path (str): Path to the JSON file to load
        
    Returns:
        dict: Parsed JSON data as a dictionary
    """
    # Assume this is your function for loading JSON data
    import json
    with open(json_path, 'r') as f:
        return json.load(f)


# compute the rotnat
def add_noise(vector, noise_level=0.01):
    """
    Add Gaussian noise to a vector.
    
    Adds random Gaussian noise to each component of the input vector
    to simulate measurement uncertainty or provide data augmentation.
    
    Args:
        vector (np.ndarray): Input vector to add noise to
        noise_level (float): Standard deviation of Gaussian noise (default: 0.01)
        
    Returns:
        np.ndarray: Vector with added Gaussian noise
    """
    noise = np.random.normal(-noise_level, noise_level, vector.shape)
    return vector + noise


def orthogonalize_and_normalize(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2)
    return v1, v2



def transformed_angle_of_object(input_dir, out_dir, num_id, angle_y, angle_x):
    try:
        info = input_dir.split("/")[-1]
        # print("info: ",info)
        shape_id = info.split('_')[0]
        # print("shape_id: ",shape_id)
        random_seed = None
        primact_type = 'pulling_' + str(num_id + 1)
        no_gui = True
        if no_gui:
            out_dir = os.path.join(out_dir, primact_type)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_info = dict()

        if random_seed is not None:
            np.random.seed(random_seed)
            out_info['random_seed'] = random_seed
        json_path = os.path.join(input_dir, 'result.json')
        data = load_data(json_path)
        cam_theta = data['camera_metadata']['theta']
        new_cam_theta = cam_theta + angle_y  # Increase cam_theta
        cam_phi = data['camera_metadata']['phi']
        new_cam_phi = cam_phi + angle_x  # Increase cam_phi
        cam_dist = data['camera_metadata']['dist']
        env = Env(show_gui=False)
        cam = Camera(env, theta=new_cam_theta, phi=new_cam_phi, dist=cam_dist)  # Example camera initialization
        out_info['camera_metadata'] = cam.get_metadata_json()

        # load shape
        object_urdf_fn = '../data_collection/asset/original_sapien_dataset/dataset/%s/mobility.urdf' % shape_id
        object_material = env.get_material(4, 4, 0.01)

        # set object scale
        state = data['object_state']
        out_info['object_state'] = state
        scale = data['scale']
        out_info['scale'] = scale

        # set object part angle
        joint_angles = env.load_object(object_urdf_fn, object_material, scale, state=state)
        out_info['joint_angles'] = joint_angles
        out_info['joint_angles_lower'] = env.joint_angles_lower
        out_info['joint_angles_upper'] = env.joint_angles_upper
        cur_qpos = env.get_object_qpos()

        # simulate some steps for the object to stay rest
        still_timesteps = 0
        wait_timesteps = 0
        while still_timesteps < 5000 and wait_timesteps < 20000:
            env.step()
            env.render()
            cur_new_qpos = env.get_object_qpos()
            invalid_contact = False
            for c in env.scene.get_contacts():
                for p in c.points:
                    if abs(p.impulse @ p.impulse) > 1e-4:
                        invalid_contact = True
                        break
                if invalid_contact:
                    break
            if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
                still_timesteps += 1
            else:
                still_timesteps = 0
            cur_qpos = cur_new_qpos
            wait_timesteps += 1

        if still_timesteps < 5000:
            shutil.rmtree(out_dir)
            env.close()
            return

        # capture the original state of object
        rgb, depth = cam.get_observation()
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # generate the corresponding point cloud
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
        cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
        cam_XYZA_pts1 = np.ones((cam_XYZA_pts.shape[0], 4))
        cam_XYZA_pts1[:, :3] = cam_XYZA_pts
        cam_XYZA_pts_world = (cam.get_metadata()['mat44'] @ cam_XYZA_pts1.T).T
        cam_XYZA_world = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts_world[:, :3], depth.shape[0],
                                                 depth.shape[1])

        # capture the surface norm
        gt_nor = cam.get_normal_map()

        object_link_ids = env.movable_link_ids
        gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)

        aff_gt_all = generate_aff(object_link_ids, env, cam, cam_XYZA_world)
        # print('aff_gt_all:', aff_gt_all[0], aff_gt_all[0].shape)
        if len(aff_gt_all) == 0:
            shutil.rmtree(out_dir)
            env.close()
            exit()
        else:
            result_aff = aff_gt_all[0]
            for mask in aff_gt_all[1:]:
                result_aff = result_aff + mask
        given_pixel = data['pixel_locs']
        given_x, given_y = given_pixel[0], given_pixel[1]

        min_x = max(given_x - 25, 0)
        max_x = min(given_x + 25, 336)
        min_y = max(given_y - 25, 0)
        max_y = min(given_y + 25, 336)

        xs, ys = np.where((result_aff / 255) > 0.6)

        valid_indices = np.where((xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y))[0]

        if len(valid_indices) == 0:
            shutil.rmtree(out_dir)
            env.close()
            return

        idx = np.random.choice(valid_indices)
        x, y = xs[idx], ys[idx]
        radius = 2
        draw.ellipse((y - radius, x - radius, y + radius, x + radius), fill='red')
        img.save(os.path.join(out_dir, 'rag_example.png'))
        out_info['pixel_locs'] = [int(x), int(y)]
        env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y] - 1])
        out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
        out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
        part_movable_link_mask = cam.get_movable_link_mask([object_link_ids[gt_movable_link_mask[x, y] - 1]])

        norm_cam = gt_nor[x, y, :3]
        norm_cam /= np.linalg.norm(norm_cam)
        out_info['norm_cam'] = norm_cam.tolist()
        norm_world = cam.get_metadata()['mat44'][:3, :3] @ norm_cam
        out_info['norm_world'] = norm_world.tolist()

        action_direction_cam = -gt_nor[x, y, :3]
        action_direction_cam /= np.linalg.norm(action_direction_cam)
        out_info['gripper_direction_camera'] = action_direction_cam.tolist()
        action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
        out_info['gripper_direction_world'] = action_direction_world.tolist()

        position_cam = cam_XYZA[x, y, :3]
        out_info['position_cam'] = position_cam.tolist()
        position_cam_xyz1 = np.ones((4), dtype=np.float32)
        position_cam_xyz1[:3] = position_cam
        position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
        position_world = position_world_xyz1[:3]
        out_info['position_world'] = position_world.tolist()

        up = np.array(action_direction_world, dtype=np.float32)
        up /= np.linalg.norm(up)

        up = add_noise(up)
        up /= np.linalg.norm(up)

        forward = np.random.randn(3).astype(np.float32)
        forward /= np.linalg.norm(forward)

        up, forward = orthogonalize_and_normalize(up, forward)

        left = np.cross(up, forward)
        left /= np.linalg.norm(left)

        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)

        left = np.cross(up, forward)
        left /= np.linalg.norm(left)

        out_info['gripper_forward_direction_world'] = forward.tolist()
        forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
        out_info['gripper_forward_direction_camera'] = forward_cam.tolist()
        out_info['gripper_up_direction_world'] = up.tolist()
        up_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ up
        out_info['gripper_up_direction_camera'] = up_cam.tolist()

        rotmat = np.eye(4).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up

        contact_dist = 0.11  

        contact_rotmat = np.array(rotmat, dtype=np.float32)
        contact_rotmat[:3, 3] = position_world - up * contact_dist
        contact_pose = Pose().from_transformation_matrix(contact_rotmat)
        out_info['contact_rotmat_world'] = contact_rotmat.tolist()

        start_rotmat = np.array(rotmat, dtype=np.float32)
        start_rotmat[:3, 3] = position_world - up * 0.2
        start_pose = Pose().from_transformation_matrix(start_rotmat)
        out_info['start_rotmat_world'] = start_rotmat.tolist()

        pull_rotmat = np.array(rotmat, dtype=np.float32)
        pull_rotmat[:3, 3] = position_world - up * 0.5
        out_info['end_rotmat_world'] = pull_rotmat.tolist()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        robot_urdf_fn = os.path.join(current_dir, 'robots', 'panda_gripper.urdf')
        
        if not os.path.exists(robot_urdf_fn):
            raise FileNotFoundError(f"Robot URDF file not found: {robot_urdf_fn}")
            
        robot_material = env.get_material(4, 4, 0.01)
        robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))

        robot.robot.set_root_pose(start_pose)
        env.render()

        out_info['start_target_part_qpos'] = env.get_target_part_qpos()
        target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
        position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1

        robot.close_gripper()
        robot.move_to_target_pose(contact_rotmat, 2000)
        robot.wait_n_steps(2000)

        suction_drive = env.scene.create_drive(
            robot.robot.get_links()[-1],
            robot.robot.get_links()[-1].get_cmass_local_pose(),
            env.target_object_part_actor_link,
            env.target_object_part_actor_link.get_cmass_local_pose(),
        )
        suction_drive.set_x_properties(stiffness=45000, damping=0)
        suction_drive.set_y_properties(stiffness=45000, damping=0)
        suction_drive.set_z_properties(stiffness=45000, damping=0)

        robot.move_to_target_pose(pull_rotmat, 2000)
        robot.wait_n_steps(2000)

        rgb_final_pose, final_depth = cam.get_observation()
        Image.fromarray((rgb_final_pose * 255).astype(np.uint8)).save(os.path.join(out_dir, 'rag_result.png'))
        target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
        position_world_xyz1_end = target_link_mat44 @ position_local_xyz1

        out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
        out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()

        out_info['result'] = 'VALID'
        out_info['final_target_part_qpos'] = env.get_target_part_qpos()
        abs_motion = abs(out_info['final_target_part_qpos'] - out_info['start_target_part_qpos'])
        j = out_info['target_object_part_joint_id']
        tot_motion = out_info['joint_angles_upper'][j] - out_info['joint_angles_lower'][j] + 1e-8
        mov_dir = np.array(out_info['touch_position_world_xyz_end'], dtype=np.float32) - \
                  np.array(out_info['touch_position_world_xyz_start'], dtype=np.float32) + [1e-8, 1e-8, 1e-8]
        mov_dir /= np.linalg.norm(mov_dir)
        intended_dir = -np.array(out_info['gripper_direction_world'], dtype=np.float32)
        mani_success = (intended_dir @ mov_dir > 0.5) and ((abs_motion > 0.1) or (abs_motion / tot_motion > 0.5))
        out_info['mani_succ'] = str(mani_success)

        if mani_success:
            with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
                json.dump(out_info, fout)
        else:
            shutil.rmtree(out_dir)
            env.close()
            return

        env.close()
    except Exception as e:
        shutil.rmtree(out_dir)
        env.close()
        return


def transformed_grasp_angles(input_dir, out_dir):
    dst_dir = os.path.join(out_dir, 'pulling_0')
    os.makedirs(dst_dir, exist_ok=True)

    shutil.copytree(input_dir, dst_dir, dirs_exist_ok=True)
    angle_list = [-0.6,-0.3, +0.3, +0.6, -0.3, +0.3]
    for i, angle in enumerate(angle_list):
        if i < 4:
            transformed_angle_of_object(input_dir, out_dir, i, angle_y=angle, angle_x=0)
        else:
            transformed_angle_of_object(input_dir, out_dir, i, angle_y=0, angle_x=angle)
