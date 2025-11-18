#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务统计脚本
统计JSON文件中的任务类型和每个任务的出现次数
"""

import json
import re
from collections import Counter
import os

def extract_task_from_content(content):
    """
    从content中提取任务信息
    查找 "The task is: " 后面的内容
    """
    # 使用正则表达式匹配 "The task is: " 后面的内容
    pattern = r"The task is:\s*(.+?)(?:\s*<image>|\s*$)"
    match = re.search(pattern, content)
    
    if match:
        task = match.group(1).strip()
        return task
    return None

def analyze_tasks(json_file_path):
    """
    分析JSON文件中的任务统计信息
    """
    print(f"正在分析文件: {json_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件 {json_file_path} 不存在")
        return
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取JSON文件，共有 {len(data)} 条记录")
        
        # 统计任务
        task_counter = Counter()
        total_records = 0
        records_with_tasks = 0
        
        for record in data:
            total_records += 1
            
            # 获取messages中的content
            if 'messages' in record and len(record['messages']) > 0:
                for message in record['messages']:
                    if message.get('role') == 'user' and 'content' in message:
                        content = message['content']
                        task = extract_task_from_content(content)
                        
                        if task:
                            task_counter[task] += 1
                            records_with_tasks += 1
                            break  # 找到任务后跳出内层循环
        
        # 输出统计结果
        print("\n" + "="*60)
        print("任务统计结果")
        print("="*60)
        print(f"总记录数: {total_records}")
        print(f"包含任务的记录数: {records_with_tasks}")
        print(f"不同任务类型数: {len(task_counter)}")
        print("\n各任务类型及出现次数:")
        print("-"*40)
        
        # 按出现次数降序排列
        for task, count in task_counter.most_common():
            print(f"{task:<30} : {count:>6} 次")
        
        # 输出汇总信息
        print("\n" + "="*60)
        print("汇总信息")
        print("="*60)
        print(f"最常见的任务: {task_counter.most_common(1)[0][0]} ({task_counter.most_common(1)[0][1]} 次)")
        print(f"最不常见的任务: {task_counter.most_common()[-1][0]} ({task_counter.most_common()[-1][1]} 次)")
        
        # 计算任务分布
        if len(task_counter) > 0:
            avg_count = sum(task_counter.values()) / len(task_counter)
            print(f"平均每个任务出现次数: {avg_count:.2f}")
        
        return task_counter
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

def main():
    """
    主函数
    """
    # JSON文件路径
    json_file_path = "/data2/xzl/RobMRAG/collect_train_data/data/train_json/train21_20000_no_apr_rag1_improve_prompt_v3.json"
    
    print("机器人任务统计脚本")
    print("="*60)
    
    # 分析任务
    task_counter = analyze_tasks(json_file_path)
    
    if task_counter:
        print(f"\n分析完成！共发现 {len(task_counter)} 种不同的任务类型。")

if __name__ == "__main__":
    main()
