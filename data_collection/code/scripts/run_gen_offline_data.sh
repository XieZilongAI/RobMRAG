# generate around 20,000 training samples, then stop it manually
python gen_offline_data.py \
  --data_dir ../data/train_data_ori_depth \
  --data_fn ../stats/train_id.txt\
  --primact_types pulling \
  --num_processes 40 \
  --num_epochs 10 \
  --starting_epoch 20 \
  --ins_cnt_fn ../stats/ins_cnt_46cats.txt \
  --mode train

# delete the extra testing dataset, and remain around 2,000 testing samples. Make sure that each category has as least 50 samples.
python gen_offline_data.py \
 --data_dir ../data/test_data_ori_depth \
 --data_fn ../stats/test_id.txt\
 --primact_types pulling \
 --num_processes 10 \
 --num_epochs 1 \
 --starting_epoch 0 \
 --ins_cnt_fn ../stats/ins_cnt_46cats.txt \
 --mode test
