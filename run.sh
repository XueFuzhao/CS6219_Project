#export CUDA_VISIBLE_DEVICES=1
#DATA_POSTFIX = "_strands7_10_error0.10_seq_120.txt"
python run_train.py '../dataset/train_data_strands7_10_error0.25_seq_120.txt' '../dataset/train_label_strands7_10_error0.25_seq_120.txt' '../dataset/dev_data_strands7_10_error0.25_seq_120.txt' '../dataset/dev_label_strands7_10_error0.25_seq_120.txt' 0.0003 8
#python run_train.py '../dataset/train_data_strands_10_error0.03_seq_60.txt' '../dataset/train_label_strands_10_error0.03_seq_60.txt' '../dataset/dev_data_strands_10_error0.03_seq_60.txt' '../dataset/dev_label_strands_10_error0.03_seq_60.txt' 0.0003 8





