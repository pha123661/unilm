python run.py with data_root=./arrow/F30K/ num_gpus=1 num_nodes=1 "task_moe_prefix_irtr_f30k_base_image384" per_gpu_batchsize=40 load_path="./weights/vlmo_base_patch16_224.pt" log_dir="./logs" precision="bf16"