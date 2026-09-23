import os
import subprocess

model = os.getenv('MODEL', 'sup_simcse_roberta_base')
encoding = os.getenv('ENCODER_TYPE', 'cross_encoder')
lr = os.getenv('LR', '3e-5')
wd = os.getenv('WD', '0.1')
transform = os.getenv('TRANSFORM', 'False')
objective = os.getenv('OBJECTIVE', 'triplet_mse')
triencoder_head = os.getenv('TRIENCODER_HEAD', 'None')
seed = os.getenv('SEED', '42')
output_dir = os.getenv('OUTPUT_DIR', 'output')
train_file = os.getenv('TRAIN_FILE', 'data/csts_train.csv')
eval_file = os.getenv('EVAL_FILE', 'data/csts_validation.csv')
test_file = os.getenv('TEST_FILE', 'data/csts_test.csv')
epoch = os.getenv('EPOCH', '5')

lambda_1 = os.getenv('LAMBDA_1', '0.1')
lambda_2 = os.getenv('LAMBDA_2', '1')
lambda_3 = os.getenv('LAMBDA_3', '1')
token_top_k = os.getenv('TOKEN_TOP_K', '3')


command = [
    "python", "run_sts.py",
    "--output_dir", f"output/{model}/0",
    "--model_name_or_path", model,
    "--objective", objective,
    "--encoding_type", encoding,
    "--pooler_type", "cls",
    "--freeze_encoder", "False",
    "--transform", transform,
    "--triencoder_head", triencoder_head,
    "--max_seq_length", "512",
    "--train_file", train_file,
    "--validation_file", eval_file,
    "--test_file", test_file,
    "--condition_only", "False",
    "--sentences_only", "False",
    "--num_train_epochs", epoch,
    "--do_train",
    "--do_eval",
    "--do_predict",
    "--evaluation_strategy", "epoch",
    "--save_strategy", "epoch",
    "--per_device_train_batch_size", "16",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", lr,
    "--weight_decay", wd,
    "--max_grad_norm", "0.0",
    "--lr_scheduler_type", "linear",
    "--warmup_ratio", "0.1",
    "--log_level", "info",
    "--disable_tqdm", "True",
    "--save_strategy", "epoch",
    "--save_total_limit", "1",
    "--seed", seed,
    "--data_seed", seed,
    "--fp16", "True",
    "--log_time_interval", "15",
    "--token_top_k", str(token_top_k),
    "--lambda_1", str(lambda_1),
    "--lambda_2", str(lambda_2),
    "--lambda_3", str(lambda_3),

]

# 执行命令
subprocess.run(command)

