df -h

export WANDB_DIR="/9950backfile/jhchen/wandb_log"
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7，
export CUDA_VISIBLE_DEVICES=0
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
set -x

curl -X POST http://172.18.127.52:8080/sleep
curl http://172.18.127.52:8080/is_sleeping

# actor="Qwen/Qwen2.5-7B-Instruct"
# actor="meta-llama/Llama-3.1-8B-Instruct"
actor="Qwen/Qwen2.5-1.5B-Instruct"
# actor="Qwen/Qwen3-1.7B"
# actor="Qwen/Qwen3-4B"
# actor="Qwen/Qwen3-8B"
rm="Qwen/Qwen3-8B"
# rm="Qwen/Qwen3-32B"
# rm="Qwen/Qwen2.5-1.5B-Instruct"
# rm="Qwen/Qwen2.5-7B-Instruct"
model_path="/public/jhchen/huggingface/pretrained_model/$actor"
rm_path="/public/jhchen/huggingface/pretrained_model/$rm"
reward_func_path="/public/jhchen/verl/verl/utils/reward_score/checklist.py"
verl_path="/public/jhchen/verl"

# train_files="$verl_path/0-verl-data/allenai/WildChat-1M-LongGen/Qwen3-235B-A22B/LongGen-deduplicated-long-checklist-split/train.parquet"
# test_files="$verl_path/0-verl-data/allenai/WildChat-1M-LongGen/Qwen3-235B-A22B/LongGen-deduplicated-long-checklist-split/test.parquet"

# train_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-checklist-2k-split/train.parquet"
# test_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-checklist-2k-split/test.parquet"

# train_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-split-2025-06-27-10:29:48/train.parquet"
# test_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-split-2025-06-27-10:29:48/test.parquet"


# train_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/refined-split-2025-06-28-09:48:12/train.parquet"
# test_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/refined-split-2025-06-28-09:48:12/test.parquet"

# train_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/checklist+refined-split-2025-07-01-11:41:20/train.parquet"
# test_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/checklist+refined-split-2025-07-01-11:41:20/test.parquet"

# train_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/length-split-2025-06-28-05:46:50/train.parquet"
# test_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/length-split-2025-06-28-05:46:50/test.parquet"

# train_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/length-upper-split-2025-07-15 07:27:12/train.parquet"
# test_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/length-upper-split-2025-07-15 07:27:12/test.parquet"

# train_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/length-half-split-2025-07-14-13:45:03/train.parquet"
# test_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/length-half-split-2025-07-14-13:45:03/test.parquet"

train_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/new-length-half-split-2025-07-15-19:11:18/train.parquet"
test_files="$verl_path/0-verl-data/allenai/WildChat-1M-preprocessed/Qwen3-235B-A22B/LongGen-2025-06-27-00:16:35/checklist-2025-06-27-10:29:48/new-length-half-split-2025-07-15-19:11:18/test.parquet"
data_tag="new-length-half-split"


train_batch_size=4
ppo_mini_batch_size=4
# train_batch_size=4
# ppo_mini_batch_size=4
max_response_length=4096
num_rollouts=8

if [[ $actor == "Qwen/Qwen2.5-1.5B-Instruct" ]]; then
    tokens=32768
elif [[ $actor == "Qwen/Qwen3-1.7B" ]]; then
    tokens=32768
elif [[ $actor == "Qwen/Qwen3-4B" ]]; then
    tokens=32768
elif [[ $actor == "Qwen/Qwen3-8B" ]]; then
    tokens=16384
elif [[ $actor == "Qwen/Qwen2.5-3B-Instruct" ]]; then
    tokens=32768
elif [[ $actor == "Qwen/Qwen2.5-7B-Instruct" ]]; then
    tokens=16384
elif [[ $actor == "meta-llama/Llama-3.1-8B-Instruct" ]]; then
    tokens=16384
fi

template="3-level"

if [[ $rm == "Qwen/Qwen3-8B" ]]; then
    rm_tp=1
elif [[ $rm == "Qwen/Qwen3-4B" ]]; then
    rm_tp=1
elif [[ $rm == "Qwen/Qwen3-32B" ]]; then
    rm_tp=2
fi

use_kl_loss=False
overlong=False
length_constraint=True
decay_factor=0.5
weight=0.5

project_name='LongGen'
experiment_name="actor:$(basename "$actor")/rm:$(basename "$rm")/$data_tag/$template/bz:$train_batch_size-rollouts:$num_rollouts-len:$max_response_length/kl:$use_kl_loss-constraint:$length_constraint-decay:$decay_factor-weight:$weight/$(date "+%Y%m%d-%H%M%S")"
# experiment_name="actor:Qwen3-4B/rm:Qwen3-8B/new-length-half-split/general/bz:64-rollouts:32-len:8192/kl:False-constraint:True-decay:0.5-weight:0.5/20250722-212203"
mkdir -p "$verl_path/1-verl-outputs/$project_name/$experiment_name"
cp "$0" "$verl_path/1-verl-outputs/$project_name/$experiment_name/launch.sh"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size\
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$tokens \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$tokens \
    actor_rollout_ref.rollout.n=$num_rollouts \
    \
    reward_model.enable=False \
    reward_model.reward_manager=llm \
    +reward_model.reward_kwargs.llm=$rm_path \
    \
    custom_reward_function.path=$reward_func_path \
    custom_reward_function.name=compute_score \
    \
    trainer.logger=['console'] \
    trainer.rollout_data_dir="$verl_path/1-verl-outputs/$project_name/$experiment_name/rollout" \
    trainer.validation_data_dir="$verl_path/1-verl-outputs/$project_name/$experiment_name/validation" \
    trainer.default_local_dir="$verl_path/1-verl-outputs/$project_name/$experiment_name" \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1