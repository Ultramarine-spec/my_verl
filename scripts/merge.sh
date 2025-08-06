root=$1

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir $root/actor \
    --target_dir $root/hf