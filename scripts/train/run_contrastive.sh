
method=contrastive
run_name=$method-$(date +%Y-%m-%d)

OUTPUT_DIR=./outputs/models/$run_name
LOG_FILE="$OUTPUT_DIR/train_log_$(date +%Y-%m-%d_%H:%M:%S.log)"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p $OUTPUT_DIR

# sleep 1h

echo "current file is: $0"
# cp "$0" "$OUTPUT_DIR"/run.sh

echo
echo "==================== Training: ===================="
echo "[INFO] logs are saved to: $LOG_FILE"
echo


# LOGLEVEL=INFO \
# OMP_NUM_THREADS=8 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# HF_HOME=/xxx/xxx/.cache/huggingface \
torchrun \
    --nnodes=1 \
    --nproc-per-node 4 \
    --master-port=$MASTER_PORT \
src/run_contrastive.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --attn_implementation flash_attention_2 \
    --train_data data/train_data.jsonl \
    --output_dir $OUTPUT_DIR \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed configs/ds_zero1_config_llama.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_negatives 5 \
    --use_inbatch_neg \
    --negatives_cross_device \
    --dataloader_drop_last True \
    --normalize_embeddings True \
    --temperature 0.02 \
    --max_query_length 1280 \
    --max_passage_length 4096 \
    --logging_steps 1 \
    --log_level info \
    --save_strategy epoch \
    --save_only_model \
    --remove_unused_columns False \
    --wandb_project contrastive \
    --run_name $run_name \
    &> $LOG_FILE

