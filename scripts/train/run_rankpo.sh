
method=rankpo
run_name=$method-$(date +%Y-%m-%d)

OUTPUT_DIR=./outputs/models/$run_name
LOG_FILE="$OUTPUT_DIR/train_log_$(date +%Y-%m-%d_%H:%M:%S.log)"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p $OUTPUT_DIR

echo "current file is: $0"
# cp "$0" "$OUTPUT_DIR"/run.sh


echo
echo "==================== Training: ===================="
echo "[INFO] logs are saved to: $LOG_FILE"
echo


# sleep 15m

model=outputs/models/xxx    # change models/xxx to the model to be trained
echo "[model] Model to be finetuned: $model"


# LOGLEVEL=INFO \
# CUDA_VISIBLE_DEVICES=2 \
# HF_HOME=/xxx/xxx/.cache/huggingface \
torchrun \
    --nnodes=1 \
    --nproc-per-node 1 \
    --master-port=$MASTER_PORT \
src/run_rankpo.py \
    --model_name_or_path $model \
    --attn_implementation flash_attention_2 \
    --train_data data/predictions/<annotated-pair-data> \
    --output_dir $OUTPUT_DIR \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed configs/ds_zero1_config_llama.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --normalize_embeddings True \
    --max_query_length 1280 \
    --max_passage_length 4096 \
    --logging_steps 1 \
    --dataset_num_proc 8 \
    --reference_free True \
    --disable_dropout False \
    --sft_weight 0.0 \
    --rankpo_weight 1.0 \
    --learning_rate 1e-5 \
    --temperature 0.1 \
    --beta 2.0 \
    --log_level info \
    --save_strategy epoch \
    --save_only_model \
    --remove_unused_columns False \
    --wandb_project rankpo \
    --run_name $run_name \
    &> $LOG_FILE



