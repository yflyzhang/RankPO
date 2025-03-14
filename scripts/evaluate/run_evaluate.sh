echo
echo "==================== Run Evaluation ===================="

# sleep 1h

# HF_HOME=/xxx/xxx/.cache/huggingface \
python \
    src/evaluate.py \
    --evaluate_all_checkpoints True \
    --model_name_or_path outputs/models/<model-name> \
    --attn_implementation flash_attention_2 \
    --output_dir outputs/test_results \
    --bf16 \
    --k 100 \
    --batch_size 64 \
    --query_data data/raw_test_data.jsonl \
    --corpus_data data/raw_authors.jsonl \
    --max_query_length 1280 \
    --max_passage_length 4096 \
    --device 2


