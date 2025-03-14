
PREFIX=random-hn1

# sleep 2h

# HF_HOME=/xxx/xxx/.cache/huggingface \
python src/get_hard_negatives.py \
--model_name_or_path outputs/models/<model-name> \
--input_file data/raw_train_data.jsonl \
--output_prefix data/llama-$PREFIX \
--bf16 \
--batch_size 32 \
--max_query_length 1280 \
--max_passage_length 4096 \
--search_range 0-100 \
--num_negatives 10 \
--num_clusters 10 \
--seed 36 \
--device 3


