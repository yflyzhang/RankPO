echo
echo "==================== Run Evaluation ===================="


lrs='5e-7 1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5'
seeds='95'
device=2


echo "seeds: $seeds"
echo "learning rates: $lrs"

for seed in $seeds; do
    for lr in $lrs; do

        echo "seed: $seed, lr: $lr"
        
        # Change model
        method=rankpo-loss
        model_name=llama-$method-seed$seed-lr$lr
        model_path=outputs/models/$method/$model_name

        # HF_HOME=/xxx/xxx/.cache/huggingface \
        python \
            src/evaluate.py \
            --evaluate_all_checkpoints False \
            --model_name_or_path $model_path \
            --attn_implementation flash_attention_2 \
            --output_dir outputs/test_results/$method \
            --bf16 \
            --k 100 \
            --batch_size 32 \
            --query_data data/raw_test_data.jsonl \
            --corpus_data data/raw_authors.jsonl \
            --max_query_length 1280 \
            --max_passage_length 4096 \
            --device $device

    done

done

