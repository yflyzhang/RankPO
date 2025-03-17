import os
import sys
import logging

import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from accelerate import PartialState
import datasets


from arguments import ModelArguments, TrainDataArguments, TrainArguments
from data_utils import ContrastiveDataCollatorWithPadding
from modeling import ModelForTraining
from contrastive_trainer import ContrastiveTrainer


# Setup logging
log_levels = {
    "debug": logging.DEBUG,         # 10
    "info": logging.INFO,           # 20
    "warning": logging.WARNING,     # 30
    "error": logging.ERROR,         # 40
    "critical": logging.CRITICAL,   # 50
}
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s]  %(message)s",
    # format="[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s",
    # format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)



def main():
    parser = HfArgumentParser((ModelArguments, TrainDataArguments, TrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: TrainDataArguments
    training_args: TrainArguments
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    
    ###############
    # Setup logging
    ###############
    # For local process 0 [Rank 0], log_level is set as `training_args.log_level`;
    # For other local processes [Rank 1-n], log_level is set as `training_args.log_level_replica`.
    # By default, `training_args.log_level_replica` is set as `warning`.
    # See `TrainArguments` for details:
    # https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/training_args.py#L914
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    is_distributed = bool(training_args.local_rank != -1)
    logger.critical(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {is_distributed}, 16-bits training: {training_args.fp16}"
    )
    
    logger.info(f"Model arguments:\n  {model_args.to_json_string()}")
    logger.info(f"Data arguments:\n  {data_args.to_json_string()}")
    logger.info(f"Training arguments:\n  {training_args.to_json_string()}")
    
    # Set seed
    set_seed(training_args.seed)

    #######################
    # Load pretrained model
    #######################
    model = ModelForTraining(
        model_name_or_path=model_args.model_name_or_path,
        attn_implementation=model_args.attn_implementation,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        use_cache=False if training_args.gradient_checkpointing else True,
        
        normalize_embeddings=training_args.normalize_embeddings,
        use_inbatch_neg=training_args.use_inbatch_neg,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
    )

    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer, # use fast tokenizer?
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # add pad token for batched inputs for Llama3.2
    if not tokenizer.pad_token:
        # # case 1: use eos_token as the pad_token [TODO/TOTEST]
        # tokenizer.pad_token = tokenizer.eos_token
        
        # case 2: use the reserved token '<|finetune_right_pad_id|>' (in llama3.2) as the pad_token
        tokenizer.pad_token = '<|finetune_right_pad_id|>'
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|finetune_right_pad_id|>')
        
        logger.warning(f"set: {tokenizer.pad_token=}, {tokenizer.pad_token_id=}")
        
        # set model.config as well
        if not model.config.pad_token_id:
            model.model.config.pad_token_id = tokenizer.pad_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.warning(f"set: {model.config.pad_token_id=}")
        
    # Add special tokens e.g., '<keyword>', '<title>', to deal with concatenated paper titles and abstracts
    # The special tokens used:
    #     '<keyword>'+'<sep>'.join(x['paper_keyword'])+'</keyword>' \
    #     '<title>'+'<sep>'.join(x['paper_title'])+'</title>' \
    #     '<abstract>'+'<sep>'.join(x['paper_abstract'])+'</abstract>'
    
    special_tokens_dict = {
        'additional_special_tokens': [
            '<keyword>', '</keyword>', 
            '<title>', '</title>', 
            '<abstract>', '</abstract>', 
            '<sep>',
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))
    model.model.resize_token_embeddings(len(tokenizer))    # model.model is `XLMRobertaModel`
    # `num_added_toks` should be 7?
    # if training_args.local_rank == 0:   # print rank0 only
    # print("====================================")
    logger.info(f"Special tokens:\n  {tokenizer.get_special_tokens_mask=}")
    logger.info(f"Resized embeddings:\n  {model.model.get_input_embeddings()}")
    logger.info(f'Config:\n  {model.config}')
    # print("====================================")
    
    
    #######################
    # Dataset preprocessing
    #######################
    train_dataset = datasets.load_dataset('json', data_files=data_args.train_data, split='train')
    
    # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    
    # Tokenize the data
    def tokenize_row(row, tokenizer, max_query_length, max_passage_length):
        tokenized_row = {}
        tokenized_row['query'] = tokenizer(row['query'], max_length=max_query_length, truncation=True)
        tokenized_row['positives'] = tokenizer(row['positives'], max_length=max_passage_length, truncation=True)
        tokenized_row['negatives'] = tokenizer(row['negatives'], max_length=max_passage_length, truncation=True)
        return tokenized_row
    
    # Compute that only on the main process for faster data processing
    with PartialState().local_main_process_first():
        train_dataset = train_dataset.map(
            tokenize_row,
            # remove_columns=column_names,
            fn_kwargs={
                "tokenizer": tokenizer, 
                "max_query_length": data_args.max_query_length,
                "max_passage_length": data_args.max_passage_length,
            },
            num_proc=data_args.dataset_num_proc,
            desc="Tokenizing",
        )
    
    # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    # torch.distributed.breakpoint(rank=1)
    

    # Add or remove wandb/WandbCallback if necessary
    if training_args.wandb_project and 'wandb' not in training_args.report_to:
        training_args.report_to.append('wandb')
        logger.warning("WandB is enabled!")
    elif not training_args.wandb_project and 'wandb' in training_args.report_to:
        training_args.report_to.remove('wandb')
        logger.warning("WandB is disabled!")
    
    # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    
    #########################
    # Instantiate trainer
    #########################
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=ContrastiveDataCollatorWithPadding(
            pad_token_id = tokenizer.pad_token_id,
            num_negatives = data_args.num_negatives,
        )
    )
    
    # Create the path if not exists; 
    # But `trainer.save_model()/_save` will do the same, so no need to do it here.
    # Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    # return
    # for p in model.parameters():print(p, p.device, p.dtype)
    
    # Training
    # if training_args.do_train:
    checkpoint = training_args.resume_from_checkpoint or None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # return
    
    trainer.save_model()    # Saves the tokenizer too for easy upload
    
    # ######## add a breakpoint for debug?
    # torch.distributed.breakpoint(rank=0)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    
    trainer.log_metrics("train", metrics)   # print/log metrics in a specially formatted way
    trainer.save_metrics("train", metrics)

    # ######## add a breakpoint for debug?
    # torch.distributed.breakpoint(rank=0)

    trainer.save_state()
    
    # print(f'\n        >>>>>> End of process [Rank {training_args.local_rank}] <<<<<<')
    logger.critical(f"End of process: [Rank {training_args.local_rank}")
    

if __name__ == "__main__":
    main()
