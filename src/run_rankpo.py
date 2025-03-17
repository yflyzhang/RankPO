#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
from typing import Optional, Literal
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)


from rankpo_trainer import RankPOTrainer
# from simpo_config import SimPOConfig
from arguments import (
    ModelArguments, TrainDataArguments, TrainArguments, RankPOArguments
)
from data_utils import RankPODataCollatorWithPadding


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, TrainDataArguments, RankPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Pass max_query_length and max_passage_length to training_args
    training_args.max_query_length = data_args.max_query_length
    training_args.max_passage_length = data_args.max_passage_length
    training_args.dataset_num_proc = data_args.dataset_num_proc

    #######
    # Setup
    #######
    logging.basicConfig(
        # format="[%(asctime)s] [%(levelname)s]  %(message)s",
        # format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] >> %(message)s",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # logging.basicConfig(
    #     format="[%(asctime)s] [%(levelname)s]  %(message)s",
    #     # format="[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s",
    #     # format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    #     level=logging.WARNING,
    # )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    # logger.info(f"Model parameters {model_args}")
    # logger.info(f"Data parameters {data_args}")
    # logger.info(f"Training/evaluation parameters {training_args}")
    
    is_distributed = bool(training_args.local_rank != -1)
    logger.critical(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {is_distributed}, 16-bits training: {training_args.fp16}"
    )
    
    logger.info(f"Model arguments:\n  {model_args.to_json_string()}")
    logger.info(f"Data arguments:\n  {data_args.to_json_string()}")
    logger.info(f"Training arguments:\n  {training_args.to_json_string()}")
    
    # # Check for last checkpoint
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        # last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None or (
            os.path.isfile(training_args.output_dir + '/config.json') 
            and os.path.isfile(training_args.output_dir + '/tokenizer.json')
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and contains trained checkpoints. "
                "Consider so set `--overwrite_output_dir` to overwrite the output_dir when necessary."
            )
    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ##########################
    # Load model and tokenizer
    ##########################
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # create `ref_model` when necessary
    ref_model = None
    if not training_args.reference_free:
        ref_model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=model_args.attn_implementation,
            use_cache=False if training_args.gradient_checkpointing else True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer, # use fast tokenizer?
    )
    
    # Add pad token if not specified
    if not tokenizer.pad_token:
        # # case 1: use eos_token as the pad_token [not recommended]
        # tokenizer.pad_token = tokenizer.eos_token
        
        # case 2: use the reserved token '<|finetune_right_pad_id|>' (in llama3.2) as the pad_token
        tokenizer.pad_token = '<|finetune_right_pad_id|>'
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|finetune_right_pad_id|>')
        
        logger.warning(f"set: {tokenizer.pad_token=}, {tokenizer.pad_token_id=}")
        
        # set model.config as well
        if not model.config.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.warning(f"set: {model.config.pad_token_id=}")
        
    # Add special tokens when necessary
    special_tokens_dict = {
        'additional_special_tokens': [
            '<keyword>', '</keyword>', 
            '<title>', '</title>', 
            '<abstract>', '</abstract>', 
            '<sep>',
        ]
    }

    # Add special tokens and resize token embeddings when necessary
    if not all(x in tokenizer.all_special_tokens for x in special_tokens_dict['additional_special_tokens']):
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        # `num_added_toks` should be 7?
    
    # if training_args.local_rank == 0:   # print rank0 only
    # print("====================================")
    logger.info(f"Special tokens:\n  {tokenizer.special_tokens_map=}")
    logger.info(f"Token embeddings:\n  {model.get_input_embeddings()}")
    logger.info(f'Model config:\n  {model.config}')
    # print("====================================")

    ###############
    # Load datasets
    ###############
    train_dataset = datasets.load_dataset('json', data_files=data_args.train_data, split='train')
    
    ###############
    # Data collator
    ###############
    data_collator=RankPODataCollatorWithPadding(
        pad_token_id = tokenizer.pad_token_id,
    )
    
    # ### >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)
    
    #########################
    # Instantiate trainer
    #########################
    trainer = RankPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # peft_config=get_peft_config(model_args),
    )

    # ### >>>>> add a breakpoint for debug? <<<<<
    # torch.distributed.breakpoint(rank=0)


    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # # Save everything else on main process
    # kwargs = {
    #     "finetuned_from": model_args.model_name_or_path,
    #     "dataset": list(data_args.dataset_mixer.keys()),
    #     "dataset_tags": list(data_args.dataset_mixer.keys()),
    #     "tags": ["alignment-handbook"],
    # }
    # if trainer.accelerator.is_main_process:
    #     trainer.create_model_card(**kwargs)
    #     # Restore k,v cache for fast inference
    #     trainer.model.config.use_cache = True
    #     trainer.model.config.save_pretrained(training_args.output_dir)

    # ##########
    # # Evaluate
    # ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(raw_datasets["test"])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()