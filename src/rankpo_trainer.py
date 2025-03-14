# Copyright 2025 The HuggingFace Team. All rights reserved.
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


# Reference code:
#   https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/dpo_trainer.py
#   https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py


import inspect
import warnings
from functools import wraps
from copy import deepcopy
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, Mapping

from datasets import Dataset
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed
from accelerate import PartialState
from transformers import (
    DataCollator, 
    PreTrainedModel, 
    PreTrainedTokenizerBase, 
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput


from transformers.utils import is_peft_available
from transformers.integrations import is_wandb_available

from trl.trainer.utils import (
    disable_dropout_in_model,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


from arguments import RankPOArguments


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


class RankPOTrainer(Trainer):
    r"""
    Initialize RankPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        args (`RankPOArguments`):
            The RankPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "rankpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[RankPOArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):

        if isinstance(model, str):
            raise ValueError(
                "Model should be specified and instantiated before Trainer is instantiated."
            )
        
        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id
        
        if tokenizer is None:
            raise ValueError("`tokenizer` must be specified to tokenize a RankPO dataset.")
        
        if data_collator is None:
            raise ValueError("`data_collator` must be specified.")
        
        self.generate_during_eval = args.generate_during_eval
        self.tokenizer = tokenizer
        self.reference_free = args.reference_free
        self.ref_model = ref_model  # could be None

        # TODO: check if we can use dropout in this task?
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
        
        if args.loss_type in ["hinge"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )
        
        self.beta = args.beta
        self.gamma_beta_ratio = args.gamma_beta_ratio
        # self.gamma = args.gamma
        self.temperature = args.temperature
        self.sft_weight = args.sft_weight
        self.rankpo_weight = args.rankpo_weight
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Dataset preparation
        train_dataset = self._prepare_dataset(train_dataset, tokenizer, args)
        if eval_dataset is not None:
            eval_dataset = self._prepare_dataset(eval_dataset, tokenizer, args)
        
        # ### >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)
        
        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
        
        # Prepare ref_model in distributed setting
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        
        
        # Set Wandb
        # in `__init__` in the main process (Rank 0 / `is_world_process_zero`)
        # TODO: by default, trainer will use WandbCallback, can improve the setting here
        if is_wandb_available() and args.wandb_project and args.process_index == 0:
            
            # import wandb
            # wandb.login()
            # Start a new wandb run to track this script
            run_name = None if args.run_name=='auto' else args.run_name
            wandb.init(
                # set the wandb project where this run will be logged
                project=args.wandb_project,
                name=run_name,
            )
            
            # In addition, by default, WandbCallback gets the project name from os.environ['WANDB_PROJECT'] or defaults to "huggingface"
            # see: https://github.com/huggingface/transformers/blob/v4.44.1/src/transformers/integrations/integration_utils.py#L834
            # project=os.getenv("WANDB_PROJECT", "huggingface")
            # So as an alternative, we can also set os.environ['WANDB_PROJECT'] to init a wandb project:
            # import os    # already imported by `from transformers.trainer import *`
            # if isinstance(self.args.wandb_project, str):
            #     os.environ['WANDB_PROJECT'] = self.args.wandb_project
    
    
    def _prepare_deepspeed(self, model: PreTrainedModel):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
    
    def _prepare_dataset(
            self,
            dataset: Dataset,
            tokenizer: PreTrainedTokenizerBase,
            args: RankPOArguments,
        ) -> Dataset:

        # ### >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        
        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            dataset = dataset.map(
                self.tokenize_row,
                # remove_columns=column_names,
                fn_kwargs={
                    "tokenizer": tokenizer, 
                    "max_query_length": args.max_query_length,
                    "max_passage_length": args.max_passage_length,
                },
                num_proc=args.dataset_num_proc,
                desc="Tokenizing",
            )
        
        return dataset
    
    def tokenize_row(self, row, tokenizer, max_query_length, max_passage_length):
        '''Tokenize a single row from a specific dataset'''
        tokenized_row = {}
        tokenized_row['query'] = tokenizer(row['query'], max_length=max_query_length, truncation=True)
        if row['preferred'] == 'A':
            chosen = row['passage1']
            rejected = row['passage2']

        elif row['preferred'] == 'B':
            chosen = row['passage2']
            rejected = row['passage1']
            
        else:
            raise ValueError(f"Format is not suported! Please provide a suitable format. {row=}")
        
        tokenized_row['chosen'] = tokenizer(chosen, max_length=max_passage_length, truncation=True)
        tokenized_row['rejected'] = tokenizer(rejected, max_length=max_passage_length, truncation=True)

        return tokenized_row
    
    # def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    #     """
    #     Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    #     """
    #     if isinstance(data, Mapping):
    #         return type(data)({k: self._prepare_input(v) for k, v in data.items()})
    #     elif isinstance(data, (tuple, list)):
    #         return type(data)(self._prepare_input(v) for v in data)
    #     elif isinstance(data, torch.Tensor):
    #         kwargs = {"device": self.args.device}   # TODO: use self.accelerator.device ?
    #         if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
    #             # NLP models inputs are int/uint and those get adjusted to the right dtype of the
    #             # embedding. Other models such as wav2vec2's inputs are already float and thus
    #             # may need special handling to match the dtypes of the model
    #             kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
    #         return data.to(**kwargs)
    #     return data
    
    def single_forward(
        self, model: nn.Module, inputs: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Run the given model on the given batch of inputs, using input_ids and attention_mask.
        
        Args:
            inputs: inputs containing `input_ids` and `attention_mask`.
            For example: {'input_ids': [], 'attention_mask': []}
        """
        outputs = model(**inputs, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        # Get sentence-level embedding by the CLS/last token.
        # For Llama, use the last non-pad token as the CLS token.
        # Get the position of the last token (non-pad token):
        sequence_lengths = attention_mask.argmin(-1) - 1
        sequence_lengths = sequence_lengths % attention_mask.shape[-1]  # prevent -1 by %
        # Get sentence embedding (last non-pad token)
        batch_size = attention_mask.shape[0]
        embeds = last_hidden_state[torch.arange(batch_size, device=attention_mask.device), sequence_lengths]
        
        # Normalize embedding, so that the inner product of two vectors is equivalent to cosine
        # if self.normalize_embeddings:
        embeds = F.normalize(embeds, dim=-1)
        return embeds
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Run the given model on the given batch of inputs, using the concatenated 
        chosen and rejected inputs together.
        
        We do this to avoid doing two forward passes for chosen and rejected respectively, 
        because it's faster.
        """
        
        # Get embeddings of query and chosen/rejected
        query_embeds = self.single_forward(model, batch['query'])
        passage_embeds = self.single_forward(model, batch['passage'])
        
        # Compute similarity
        batch_size = query_embeds.shape[0]
        group_size = passage_embeds.shape[0] // batch_size  # group_size = 2 for one chosen vs one rejected
        query_reps = query_embeds[:, None, :,]
        passage_reps = passage_embeds.reshape(batch_size, group_size, -1).transpose(-1, -2)
        # [batch_size, 1, embed_size] x ([batch_size, group_size, embed_size].transpose(1, 2)) -> [batch_size, 1, group_size] -> squeeze(1)
        # dim(scores): [batch_size, group_size]
        # For normalized embeddings, inner product is equal to cosine similarity
        scores = torch.matmul(query_reps, passage_reps).squeeze(1)
        
        return scores
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the RankPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        
        # model: chosen and rejected
        scores = self.concatenated_forward(model, batch)    # scores may have be adjusted by temperature
        # logps = scores.log_softmax(dim=-1)  # log_softmax
        # chosen_logps = logps[:, 0]          # [chosen, rejected]
        # rejected_logps = logps[:, 1]
        # simplified version: cancel log_softmax? # TODO
        chosen_scores = scores[:, 0]
        rejected_scores = scores[:, 1]
        
        # ref_model: chosen and rejected
        ref_chosen_scores, ref_rejected_scores = 0, 0
        if self.ref_model is not None:
            # ref_scores = self.compute_ref_log_probs(batch)
            with torch.inference_mode():
                ref_scores = self.concatenated_forward(self.ref_model, batch)   # ref_scores may have be adjusted by temperature
                # ref_logps = ref_scores.log_softmax(dim=-1)  # log_softmax
                # ref_chosen_logps = ref_logps[:, 0]          # [chosen, rejected]
                # ref_rejected_logps = ref_logps[:, 1]
                # simplified version: cancel log_softmax? # TODO
                ref_chosen_scores = ref_scores[:, 0]
                ref_rejected_scores = ref_scores[:, 1]
        
        # ### >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        
        loss = 0    # to add loss

        # RankPO loss
        if self.rankpo_weight > 0.0:
            losses = self.rankpo_loss(
                chosen_scores,
                rejected_scores,
                ref_chosen_scores,
                ref_rejected_scores,
            )

            rankpo_loss = losses.mean()
            loss += self.rankpo_weight * rankpo_loss

            metrics[f"{prefix}rankpo_loss"] = self.accelerator.gather_for_metrics(rankpo_loss).detach().mean().item()
        
        # sft/supervised contrastive learning loss
        if self.sft_weight > 0.0:
            temp_scores = scores / self.temperature     # scaled by temperature
            target = torch.zeros(temp_scores.size(0), device=temp_scores.device, dtype=torch.long)
            loss_func = nn.CrossEntropyLoss()
            sft_loss = loss_func(temp_scores, target)
            loss += self.sft_weight * sft_loss
            metrics[f"{prefix}sft_loss"] = self.accelerator.gather_for_metrics(sft_loss).detach().mean().item()
        
        
        device = self.accelerator.device
        chosen_rewards = self.beta * (chosen_scores-ref_chosen_scores).to(device).detach()
        rejected_rewards = self.beta * (rejected_scores-ref_rejected_scores).to(device).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()

        metrics[f"{prefix}scores/chosen"] = self.accelerator.gather_for_metrics(chosen_scores).detach().mean().item()
        metrics[f"{prefix}scores/rejected"] = self.accelerator.gather_for_metrics(rejected_scores).detach().mean().item()
        metrics[f"{prefix}scores/margins"] = self.accelerator.gather_for_metrics(chosen_scores-rejected_scores).detach().mean().item()
        
        return loss, metrics
    

    def rankpo_loss(
        self,
        chosen_scores: torch.FloatTensor,
        rejected_scores: torch.FloatTensor,
        ref_chosen_scores: Union[None, torch.FloatTensor],
        ref_rejected_scores: Union[None, torch.FloatTensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the RankPO loss for a batch of policy model log probabilities.
        
        Args:
            chosen_scores: Chosen probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            rejected_scores: Rejected probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            ref_chosen_scores:
            ref_rejected_scores:

        Returns:
            The losses tensor: Contains the RankPO loss for each example in the batch.
        """
        
        # Get the advantages of chosen against rejected
        advantanges = chosen_scores - rejected_scores
        if not self.reference_free:     # with reference model    
            ref_advantages = ref_chosen_scores - ref_rejected_scores
            advantanges -= ref_advantages
        
        advantanges /= self.temperature    # scaled by temperature?
        advantanges = advantanges.to(self.accelerator.device)

        # logits = advantanges - self.gamma       # TODO: add gamma?
        logits = advantanges - self.gamma_beta_ratio    # note: use gamma_beta_ratio instead of real gamma

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )
        
        return losses
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        
        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss


    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):

        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)


    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "rankpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)