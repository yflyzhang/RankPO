import os
from dataclasses import dataclass, field, fields
from typing import Optional, Union, List, Literal
import json

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments of model.
    """

    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )

    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use;"
                "you can use --attn_implementation=flash_attention_2,"
                "in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )

    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )

    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    
    # add to_json format support
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return json.dumps(d, indent=2)



@dataclass
class TrainDataArguments:
    """
    Arguments of train dataset
    """

    # Define a list of strings for multiple files
    train_data: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of train dataset file paths"}
    )
    
    num_negatives: int = field(
        default=5, 
        metadata={"help": "Number of negative sample size per query"}
    )
    
    max_query_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input query sequence length after tokenization for passage. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    
    max_passage_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input passage sequence length after tokenization for passage. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    
    dataset_num_proc: Optional[int] = field(
        default=8, metadata={"help": "The number of workers to use to tokenize the data. Defaults to None."}
    )

    # def __post_init__(self):
    #     if not os.path.exists(self.train_data):
    #         raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")
    
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return json.dumps(d, indent=2)



@dataclass
class TrainArguments(TrainingArguments):
    """
    Inherit from `transfomers.TrainingArguments`
    """
    # ------------
    # bf16 is defined in transformers.TrainingArguments
    # if (args.fp16 or args.bf16) and args.half_precision_backend == "auto":
    #     logger.info(f"Using {args.half_precision_backend} half precision backend")
    
    # bf16 is invoked by the following:
    # self.amp_dtype = torch.bfloat16
    # trainer.autocast_smart_context_manager
    # Trainer.compute_loss_context_manager
    # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer.py#L3484
    # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer.py#L673
    # ------------
    # bf16: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
    #             " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
    #         )
    #     },
    # )

    # debug: bool = field(
    #     default=False, metadata={"help": "Whether to enter debug mode. If True, breakpoints will work"}
    # )

    use_inbatch_neg: bool = field(
        default=True, metadata={"help": "Use passages in the same batch as negatives"}
    )
    
    negatives_cross_device: bool = field(
        default=True, metadata={"help": "Share negatives across devices"}
    )
    
    temperature: Optional[float] = field(
        default=0.02, metadata={"help": "Temperature used to adjust the distribution."}
    )

    normalize_embeddings: bool = field(
        default=True, metadata={"help": "Whether or not to normalize embeddings"}
    )
    
    wandb_project: str = field(
        default='huggingface',
        metadata={
            "help": (
                "wandb project name for logging."
                "For empty string(`''`), it will disable wandb (by overwriting `--report_to`)"
            )
        }
    )
    

    # `run_name` is already declared in `transformers.TrainingArguments`
    # Redeclare it here:
    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging. "
            "If not specified, will be the same as `output_dir`"
            "If specified as `'auto'`, run name will be automatically set by wandb"
        },
    )
    

    # report_to: Union[None, str, List[str]] = field(
    #     default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    # )
    # if self.report_to is None:
    #     logger.info(
    #         "The default value for the training argument `--report_to` will change in v5 (from all installed "
    #         "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
    #         "now. You should start updating your code and make this info disappear :-)."
    #     )
    #     self.report_to = "all"


    ## `log_level` for Rank 0 (main node),
    ## for replica nodes, should set `log_level_replica` (defaults to `warning`) to other log behaviors
    # log_level: Optional[str] = field(
    #     default="passive",
    #     metadata={
    #         "help": (
    #             "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
    #             " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
    #             " lets the application set the level. Defaults to 'passive'."
    #         ),
    #         "choices": trainer_log_levels.keys(),
    #     },
    # )
    
    # log_level_replica: Optional[str] = field(
    #     default="warning",
    #     metadata={
    #         "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
    #         "choices": trainer_log_levels.keys(),
    #     },
    # )



@dataclass
class EvaluateArguments:
    """
    Arguments for evaluation.
    """
    
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={'help': 'The model name or checkpointing path for encoding.'}
    )
    
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use;"
                "you can use --attn_implementation=flash_attention_2,"
                "in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )

    bf16: bool = field(
        default=False,
        metadata={'help': 'Use bf16 in inference?'}
    )
    
    device: Union[str, int] = field(
        default=None,    # 1 -> cuda:1
        metadata={'help': 'Device for inference'}
    )
    
    corpus_data: str = field(
        default=None,
        metadata={'help': 'candidate passages'}
    )
    query_data: str = field(
        default=None,
        metadata={'help': 'queries and their positive passages for evaluation'}
    )
    
    max_query_length: int = field(
        default=32,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=128,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    cutoffs: str = field(
        default="1,5,10,20,100",
        metadata={'help': 'Cutoffs to compute metric@k, e.g. MRR@k'}
    )
    
    save_index: bool = field(
        default=False,
        metadata={'help': 'Save faiss index at output_dir?'}
    )
    load_index: bool = field(
        default=False,
        metadata={'help': 'Load faiss index from output_dir?'}
    )
    output_dir: str = field(
        default="",
        metadata={'help': 'Path to save faiss index.'}
    )
    
    evaluate_all_checkpoints: bool = field(
        default=False,
        metadata={'help': 'Evaluate all model checkpoints inside the given model path?'}
    )
    
    log_level: Optional[str] = field(
        default="info",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: "
                "'debug', 'info', 'warning', 'error' and 'critical'"
            ),
        },
    )


    wandb_project: str = field(
        default='',
        metadata={
            "help": (
                "wandb project name for logging."
                "For empty string(`''`), it will disable wandb"
            )
        }
    )
    

    # `run_name` is already declared in `transformers.TrainingArguments`
    # Redeclare it here:
    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging. "
            "If not specified, will be the same as `output_dir`"
            "If specified as `'auto'`, run name will be automatically set by wandb"
        },
    )

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return json.dumps(d, indent=2)




@dataclass
class HardNegativeDataArguments:
    """
    Arguments to generate hard negative data
    """
    
    model_name_or_path: str = field(
        default=None,
        metadata={'help': 'The model name or checkpointing path for encoding.'}
    )
    
    input_file: str = field(
        default=None,
        metadata={
            'help': "The input file to mine hard negatives from."
            "This script will retrieve top-k documents for each query, "
            "and choose negatives by the specified strategy from the top-k documents"
            "(not including the positive documents)."
        }
    )
    
    output_file: str = field(
        default=None,
        metadata={
            'help': "The output file to store the mined hard negatives."
        }
    )

    output_prefix: str = field(
        default=None,
        metadata={
            'help': "Prefix of the output file to store the mined hard negatives."
        }
    )

    batch_size: int = field(
        default=32,
        metadata={'help': 'Inference batch size.'}
    )
    
    max_query_length: int = field(
        default=32,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=128,
        metadata={'help': 'Max passage length.'}
    )
   
    search_range: str = field(
        default="0-100",
        metadata={
            'help': "Range for negative samples are drawn from."
        }
    )

    device: Union[str, int] = field(
        default=0,    # e.g., 1 -> cuda:1
        metadata={'help': 'Device for inference. Useful for single card'}
    )

    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )

    bf16: bool = field(
        default=False,
        metadata={'help': 'Use bf16 in inference?'}
    )
    
    method: str = field(
        default=None,
        metadata={
            "help": "Method to sample hard negatives"
            "Should be one of the following:"
            "1. `topk`: use topk as hard negatives;"
            "2. `sample`: sampled from [search_range] as hard negatives"
            "3. `cluster`: use cluster information to guide hard negative selection"
        }
    )
    
    # use_gpu: bool = field(
    #     default=False,
    #     metadata={'help': 'Whether to use faiss-gpu for searching'}
    # )

    
    num_negatives: int = field(
        default=10,
        metadata={
            'help': "The number of negatives"
        }
    )

    num_clusters: int = field(
        default=10,
        metadata={
            'help': "The number of clusters"
        }
    )

    alpha: float = field(
        default=1.0,
        metadata={
            'help': "Probability/weight of choosing a negative sample, alpha*(lambda_**k)"
        }
    )

    lambda_: float = field(
        default=None,    # 1.0 corresponds to random sample
        metadata={
            'help': "Probability/weight of choosing a negative sample, alpha*(lambda_**k)"
        }
    )
    
    seed: int = field(
        default=42,
        metadata={
            'help': "Random seed"
        }
    )
    
    # index_factory: str = field(
    #     default="Flat",
    #     metadata={'help': 'Faiss index factory.'}
    # )
    
    log_level: Optional[str] = field(
        default="info",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: "
                "'debug', 'info', 'warning', 'error' and 'critical'"
            ),
        },
    )
    
    
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return json.dumps(d, indent=2)
    





@dataclass
class PredictionDataArguments:
    """
    Arguments to get top predictions
    """
    
    model_name_or_path: str = field(
        default=None,
        metadata={'help': 'The model name or checkpointing path for encoding.'}
    )
    
    input_file: str = field(
        default=None,
        metadata={
            'help': "The input file to generate predictions by the model."
            "This script will retrieve top-k documents for each query, "
            "and choose smaples by the specified strategy from the top-k documents"
        }
    )

    corpus_data: str = field(
        default=None,
        metadata={'help': 'candidate passages'}
    )
    query_data: str = field(
        default=None,
        metadata={'help': 'queries and their positive passages for evaluation'}
    )
    
    output_file: str = field(
        default=None,
        metadata={
            'help': "The output file to store the predicted candidates for each query."
        }
    )

    output_prefix: str = field(
        default=None,
        metadata={
            'help': "Prefix of the output file to store the results."
        }
    )

    batch_size: int = field(
        default=32,
        metadata={'help': 'Inference batch size.'}
    )
    
    max_query_length: int = field(
        default=32,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=128,
        metadata={'help': 'Max passage length.'}
    )
   
    search_range: str = field(
        default="0-100",
        metadata={
            'help': "Range for topk samples are drawn from."
        }
    )

    device: Union[str, int] = field(
        default=0,    # e.g., 1 -> cuda:1
        metadata={'help': 'Device for inference. Useful for single card'}
    )

    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )

    bf16: bool = field(
        default=False,
        metadata={'help': 'Use bf16 in inference?'}
    )
    
    method: str = field(
        default=None,
        metadata={
            "help": "Method to sample hard negatives"
            "Should be one of the following:"
            "1. `topk`: use topk;"
            "2. `sample`: sampled k from [search_range]"
        }
    )
    
    num_predictions: int = field(
        default=10,
        metadata={
            'help': "The number of predictions for each query"
        }
    )

    seed: int = field(
        default=42,
        metadata={
            'help': "Random seed"
        }
    )
    
    log_level: Optional[str] = field(
        default="info",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: "
                "'debug', 'info', 'warning', 'error' and 'critical'"
            ),
        },
    )
    
    
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return json.dumps(d, indent=2)




@dataclass
class RankPOArguments(TrainingArguments):
    """
    Subclass of `transfomers.TrainingArguments`
    """

    # overwrite_output_dir: already defined in `TrainingArguments`
    # overwrite_output_dir: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "Overwrite the content of the output directory. "
    #             "Use this to continue training if output_dir points to a checkpoint directory."
    #         )
    #     },
    # )
    
    # model_name_or_path: str = field(
    #     default="meta-llama/Llama-3.2-1B",
    #     metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    # )

    reference_free: bool = field(
        default=False, 
        metadata={"help": "If `True`, no reference mode used (like SimPO)."}
    )

    # train_data: str = field(
    #     default=None, metadata={"help": "Path to train data"}
    # )
    
    # num_negatives: int = field(
    #     default=5, 
    #     metadata={"help": "Number of negative sample size per query"}
    # )
    
    # `max_query_length` and `max_passage_length` have been declared in `TrainDataArguments`
    # max_query_length: int = field(
    #     default=32,
    #     metadata={
    #         "help": "The maximum total input query sequence length after tokenization for passage. "
    #                 "Sequences longer than this will be truncated, sequences shorter will be padded."
    #     },
    # )
    
    # max_passage_length: int = field(
    #     default=128,
    #     metadata={
    #         "help": "The maximum total input passage sequence length after tokenization for passage. "
    #                 "Sequences longer than this will be truncated, sequences shorter will be padded."
    #     },
    # )

    # label_smoothing: float = 0.0

    # use_inbatch_neg: bool = field(
    #     default=True, metadata={"help": "Use passages in the same batch as negatives"}
    # )
    
    # negatives_cross_device: bool = field(
    #     default=True, metadata={"help": "Share negatives across devices"}
    # )
    
    temperature: Optional[float] = field(
        default=0.02, metadata={"help": "Temperature used to adjust the distribution."}
    )

    beta: Optional[float] = field(
        default=1.0, metadata={"help": "/beta in loss function."}
    )

    # gamma: Optional[float] = field(
    #     default=0.0, metadata={"help": "/gamma in loss function."}
    # )

    gamma_beta_ratio: Optional[float] = field(
        default=0.0, 
        metadata={
            "help": "The ratio between the target reward margin (gamma) and beta in SimPO loss."
            "gamma_beta_ratio = gamma / beta"
            "gamma = gamma_beta_ratio * beta"
        }
    )

    sft_weight: Optional[float] = field(
        default=0.0, metadata={"help": "weight of sft/supervised contrastive learning in loss function."}
    )

    rankpo_weight: Optional[float] = field(
        default=1.0, metadata={"help": "weight of rankpo in loss function."}
    )

    loss_type: Literal["sigmoid", "hinge"] = field(
        default='sigmoid', metadata={"help": "The type of loss to use: ('sigmoid', 'hinge')"}
    )

    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing factor. This argument is required if you want to use the default data collator."}
    )

    generate_during_eval: bool = field(
        default=False, metadata={"help": "Whether to sample and log generations during evaluation step."}
    )

    normalize_embeddings: bool = field(
        default=True, metadata={"help": "Whether or not to normalize embeddings."}
    )

    disable_dropout: bool = field(
        default=True, metadata={"help": "Whether or not to disable dropouts in `model`."}
    )
    
    wandb_project: str = field(
        default='huggingface',
        metadata={
            "help": "wandb project name for logging."
            "For empty string(`''`), it will disable wandb (by overwriting `--report_to`)"
        }
    )
    
    # `run_name` is already declared in `transformers.TrainingArguments`
    # Redeclare it here:
    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging. "
            "If not specified, will be the same as `output_dir`"
            "If specified as `'auto'`, run name will be automatically set by wandb"
        },
    )





