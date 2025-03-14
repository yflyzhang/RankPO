import os
import sys
import json
import logging
from typing import Optional, Union, List

# import wandb
import numpy as np
from datetime import datetime

from transformers import HfArgumentParser

from modeling import ModelForInference
from arguments import EvaluateArguments
from utils import create_faiss_index, faiss_search, compute_metrics



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



########################################
# File path to store evaluation results
########################################
def get_save_path(
    model_path, 
    output_dir,
    can_overwrite=True,
    file_type='json',
):
    """
    Get file path to save the evaluation results
    """
    assert output_dir is not None, "output_dir is empty!"

    # Use model name as prefix
    path_segs = os.path.normpath(model_path).split(os.sep)
    if len(path_segs)>=2 and path_segs[-1].startswith('checkpoint-'):
        
        # For "models/model-xxx/checkpoint-xx", "output_dir = test_results",
        # use "test_results/model-xxx/checkpoint-xxx.json" as the name
        # model_prefix = '_'.join(path_names[-2:])
        output_dir = os.path.join(output_dir, path_segs[-2])
        result_filename = f"{path_segs[-1]}.{file_type}"
        
    else:
        # For "models/model-xxx", use "test_results/model-xxx/main.json" as the name
        # model_prefix = os.path.basename(model_path.rstrip('/'))
        output_dir = os.path.join(output_dir, path_segs[-1])
        result_filename = f"main.{file_type}"      # final/main model (could be the last or the best checkpoint)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, result_filename)
    
    # Add time stamp to avoid overwrite?
    if not can_overwrite and os.path.isfile(save_path):
        now = datetime.now()
        names = result_filename.rsplit('.', 1)
        result_filename = f"{names[0]}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.{names[1]}"
        save_path = os.path.join(output_dir, result_filename)
        logger.info(f"To avoid overwrite, create a new file: {save_path}")
    
    return save_path


###################################
# Get All Checkpoint Inside a Path
###################################
def get_all_checkpoint_paths(model_path):
    all_checkpoint_paths = []
    for (dirpath, dirnames, filenames) in os.walk(model_path):
        # Use CONFIG_NAME to indicate the model checkpoint path
        # TODO: use regular expression to support case like 'model-00001-of-00002.safetensors'
        # if CONFIG_NAME in filenames and (SAFE_WEIGHTS_NAME in filenames or WEIGHTS_NAME in filenames):
        if "config.json" in filenames:
            all_checkpoint_paths.append(dirpath)
    return all_checkpoint_paths





############################
# Main for evaluation
############################
def main():

    start_time = datetime.now()
    parser = HfArgumentParser([EvaluateArguments])
    args = parser.parse_args_into_dataclasses()[0]
    
    if args.log_level in log_levels:
        logger.setLevel(log_levels[args.log_level])
    
    logger.info(f"{'#'*20} Start {'#'*20}")    # add separators for better visualization
    logger.info(f"Evaluation arguments:\n  {args.to_json_string()}")

    assert os.path.isfile(args.query_data), f"file {args.query_data} not exists!"
    assert os.path.isfile(args.corpus_data), f"file {args.corpus_data} not exists!"

    # breakpoint()    # to debug here
    models_to_eval = []
    if args.evaluate_all_checkpoints:
        models_to_eval = get_all_checkpoint_paths(args.model_name_or_path)
    else:
        if os.path.isfile(args.model_name_or_path + '/config.json'):
            models_to_eval = [args.model_name_or_path]
    logger.info(f"Models to evaluate: {models_to_eval}\n")
    
    # Do nothing if no model is found
    if len(models_to_eval) == 0:
        logger.error("No checkpoint is not found!")
        return
    
    
    if args.wandb_project:      # wandb is enabled
        import wandb
        # Start a new wandb run to track this script
        run_name = None if args.run_name=='auto' else args.run_name
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            name=run_name,
        )


    # query, label data
    with open(args.query_data) as f:
        query = []
        labels = []
        for line in f:
            d1 = json.loads(line.strip())
            query.append(d1['query']['text'])
            labels.append(d1['positives']['index'])
    
    # corpus data
    with open(args.corpus_data) as f:
        corpus = []
        for line in f:
            d1 = json.loads(line.strip())
            corpus.append(d1['text'])
    
    logger.info(f"Evaluation dataset size: {len(query)}")
    logger.info(f"Evaluation corpus dataset size: {len(corpus)}\n")   
    
    # # breakpoint()
    # # return 
    
    res = {}
    for model_to_eval in models_to_eval:

        # Get save path and see if the evaluation is already done!
        # to avoid repetitive operation!
        save_path = get_save_path(
            model_path=model_to_eval,
            output_dir=args.output_dir,
            can_overwrite=True
        )
        if os.path.isfile(save_path) and not args.overwrite_output_dir:
            # raise ValueError(
            logger.warning(
                f"Output file [{save_path}] already exists. "
                "Skip current model!\n"
                "Consider so set `--overwrite_output_dir` to overwrite the output_dir when necessary."
            )
            # breakpoint()    # to debug here
            continue
        
        # breakpoint()    # to debug here

        logger.info(f"{'#'*20} Evaluate {'#'*20}")    # add separators for better visualization
        logger.info(f"{args=}\n")
        logger.info(f"Evaluate model at ['{model_to_eval}']")
        logger.info(f"Load model: {model_to_eval}")
        model = ModelForInference(
            model_to_eval,
            attn_implementation=args.attn_implementation,
            use_fp16=args.fp16,
            use_bf16=args.bf16,
            device=args.device,
        )
        model.eval()
        # logger.info(f"Model: {model}")
        logger.info(f"model.device: {model.device}\n")

        batch_size = args.batch_size
        max_query_length = args.max_query_length
        max_passage_length = args.max_passage_length

        # Encoding corpus and queries
        logger.info(f'Embed query ({len(query)=}, {batch_size=}, {max_query_length=}):')
        query_embedding = model.encode(query, batch_size=batch_size, max_length=max_query_length)

        logger.info(f'Embed corpus ({len(corpus)=}, {batch_size=}, {max_passage_length=}):')
        corpus_embedding = model.encode(corpus, batch_size=batch_size, max_length=max_passage_length)
        
        # Crete faiss index
        logger.info(f"Create faiss index:")
        faiss_index = create_faiss_index(corpus_embedding)
        
        # Faiss search
        logger.info(f"Faiss search:")
        all_scores, all_indices = faiss_search(
            faiss_index, 
            query_embedding,
            topk=args.k, 
            batch_size=args.batch_size, 
        )
        
        # breakpoint()    # to debug here
        
        # Compute metrics
        logger.info(f"Compute metrics:")
        cutoffs = args.cutoffs
        if isinstance(cutoffs, str):
            cutoffs = [int(c.strip()) for c in cutoffs.split(',')]
        metrics = compute_metrics(all_indices, all_scores, labels, cutoffs=cutoffs)
        
        print("Evaluation results:")
        print("\n".join(f"    {k:15} {v}" for k, v in metrics.items()))
        # print(json.dumps(metrics, indent=4, sort_keys=True))
        
        # Save evaluation results
        # Move it forward! to avoid repetitive operation!
        # save_path = get_save_path(
        #     model_path=model_to_eval,
        #     output_dir=args.output_dir,
        #     overwrite=False
        # )
        with open(save_path, 'w') as f:
            # f.write(json.dumps(metrics, indent=4, sort_keys=True))
            f.write(json.dumps(metrics, indent=4))
            logger.info(f"Evaluation results saved at {save_path}")
        
        # breakpoint()

        # Save predicted indices and scores
        fpath = save_path.rsplit('.', 1)[0] + '-indices.npy'
        with open(fpath, 'wb') as f:
            np.save(f, all_indices)
        fpath = save_path.rsplit('.', 1)[0] + '-scores.npy'
        with open(fpath, 'wb') as f:
            np.save(f, all_scores)
        
        # Record to results
        name = os.path.basename(save_path).split('.')[0]
        res[name] = metrics
        
        # --------------------------
        # Log eval results via WandB
        # --------------------------
        if args.wandb_project:
            # log charts
            wandb.log(metrics)
            # log table
            table = wandb.Table(data=list(metrics.items()), columns=["metric", "value"])
            wandb.log({'histogram': wandb.plot.bar(table, "metric", "value", title="title")})
        
        
        time_elapsed = datetime.now() - start_time
        logger.info(f"Time elapsed for evaluation: {time_elapsed}")
        logger.info(f"{'#'*20}  End  {'#'*20}\n\n")
    
    # Save all the results in one file
    if len(res) == len(models_to_eval):
        logger.info(f"{'#'*20}  Save All Evals  {'#'*20}\n\n")
        all_save_path = os.path.join(save_path.rsplit(os.sep, 1)[0], 'all_eval_results.json')
        with open(all_save_path, 'w') as f:
            f.write(json.dumps(res, indent=4))
            logger.info(f"All evaluation results saved at {all_save_path}")
        
    time_elapsed = datetime.now() - start_time
    logger.info(f"Total time elapsed for all evaluations: {time_elapsed}")
    logger.info(f"{'#'*20}  End of All Evals  {'#'*20}\n\n")

        
if __name__ == "__main__":
    main()
