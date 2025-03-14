import random
import numpy as np
from tqdm import tqdm
import faiss
from contextlib import contextmanager

import torch
from accelerate import PartialState

from sklearn.metrics import roc_auc_score, roc_curve, ndcg_score



def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Ref: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer_utils.py#L92
    
    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    if deterministic:
        torch.use_deterministic_algorithms(True)
    


########################
# Create FAISS index
########################
def create_faiss_index(embeddings):
    
    # note: faiss only accepts float32
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[-1]

    # create faiss index
    faiss_index = faiss.IndexFlatIP(dim)    # similarity search (Inner Product, defaults to cosine)
    # faiss_index = faiss.IndexFlatL2(dim)    # distance search (not suitable here)
    
    faiss_index.add(embeddings)
    # print(f"#Total in faiss index: {faiss_index.ntotal}")

    return faiss_index



########################
# Batched FAISS search
########################
def faiss_search(
    faiss_index, 
    query_embedding,                      # embedding of query
    topk: int = 100,            # return topk
    batch_size: int = 256
):
    """
    Faiss KNN Search
    """
    
    all_scores, all_indices = [], []
    for i in tqdm(range(0, len(query_embedding), batch_size), desc="Faiss search"):
        batch_embeds = query_embedding[i: i + batch_size]
        batch_scores, batch_indices = faiss_index.search(
            batch_embeds.astype(np.float32),
            k=topk
        )
        all_scores.append(batch_scores)
        all_indices.append(batch_indices)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices



####################
# Compute metrics
####################
def compute_metrics(
    preds, 
    preds_scores, 
    labels, 
    cutoffs=[1, 5, 10, 20, 100]
):
    """
    Compute MRR/Recall/AUC/nDCG at cutoffs.
    """
    assert len(preds) == len(labels), 'shape not match for predictions and labels'
    if any(len(x) < max(cutoffs) for x in preds):
        print(f'Warning: No enough predictions for some cutoffs, e.g. cutoff {max(cutoffs)}')
    
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        label = set(label)    # `set` is more faster than `list` in terms of `in`
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for j, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[j] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr
    
    # Recall
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for i, cutoff in enumerate(cutoffs):
            # recall = np.intersect1d(label, pred[:cutoff])
            # recalls[i] += len(recall) / len(label)
            common = np.intersect1d(label, pred[:cutoff])
            recalls[i] += len(common) / max(min(cutoff, len(pred), len(label)), 1)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall
    
    # naive AUC 
    pred_hard_encodings = []
    for pred, label in zip(preds, labels):
        pred_hard_encoding = np.isin(pred, label).astype(int).tolist()
        pred_hard_encodings.append(pred_hard_encoding)
    pred_hard_encodings = np.asarray(pred_hard_encodings)

    for i, cutoff in enumerate(cutoffs):
        # print(cutoff)
        pred_hard_encodings1d = pred_hard_encodings[:, :cutoff].flatten() 
        preds_scores1d = preds_scores[:, :cutoff].flatten()
        # auc = roc_auc_score(pred_hard_encodings[:, :cutoff], preds_scores[:, :cutoff])
        auc = roc_auc_score(pred_hard_encodings1d, preds_scores1d)
        metrics[f"AUC@{cutoff}"] = auc
    
    # nDCG
    for k, cutoff in enumerate(cutoffs):
        nDCG = ndcg_score(pred_hard_encodings, preds_scores, k=cutoff)
        metrics[f"nDCG@{cutoff}"] = nDCG
            
    return metrics





###################################
# Get split_between_processes
###################################
# Adapted from:
# https://github.com/huggingface/accelerate/blob/v1.1.0/src/accelerate/state.py#L389

@contextmanager
def split_between_processes(
    inputs: list | tuple | dict | torch.Tensor, 
    apply_padding: bool = False,
    evenly_split: bool = False,     # customed behavior
):
    """
    Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
    distributed inference, such as with different prompts.


    Note that when using a `dict`, all keys need to have the same number of elements.


    Args:
        inputs (`list`, `tuple`, `torch.Tensor`, `dict` of `list`/`tuple`/`torch.Tensor`, or `datasets.Dataset`):
            The input to split between processes.
        apply_padding (`bool`, `optional`, defaults to `False`):
            Whether to apply padding by repeating the last element of the input so that all processes have the same
            number of elements. Useful when trying to perform actions such as `gather()` on the outputs or passing
            in less inputs than there are processes. If so, just remember to drop the padded elements afterwards.
        
        evenly_split (`bool`, `optional`, defaults to `False`):
            Whther to split the inputs as evenly as possible.


    Example:

    ```python
    # Assume there are four processes
    from accelerate import PartialState

    state = PartialState()
    inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    with state.split_between_processes(inputs, evenly_split=False) as inputs:
        print(inputs)
    # Process 0:    [1, 2, 3]
    # Process 1:    [4, 5, 6]
    # Process 2:    [7, 8, 9]
    # Process 3:    [9, X, X]   # this is better to remove padded index
    
    
    with state.split_between_processes(inputs, evenly_split=True) as inputs:
        print(inputs)
    # Process 0:    [1, 2, 3]
    # Process 1:    [4, 5, X]
    # Process 2:    [6, 7, X]
    # Process 3:    [8, 9, X]   # this is harder to recover the original
    ```
    """
    
    state = PartialState()
    
    if state.num_processes == 1:
        yield inputs
        return
    length = len(inputs)
    # Nested dictionary of any types
    if isinstance(inputs, dict):
        length = len(inputs[list(inputs.keys())[0]])
        if not all(len(v) == length for v in inputs.values()):
            raise ValueError("All values in the dictionary must have the same length")
    
    if evenly_split:
        # evenly split as much as possible, and pad many last devices
        num_samples_per_process, num_extras = divmod(length, state.num_processes)
        start_index = state.process_index * num_samples_per_process + min(state.process_index, num_extras)
        end_index = start_index + num_samples_per_process + (1 if state.process_index < num_extras else 0)
        
    else:
        # split to first few devices, and pad only the last one/few devices
        # num_samples_per_process = math.ceil(length / state.num_processes) 
        num_samples_per_process = (length + state.num_processes - 1) // state.num_processes  # ceil
        start_index = state.process_index * num_samples_per_process
        end_index = start_index + num_samples_per_process

    def _split_values(inputs, start_index, end_index):
        if isinstance(inputs, (list, tuple, torch.Tensor)):
            if start_index >= len(inputs):
                result = inputs[-1:]
            else:
                result = inputs[start_index:end_index]
            if apply_padding:
                if isinstance(result, torch.Tensor):
                    from accelerate.utils import pad_across_processes, send_to_device


                    # The tensor needs to be on the device before we can pad it
                    tensorized_result = send_to_device(result, state.device)
                    result = pad_across_processes(tensorized_result, pad_index=inputs[-1])
                else:
                    if evenly_split:
                        # result += [result[-1]] * (num_samples_per_process + 1 - len(result))
                        result += [inputs[-1]] * (num_samples_per_process + int(num_extras > 0) - len(result))
                    else:
                        result += [inputs[-1]] * (num_samples_per_process - len(result))

            return result
        elif isinstance(inputs, dict):
            for key in inputs.keys():
                inputs[key] = _split_values(inputs[key], start_index, end_index)
            return inputs
        else:
            from accelerate.utils import is_datasets_available
            if is_datasets_available():
                from datasets import Dataset


                if isinstance(inputs, Dataset):
                    if start_index >= len(inputs):
                        start_index = len(inputs) - 1
                    if end_index > len(inputs):
                        end_index = len(inputs)
                    result_idcs = list(range(start_index, end_index))
                    if apply_padding:
                        result_idcs += [end_index - 1] * (num_samples_per_process + 1 - len(result_idcs))
                    return inputs.select(result_idcs)
            return inputs


    yield _split_values(inputs, start_index, end_index)





