import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.nn.utils.rnn import pad_sequence


############################
# Contrastive data collator
############################
# Note: We can custom the data collator, no need to inherit from `DataCollatorWithPadding`:
# https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/data/data_collator.py#L271

@dataclass
class ContrastiveDataCollatorWithPadding:
    """
    Wrapper that groups query and passage and does padding then.
    Note: Inputs are list or dict of tensors. Tokenization is done before calling this class!
    """
    
    pad_token_id: int = 0
    num_negatives: int = 5
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''features: list of row data'''
        query_input_ids_to_pad = []
        query_attention_mask_to_pad = []
        passage_input_ids_to_pad = []
        passage_attention_mask_to_pad = []

        # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        # torch.distributed.breakpoint(rank=1)

        for row in features:
            # Remember to convert to tensor before padding
            # 1.Query
            query_input_ids_to_pad.append(torch.tensor(row['query']['input_ids'], dtype=torch.long))
            query_attention_mask_to_pad.append(torch.tensor(row['query']['attention_mask'], dtype=torch.long))
            
            # 2.Passage = postive + negative
            # 2.1 One positive  
            sample_id = random.choice(range(len(row['positives']['input_ids'])))     # randomly choose one positive
            passage_input_ids_to_pad.append(torch.tensor(row['positives']['input_ids'][sample_id], dtype=torch.long))
            passage_attention_mask_to_pad.append(torch.tensor(row['positives']['attention_mask'][sample_id], dtype=torch.long))

            # 2.2 Several negatives
            # sample_ids = list(range(len(row['negatives']['input_ids'])))      # for test
            sample_ids = random.sample(range(len(row['negatives']['input_ids'])), self.num_negatives)     # randomly choose several negatives
            for i in sample_ids:
                tensor = torch.tensor(row['negatives']['input_ids'][i], dtype=torch.long)
                passage_input_ids_to_pad.append(tensor)
                tensor = torch.tensor(row['negatives']['attention_mask'][i], dtype=torch.long)
                passage_attention_mask_to_pad.append(tensor)
            
        # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        # torch.distributed.breakpoint(rank=1)

        # Padding
        padded_batch = {}
        padded_batch['query'] = {
            'input_ids': pad_sequence(query_input_ids_to_pad, batch_first=True, padding_value=self.pad_token_id),   # [batch_size, max_query_length]
            'attention_mask': pad_sequence(query_attention_mask_to_pad, batch_first=True, padding_value=0)
        }
        
        padded_batch['passage'] = {
            'input_ids': pad_sequence(passage_input_ids_to_pad, batch_first=True, padding_value=self.pad_token_id), # [batch_size * (num_positives + num_negatives), max_passage_length]
            'attention_mask': pad_sequence(passage_attention_mask_to_pad, batch_first=True, padding_value=0)
        }
        
        # >>>>> add a breakpoint for debug? <<<<<
        # torch.distributed.breakpoint(rank=0)
        # torch.distributed.breakpoint(rank=1)

        return padded_batch



############################
# Contrastive data collator
############################
# Tokenize and padding

# @dataclass
# class TrainDataCollatorWithTokenizingAndPadding:
#     r"""
#     Wrapper that tokenizes query and passage and does padding then.
#     Note: Inputs are texts, and tokenization is done in this class!
#     """
    
#     tokenizer: 'PreTrainedTokenizer'
#     padding: bool = True
#     truncation: bool = True
#     max_query_length: int = 32
#     max_passage_length: int = 128
#     return_tensors: str = "pt"
    
#     def __call__(self, features):
#         # (query, passage): features[0] is the query; features[1] is the passages (positve + negative)
#         query = [feat[0] for feat in features]
#         passage = [feat[1] for feat in features]

#         if isinstance(query[0], list):
#             query = sum(query, [])
#         if isinstance(passage[0], list):
#             passage = sum(passage, [])
        
#         # generate tokenized batch
#         batch = {}
#         batch['query'] = self.tokenizer(
#             query,
#             padding=self.padding,
#             truncation=self.truncation,
#             max_length=self.max_query_length,
#             return_tensors=self.return_tensors,
#         )
#         batch['passage'] = self.tokenizer(
#             passage,
#             padding=self.padding,
#             truncation=self.truncation,
#             max_length=self.max_passage_length,
#             return_tensors=self.return_tensors,
#         )
#         return batch


############################
# RankPO data collator
############################
@dataclass
class RankPODataCollatorWithPadding:

    r"""
    DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        mask_pad_token_id (`int`, defaults to 0):
            The pad_token_id used for attention masking.
    
    Examples:
    ```
    >>> collator = DataCollatorForPreference(pad_token_id=128004, mask_pad_token_id=0)
    >>> examples = [
        {
            "query": {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]},
            "chosen": {'input_ids': [4, 5], 'attention_mask': [1, 1]}, 
            "rejected": {'input_ids': [6], 'attention_mask': [1]}
        },
        {
            "query": {'input_ids': [7, 8], 'attention_mask': [1, 1]},
            "chosen": {'input_ids': [9], 'attention_mask': [1]}, 
            "rejected": {'input_ids': [10, 11, 12], 'attention_mask': [1, 1, 1]}
        },
    ]
    >>> collator(examples)
    {'query': {'input_ids': tensor([[     1,      2,      3],
                                    [     7,      8, 128004]]),
               'attention_mask': tensor([[1, 1, 1],
                                         [1, 1, 0]])},
    'passage': {'input_ids': tensor([[     4,      5, 128004],
                                     [     6, 128004, 128004],
                                     [     9, 128004, 128004],
                                     [    10,     11,     12]]),
                'attention_mask': tensor([[1, 1, 0],
                                          [1, 0, 0],
                                          [1, 0, 0],
                                          [1, 1, 1]])}}
    ```
    """
    
    pad_token_id: int = 0
    # mask_pad_token_id: int = 0
    # label_pad_token_id: int = -100
    # return_tensors: str = "pt"
    keys = ['query', 'chosen', 'rejected']
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        
        for k in self.keys:
            assert k in features[0].keys(), f"key: '{k}' is missing."
        
        # 1. Query
        query_input_ids_to_pad = [torch.LongTensor(row['query']['input_ids']) for row in features]
        query_attention_mask_to_pad = [torch.LongTensor(row['query']['attention_mask']) for row in features]
        
        # 2. Chosen + Rejected: append chosen and rejected inputs
        passage_input_ids_to_pad = []
        passage_attention_mask_to_pad = []
        for row in features:
           # input_ids: chosen + rejected
           passage_input_ids_to_pad.append(torch.LongTensor(row['chosen']['input_ids']))
           passage_input_ids_to_pad.append(torch.LongTensor(row['rejected']['input_ids']))
           
           # attention_mask: chosen + rejected
           passage_attention_mask_to_pad.append(torch.LongTensor(row['chosen']['attention_mask']))
           passage_attention_mask_to_pad.append(torch.LongTensor(row['rejected']['attention_mask']))

        # Padded batch
        padded_batch = {}
        padded_batch['query'] = {
            'input_ids': pad_sequence(query_input_ids_to_pad, batch_first=True, padding_value=self.pad_token_id),   # [batch_size, max_query_length]
            'attention_mask': pad_sequence(query_attention_mask_to_pad, batch_first=True, padding_value=0)
        }
        
        padded_batch['passage'] = {
            'input_ids': pad_sequence(passage_input_ids_to_pad, batch_first=True, padding_value=self.pad_token_id), # [batch_size * (num_positives + num_negatives), max_passage_length]
            'attention_mask': pad_sequence(passage_attention_mask_to_pad, batch_first=True, padding_value=0)
        }

        return padded_batch

