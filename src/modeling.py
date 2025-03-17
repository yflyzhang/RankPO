import logging
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.utils import ModelOutput as _ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput(_ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None



class ALLGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    Reference code:
        https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
        https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
        https://www.cnblogs.com/wolfling/p/15350067.html
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out



class AllGatherIntoTensorLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    Reference code:
        https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
        https://github.com/huggingface/accelerate/blob/v0.34.0/src/accelerate/utils/operations.py#L321
        https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
        https://www.cnblogs.com/wolfling/p/15350067.html
        
        # Basic reference:
        # all_tensors = torch.empty(
        #     self.world_size * tensor.numel(),
        #     dtype=tensor.dtype,
        #     device=tensor.device,
        # )
        # dist.all_gather_into_tensor(all_tensors, tensor)
        # return all_tensors.view(-1, *tensor.size()[1:])
    """
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.empty(
            # dist.get_world_size(), input.numel(), # size: [world_size, input.numel]
            dist.get_world_size() * input.numel(),
            dtype=input.dtype,
            device=input.device,
        )
        dist.all_gather_into_tensor(output, input)
        
        # ######## add a breakpoint for debug?
        # torch.distributed.breakpoint(rank=0)

        return output
    
    @staticmethod
    def backward(ctx, grads):
        
        (input,) = ctx.saved_tensors
        
        # grad_out = torch.zeros_like(input)
        grad_out = torch.empty(
            input.numel(),
            dtype=input.dtype,
            device=input.device,
        )
        
        # ######## add a breakpoint for debug?
        # torch.distributed.breakpoint(rank=0)

        # grad_out[:] = grads[dist.get_rank()]
        grad_out[:] = grads.view(dist.get_world_size(), -1)[dist.get_rank()]
        # after view/reshape, grads.size : [world_size, input.dim1, input.dim2]
        
        # ######## add a breakpoint for debug?
        # torch.distributed.breakpoint(rank=0)

        return grad_out.view(input.size())  # grad_out.size should be equal to input.size
        # return grad_out
    


############################
# Model for Training
############################
class ModelForTraining(nn.Module):
    """
    Model for training
    
    Adapted from:
    https://github.com/FlagOpen/FlagEmbedding/blob/lm-cocktail/FlagEmbedding/baai_general_embedding/finetune/modeling.py
    """

    def __init__(
        self,
        model_name_or_path: str = None,
        *,
        attn_implementation: str = None,
        use_cache: bool = True,
        cache_dir: str = None,
        token: str = None,
        trust_remote_code: bool = False,
        normalize_embeddings: bool = True,
        use_inbatch_neg: bool = True,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
    ):
        """
        Args:
            model_name_or_path (`str`, defaults to `None`):
                Can be either:
                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing model weights saved using
                    [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
            attn_implementation (`str`, defaults to `None`):
                Attention implementation, e.g., `flash_attention_2`
            cache_dir (`str`, defaults to `None`):
                Where do you want to store the pretrained models downloaded from s3.
            use_cache (`bool`, defaults to `True`):
                Whether or not to use cache for generation.
                If `use_cache` is True, `past_key_values` are returned and can be used to speed up decoding.
            token (`str`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files, e.g., download meta-llama/Llama-3.2-1B.
            trust_remote_code (`bool`, defaults to `False`):
                Whether to trust the execution of code from datasets/models defined on the Hub.
            normalize_embeddings (`bool`, defaults to `True`):
                Whether or not to normalize embeddings.
            use_inbatch_neg (`bool`, defaults to `True`):
                Whether or not to use passages in the same batch as negatives.
            negatives_cross_device (`bool`, defaults to `False`):
                Whether or not to share negatives across devices.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to modulate the embeddings.
        """
        super().__init__()
        model_init_kwargs = dict(
            attn_implementation=attn_implementation,
            cache_dir=cache_dir,
            use_cache=use_cache,
            # Disable caching if gradient checkpointing is enabled (not compatible)
            token=token,
            trust_remote_code=trust_remote_code,
        )

        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            **model_init_kwargs
        )
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        self.normalize_embeddings = normalize_embeddings
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config
        
        if not normalize_embeddings:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normalize_embeddings:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normalize_embeddings=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
    
    def embed(self, inputs):
        """Return the sentence-level embeddings of inputs
        Args:
            inputs (Dict[torch.Tensor]):
                A dictionary containing input features like `input_ids` and `attention_mask`.                
                e.g., {'input_ids': [Tensor], 'attention_mask':[Tensor]},

                so that it can be called by `model(**inputs)`
        Returns:
            `[torch.Tensor]`: Encoded sentence embeddings.
        """
        if inputs is None:
            return None
        outputs = self.model(**inputs, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # get sentence-level embedding by the CLS/last token
        if 'Llama' in self.config.architectures[0]:     # for Llama, use the last non-pad token as the CLS token
            # get the position of the last token (non-pad token)
            sequence_lengths = attention_mask.argmin(-1) - 1
            sequence_lengths = sequence_lengths % attention_mask.shape[-1]  # prevent -1 by %
            # get sentence embedding (last non-pad token)
            batch_size = attention_mask.shape[0]
            embeds = last_hidden_state[torch.arange(batch_size, device=attention_mask.device), sequence_lengths]
        else:           # for BME-M3/XLMRoberta, use the first token which is the CLS token
            embeds = last_hidden_state[:, 0]
        
        # normalize embedding, so that the inner product of two vectors is equivalent to cosine
        if self.normalize_embeddings:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        
        return embeds.contiguous()
    
    def compute_similarity(self, q_reps, p_reps):
        """
        Compute the similarity between query and passage embedding vectors.
        Inner product, but is equivalent to cosine similarity when sentence embeddings are L2-normalized

        Args:
            q_reps (`torch.Tensor`): Query representations.
            p_reps (`torch.Tensor`): Passage representations.

        Returns:
            `torch.Tensor`: Similarity scores.
        """
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
    
    def forward(
        self, 
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
    ):
        """
        Note: Keywrod arguments are `query` and `passage` as specified by the dataloader.
        Don't change the name of `query` to other names like `query_inputs`.
        
        Args:
            query (Dict[torch.Tensor]):
                Inputs for query embeddings, 
                e.g., {'input_ids': [Tensor], 'attention_mask':[Tensor]}.
                so that it can be called by `model(**query)`
            passage (Dict[torch.Tensor]):
                Inputs for passage embeddings, 
                e.g., {'input_ids': [Tensor], 'attention_mask':[Tensor]}.
        Returns:
            `ModelOutput()` comprising of:
                q_reps (Optional[Tensor]): vector representation of query
                p_reps (Optional[Tensor]): vector representation of passage
                loss (Optional[Tensor]): loss value
                scores (Optional[Tensor]): similarity scores between query and passage
        """
        q_reps = self.embed(query)      # q_reps: [batch_size, hidden_dim] TODO:?
        p_reps = self.embed(passage)    # p_reps: [batch_size*group_size, hidden_dim] ?
        
        if self.training:
            
            # ######## add a breakpoint for debug?
            # torch.distributed.breakpoint(rank=0)
            # torch.distributed.breakpoint(rank=1)
            
            if self.negatives_cross_device and self.use_inbatch_neg:
                # use samples from other devices as negatives
                q_reps = self.distributed_gather(q_reps)
                p_reps = self.distributed_gather(p_reps)
            
            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature      # adjust by temperature
                # dim(scores): 
                #   1. negatives_cross_device=True:  n_gpu * [batch_size, batch_size*group_size]
                #   2. negatives_cross_device=False: [batch_size, batch_size*group_size]
                
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size    # index of each positive sample    
                
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1)
                scores = scores / self.temperature      # adjust by temperature
                # [batch_size, 1, embed_size] x ([batch_size, group_size, embed_size].transpose(-2, -1)) -> [batch_size, 1, group_size] -> squeeze(1)
                # dim(scores): [batch_size, group_size]
                
                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)    # positive: the first one
            
            # Compute the loss between scores and target.
            loss = self.criterion(scores, target)
        
            # ######## add a breakpoint for debug?
            # torch.distributed.breakpoint(rank=0)
            # torch.distributed.breakpoint(rank=2)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return ModelOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
    
    
    def distributed_gather(self, tensor: Optional[torch.Tensor], use_method:int = 1):
        """
        Gather tensor across distributed devices.
        
        Args:
            tensor_to_gather (`Optional[torch.Tensor]`): Tensor to gather.
            use_method (int): use which way to gather tensors
        
        Returns:
            `Optional[torch.Tensor]`: Gathered tensor.
        
        Reference:
            https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
            https://github.com/huggingface/accelerate/blob/v0.34.0/src/accelerate/utils/operations.py#L321
            https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
            https://www.cnblogs.com/wolfling/p/15350067.html
        """

        # if tensor is None:
        #     return
        
        # tensor is at least 1dim
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        
        # Can only gather contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        ######## Method 1: Use `dist.all_gather` ########
        """
        Note: 
        Without the line: `all_tensors[self.process_rank] = tensor`, error occurs: 
        "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        One possible reason: `dist.all_gather` itself does not propagate back the gradient!
        `all_tensors` contains all `tensor` from all GPUs. All tensors there have no grad_fn 
        except the one in the current GPU because of the setting: `all_tensors[self.process_rank] = tensor`。
        See also:
            https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
            https://www.cnblogs.com/wolfling/p/15350067.html
            https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
        """
        if use_method == 1:
            all_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(all_tensors, tensor)
            all_tensors[self.process_rank] = tensor
            return torch.cat(all_tensors, dim=0)
        
        
        ######## Method 2: Use `dist.all_gather` with the help of `torch.autograd.Function` ########
        """
        Use the wrapper `ALLGatherLayer` to propagate back the gradient while using `dist.all_gather`
        """
        if use_method == 2:
            all_tensors = ALLGatherLayer.apply(tensor)
            return torch.cat(all_tensors, dim=0)
        
        
        ######## TODO: Method 3: Use `dist.all_gather_into_tensor` [experimental] ########
        # `dist.all_gather_into_tensor` is thought to be more efficient than `dist.all_gather`
        # It's not supported for backend 'gloo'.
        if use_method == 3:
            # Basic reference:
            # all_tensors = torch.empty(
            #     self.world_size * tensor.numel(),
            #     dtype=tensor.dtype,
            #     device=tensor.device,
            # )
            # dist.all_gather_into_tensor(all_tensors, tensor)
            # return all_tensors.view(-1, *tensor.size()[1:])
            
            all_tensors = AllGatherIntoTensorLayer.apply(tensor)
            all_tensors = all_tensors.view(-1, *tensor.size()[1:])
            return all_tensors
        


############################
# Model for Inference
############################
class ModelForInference(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = None,
        attn_implementation: str = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        use_bf16: bool = False,
        device: int = 0
    ) -> None:
        """
        Args:
            model_name_or_path (`str`, defaults to `None`):
                Can be either:
                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing model weights saved using
                    [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
            attn_implementation (`str`, defaults to `None`):
                Attention implementation, e.g., `flash_attention_2`
            normalize_embeddings (`bool`, defaults to `True`):
                Whether or not to normalize embeddings.
            use_fp16 (`bool`, defaults to `False`):
                Whether to use half-precision (fp16) for faster inference if supported.
            use_bf16 (`bool`, defaults to `False`):
                Whether to use half-precision (bf16) for faster inference if supported.
            device (`str` | `torch.device`):
                specify the device to use. 
                Could be 'cuda', 'cuda:1', 'cpu', `torch.device(*)`
                # TODO: support multiple gpus, use `Accelerator.prepare`?
        """
        super().__init__()

        if use_bf16 and use_fp16:
            raise ValueError("Cannot use fp16 and bf16 in the same time!")
        
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        
        torch_dtype = torch.float32
        if use_fp16:
            torch_dtype=torch.float16
        elif use_bf16:
            torch_dtype=torch.bfloat16

        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.normalize_embeddings = normalize_embeddings
        self.config = self.model.config

        if not self.tokenizer.pad_token:
            raise ValueError("pad_token is not specified!")

        self.model = self.model.to(self.device)
    
    # Enable inference mode
    @torch.inference_mode()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        description: str = 'Encoding',
    ) -> Union[np.array, torch.Tensor]:
        """
        Args:
            sentences (Union[`str`, `List[str]`]):
                List of sentences or a single sentence to encode.
            batch_size (`int`, defaults to 256):
                The number of samples per batch for encoding.
            max_length (`int`, optional, defaults to 512):
                Maximum length for each sequence. Longer sequences will be truncated.
            convert_to_numpy (`bool`, optional, defaults to True):
                Whether to convert the output embeddings to a NumPy array.
        
        Returns:
            `Union[np.array, torch.Tensor]`: Encoded sentence embeddings.
        """
        self.model.eval()
        
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        
        if not isinstance(sentences[0], str):
            raise ValueError("Input items should be text.")
        
        all_embeddings = []
        for i in tqdm(
            range(0, len(sentences), batch_size), 
            desc=description,
        ):
            batch_sentences = sentences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state

            # Get sentence-level embedding by the CLS token
            attention_mask = inputs['attention_mask']
            if 'Llama' in self.config.architectures[0]:     # for Llama, use the last non-pad token as the CLS token
                # get the position of the last token (non-pad token)
                sequence_lengths = attention_mask.argmin(-1) - 1
                sequence_lengths = sequence_lengths % attention_mask.shape[-1]  # prevent -1 by %
                # get sentence embedding (last non-pad token)
                batch_size = attention_mask.shape[0]
                embeddings = last_hidden_state[torch.arange(batch_size, device=attention_mask.device), sequence_lengths]
            else:           # for BME-M3/XLMRoberta, use the first token which is the CLS token
                embeddings = last_hidden_state[:, 0]
            
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            if convert_to_numpy:
                if embeddings.dtype == torch.bfloat16:  # numpy doesnot support bf16, so upcast bf16 to float first
                    embeddings = embeddings.float()
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
            # Release memory to avoid OOM. 
            del inputs, embeddings
            torch.cuda.empty_cache()
        
        
        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

