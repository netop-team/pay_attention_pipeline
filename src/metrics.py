from abc import ABC, abstractmethod
import torch
from src.GUIDE import GUIDEModel
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from IPython.display import clear_output
from typing import Dict, Union
from typing_extensions import override
from tqdm import tqdm
from time import time
import gc

class BaseMetric(ABC):
    def __init__(
        self,
        base_model : AutoModel,
        tokenizer : AutoTokenizer,
        num_layers : int,
    ) -> None:

        self.attn_saver_model = GUIDEModel(
            base_model, 
            tokenizer,
            should_save_params=True
        )

        self.tokenizer = tokenizer
        self.num_layers : int = num_layers
        self.tokens = None

        self.reset()

    def reset(self):
        self.dp = {
            "influences":  {  layer : [] for layer in range(-1,self.num_layers)},
            "influences_heads":   {  head : [] for head in range(-1,self.num_layers)},
            "embeddings": {layer : [] for layer in range(-1,self.num_layers)},
            "outputs" :  {layer : [] for layer in range(-1,self.num_layers)}
        }

    def influence_of_sums(
        self,
        v1 : torch.Tensor,
        I1: float,
        v2 : torch.Tensor,
        I2 : float,
        p_norm : int = 1,

    ):
        n1 = torch.norm(v1, dim = 1, p = p_norm)\
            .pow(p_norm)
        n2 = torch.norm(v2, dim = 1, p = p_norm)\
            .pow(p_norm)

        return (n1*I1 + n2*I2)/(n1 + n2)

        

    @abstractmethod
    def compute_influence(
        self,
        layer : int,
        use_values : bool = False,
        p_norm : int = 1,
        **kwargs
    ):
        ...

    def __call__(
        self,
        text : str, 
        instruction : str,
        delta_attention : float,
        use_values : bool = False,
        *args: torch.Any, 
        **kwds: torch.Any
    ):
        self.attn_saver_model.remove_hooks()
        results = dict()
        self.attn_saver_model.set_delta_attention(delta_attention)

        messages = [
            {"role": "user", "content": text},
        ]

        template = self.tokenizer\
            .apply_chat_template(messages, tokenize = False)
        
        # return template
        
        splits = template.split(instruction)
        initial_prompt = splits[0]
        context = instruction.join(splits[1:])

        assert (hash(initial_prompt+instruction+context) == hash(template)), "Error in spliting strings. Initial and final string does not match"

        initial_tokens = self.tokenizer.encode(initial_prompt, return_tensors='pt', add_special_tokens = False)
        instruction_tokens = self.tokenizer.encode(instruction, return_tensors='pt', add_special_tokens = False)
        context_tokens = self.tokenizer.encode(context, return_tensors='pt', add_special_tokens = False)

        start_idx = initial_tokens.size(1)
        end_idx = start_idx + instruction_tokens.size(1)

        
        tokens = torch.concat([
            initial_tokens.squeeze(), 
            instruction_tokens.squeeze(),
            context_tokens.squeeze()
        ]).unsqueeze(0)

        self.tokens = tokens

        q = self.tokenizer.decode(tokens.squeeze()[start_idx: end_idx])

        assert instruction in q, "Error in tokenization. Not giving attention to correct tokens"


        self.attn_saver_model.set_reference_tokens(start_idx, end_idx)
        self.attn_saver_model.insert_hook()


        print(f"Studying influence to '{q}'")

        t0 = time()
        with torch.no_grad():
            self.attn_saver_model(tokens, output_attentions = True)
        t1 = time()

        token_index_in_text = torch.arange(start_idx, end_idx, step=1)

        # layer -1 is the initial input
        # computing influence before layer 0
        embedding : torch.Tensor = self.attn_saver_model\
            .internal_parameters[0]\
            ['raw_embedding']\
            .squeeze()\
            .to("cuda")

        influence_0 = torch.zeros(len(embedding))
        influence_0[token_index_in_text] = 1

        self.dp["influences"][-1] = torch.tensor(
            influence_0 ,
            dtype = embedding.dtype
        ).to("cpu")

        self.dp['embeddings'][-1] = embedding

        for layer in tqdm(range(0, self.num_layers, 1)):

            self.compute_influence(
                layer,
                use_values,
                p_norm =1,
                **kwds
            )

            
        self.dp.pop('embeddings')
        self.dp.pop("influences_heads")
        print("Passing tensors to CPU...")

        for layer in range(self.num_layers):
            self.dp['influences'][layer] = self.dp['influences'][layer].to("cpu")

        self.attn_saver_model.remove_hooks()
        self.attn_saver_model.reset_internal_parameters()
            
        results = self.dp
        self.reset()

        gc.collect()
        torch.cuda.empty_cache() 

        
        return results['influences']
    


class Influence(BaseMetric):
    def compute_influence(
        self, 
        layer : int,
        use_values : bool = False, 
        p_norm: int = 1,
        **kwargs
    ):
        values = self.attn_saver_model\
            .internal_parameters[layer]\
            ['value']\
            .squeeze()\
            .to("cuda")
                
        if not use_values:
            values = None
        
        attn_matrix : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]['avg_attention_heads']\
            .squeeze()\
            .to("cuda")

        embedding : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]\
            ['raw_embedding']\
            .squeeze()\
            .to("cuda")

        output_matrix : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]\
            ['modified_embedding']\
            .squeeze()\
            .to("cuda")
        

        if layer - 2 in self.dp['influences']:
            if self.dp['influences'][layer-2].device != "cpu":
                self.dp['influences'][layer-2].to("cpu")

        if layer-2 in self.dp['embeddings']:
            self.dp['embeddings'].pop(layer-2)

        last_influence = self.dp["influences"][layer-1].to("cuda")
        last_embedding = self.dp['embeddings'][layer -1 ].to("cuda")

        if values is not None:
            v_norm = values.norm(dim =1, p =1)
            device = v_norm.device
            attn_matrix = attn_matrix.to(device)
            influence_out = (v_norm* attn_matrix) @ (self.dp["influences"][layer-1].to("cuda"))

            influence_out = influence_out/(attn_matrix @ (v_norm))

        else:
            influence_out = attn_matrix @ last_influence

        influence = self.influence_of_sums(
            last_embedding,
            last_influence,
            output_matrix,
            influence_out,
            p_norm,
            **kwargs
        )

        self.dp['influences'][layer]= influence
        self.dp['embeddings'][layer] = embedding

class InfluenceHeads(BaseMetric):
    def influence_heads(
        self,
        layer : int,
        attn_matrix : torch.Tensor,
        values : torch.Tensor = None,
        p_norm : int = 1,
        **kwargs
    ):
        '''
        embedding : n dimensional tensor
        embedding_idx : int
        layer : int
        out : n dimensional tensor
        attn_vector : n dimensional tensor
        instruction_tokens_id : k dimensional tensor
        values : n x 4096 matrix
        '''


        if values is not None:

            v_norm = values.norm(dim =1, p =1)
            device = v_norm.device
            attn_matrix = attn_matrix.to(device)
            influence_heads = (v_norm* attn_matrix) @ (self.dp["influences"][layer-1].to("cuda"))

            influence_heads = influence_heads/(attn_matrix @ (v_norm))

        else:
            influence_heads = attn_matrix @ (self.dp["influences"][layer-1].to("cuda"))

        self.dp['influences_heads'] = influence_heads

    def influence_of_concat(
        self,
        attn_output_per_head : torch.Tensor,
    ):
        """_summary_

        Args:
            attn_output_per_head (torch.Tensor): size (32 x s x 128)

        Returns:
            _type_: _description_
        """        
        influence_heads = self.dp['influences_heads']

        dtype = attn_output_per_head.dtype

        norms = attn_output_per_head.norm(dim = -1)

        influence_heads = influence_heads.to("cuda").to(dtype)
        influence_concat = (norms * influence_heads).sum(dim = 0)/norms.sum(dim = 0)

        return influence_concat
    
    def influence_layer(
        self,
        influence_concat : torch.tensor, 
        concatenated_output : torch.Tensor,
        embedding : torch.Tensor,
        layer : int
    ):  
        if layer - 2 in self.dp['influences']:
            if self.dp['influences'][layer-2].device != "cpu":
                self.dp['influences'][layer-2].to("cpu")

        if layer-2 in self.dp['embeddings']:
            self.dp['embeddings'].pop(layer-2)


        self.dp['embeddings'][layer]= embedding

        last_influence = self.dp['influences'][layer-1].to("cuda")
        last_embedding = self.dp['embeddings'][layer -1 ].to("cuda")
        
        influence = self.influence_of_sums(
            last_embedding,
            last_influence,
            concatenated_output,
            influence_concat,
            1,
        )

        self.dp['influences'][layer] = influence

    def compute_influence(
        self, 
        layer : int, 
        use_values : bool = False,
        p_norm: int = 1, 
        **kwargs
    ):
        values = self.attn_saver_model\
            .internal_parameters[layer]\
            ['value']\
            .squeeze()\
            .to("cuda")
                
        if not use_values:
            values = None
        
        attn_matrix : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]['attention']\
            .squeeze()\
            .to("cuda")

        embedding : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]\
            ['raw_embedding']\
            .squeeze()\
            .to("cuda")

        output_matrix : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]\
            ['modified_embedding']\
            .squeeze()\
            .to("cuda")
        
        output_per_head = self.attn_saver_model\
            .internal_parameters\
            [layer]\
            ['output_before_mlp']\
            .squeeze()\
            .to("cuda")
        
        self.influence_heads(
            layer,
            attn_matrix,
            values,
            p_norm =1,
            **kwargs
        )

        influence_concat = self.influence_of_concat(
            output_per_head
        )

        self.influence_layer(
            influence_concat, 
            output_matrix,
            embedding,
            layer
        )

class AttentionRollout(Influence):
    @override
    def influence_of_sums(
        self,
        v1 : torch.Tensor,
        I1: float,
        v2 : torch.Tensor,
        I2 : float,
        p_norm : int = 1,
    ):
        return (I1 + I2)/2