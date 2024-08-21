from collections import OrderedDict     
import types
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Optional, Tuple
from copy import deepcopy
import math
from torch import nn
import pdb
import seaborn as sns
import matplotlib.pyplot as plt

def decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class GUIDEModel(torch.nn.Module):
    def __init__(
        self,
        model : AutoModel,
        tokenizer : AutoTokenizer,
        delta_attention : float = 0,
        augmented_layers : Union[str, List] = 'all',
        should_save_params : bool = True,
        *args,
        **kwargs
    ) -> None:
        """
        A model that extends a pre-trained transformer model with custom attention mechanisms.

        The GUIDEModel class allows modifying the attention mechanisms in specific layers of a transformer model. It can
        augment the attention with a delta value, apply rotary position embeddings, and capture internal parameters
        during the forward pass for analysis.

        Args:
            model (AutoModel): The base transformer model to be extended.
            tokenizer (AutoTokenizer): The tokenizer corresponding to the base model.
            delta_attention (float, optional): The value to modify attention scores. Defaults to None.
            augmented_layers (Union[str, List], optional): Specifies which layers to augment ('all', 'none', 'first', 'last', or a list of layer indices). Defaults to 'all'.
            should_save_params (bool, optional): Whether to save internal parameters during the forward pass. Defaults to True.
        """
        
        super().__init__(*args, **kwargs)

        self.base_model : AutoModel = model
        self.internal_parameters : List[torch.Tensor] = []
        self.DELTA_ATTENTION = delta_attention
        self.tokenizer = tokenizer
        self.start_idx = None
        self.end_idx = None
        self.save_internal_params = should_save_params
        
        options = ["all", "none", "first", "last"]

        if type(augmented_layers) == str:
            assert augmented_layers in options, f"augmented_layers must be one of {options}" 

        self.augmented_layers = augmented_layers
        self.has_hook = False
        self.remove_hooks()
        self.insert_hook()
        self.num_layers = len(self.base_model.model.layers)

        self.insert_code_on_forward()

    def insert_code_on_forward(self,):

        for layer in range(self.num_layers):
            self.base_model.model.layers[layer].forward = types.MethodType(
                decoder_forward, 
                self.base_model.model.layers[layer]
            )


    def generate(self, *args,**kwargs):
        self.internal_parameters =[]
        return self.base_model.generate(*args,**kwargs)
    
    def set_delta_attention(self, delta):
        """
        Sets the delta attention value for modifying attention scores.

        Args:
            delta (float): The value to adjust attention scores.
        """
        self.DELTA_ATTENTION = delta
    
    def set_reference_tokens(self, start_idx : int, end_idx : int):
        """
        Sets the start and end indices of the tokens to be used as references for attention modification.

        Args:
            start_idx (int): The starting index of the reference tokens.
            end_idx (int): The ending index of the reference tokens.
        """

        self.start_idx = start_idx
        self.end_idx = end_idx

    def reset_internal_parameters(self):
        """
        Resets the internal parameters list that stores intermediate values during the forward pass.
        """
        self.internal_parameters = []

    def remove_hooks(self):
        """
        Removes any forward hooks attached to the self-attention modules of the base model.
        """
        for name, module in self.base_model.named_modules():
            if name.endswith("self_attn"):
                module._forward_hooks = OrderedDict()

        self.has_hook = False

    def insert_hook(self):
        """
        Inserts forward hooks into the self-attention modules of the base model to modify and capture attention-related parameters.
        """
        for name, internal_module in self.base_model.named_modules():
            if name.endswith("self_attn"):
                internal_module.register_forward_hook(self.get_forward_params)

        self.has_hook = True

    def __call__(self, tokens, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        """
        Calls the base model with the provided tokens and captures the internal parameters.

        Args:
            tokens (torch.Tensor): The input tokens for the model.
            *args: Variable length argument list for the base model's forward method.
            **kwds: Arbitrary keyword arguments for the base model's forward method.

        Returns:
            The output of the base model.
        """
        self.internal_parameters =[]
        return self.base_model(tokens, **kwds)

    def rotate_half(self, x):
      """Rotates half the hidden dims of the input."""
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
      """Applies Rotary Position Embedding to the query and key tensors.

      Args:
          q (`torch.Tensor`): The query tensor.
          k (`torch.Tensor`): The key tensor.
          cos (`torch.Tensor`): The cosine part of the rotary embedding.
          sin (`torch.Tensor`): The sine part of the rotary embedding.
          position_ids (`torch.Tensor`, *optional*):
              Deprecated and unused.
          unsqueeze_dim (`int`, *optional*, defaults to 1):
              The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
              sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
              that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
              k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
              cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
              the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
      Returns:
          `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
      """
      cos = cos.unsqueeze(unsqueeze_dim)
      sin = sin.unsqueeze(unsqueeze_dim)
      q_embed = (q * cos) + (self.rotate_half(q) * sin)
      k_embed = (k * cos) + (self.rotate_half(k) * sin)
      return q_embed, k_embed

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
      """
      This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
      num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
      """
      batch, num_key_value_heads, slen, head_dim = hidden_states.shape
      if n_rep == 1:
          return hidden_states
      hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
      return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @torch.no_grad
    def get_forward_params(self, module, input, output ):
        """
        Modifies and captures attention-related parameters during the forward pass.

        Args:
            module (nn.Module): The current attention module being processed.
            input (torch.Tensor): The input tensor to the module.
            output (torch.Tensor): The output tensor from the module.
        """
        if type(self.augmented_layers) != str:
            if module.layer_idx in self.augmented_layers:
                return self.change_and_fetch_attention(
                    module,
                    input,
                    output,
                    delta_attention=self.DELTA_ATTENTION
                )
            
            else:
                self.change_and_fetch_attention(
                    module,
                    input,
                    output,
                    delta_attention=0
                )
        
        
        if self.augmented_layers == "all":
            return self.change_and_fetch_attention(
                module,
                input,
                output,
                delta_attention=self.DELTA_ATTENTION
            )
        
        if self.augmented_layers == "none":
            return self.change_and_fetch_attention(
                module,
                input,
                output,
                delta_attention = 0
            )
        
        if self.augmented_layers == "first":
            if module.layer_idx == 0:
                return self.change_and_fetch_attention(
                    module,
                    input,
                    output,
                    delta_attention=self.DELTA_ATTENTION
                )

            else:
                self.change_and_fetch_attention(
                    module,
                    input,
                    output,
                    delta_attention=0
                )

        if self.augmented_layers == "last":
            if module.layer_idx == 31:
                return self.change_and_fetch_attention(
                    module,
                    input,
                    output,
                    delta_attention=self.DELTA_ATTENTION
                )

            else:
                self.change_and_fetch_attention(
                    module,
                    input,
                    output,
                    delta_attention=0
                )

    def change_and_fetch_attention(self,module,input,output, delta_attention):
        """
        Modifies attention scores and fetches internal parameters for analysis.

        Args:
            module (nn.Module): The current attention module.
            input (torch.Tensor): The input tensor to the module.
            output (torch.Tensor): The output tensor from the module.
            delta_attention (float): The value to adjust attention scores.

        Returns:
            torch.Tensor: The modified attention output.
        """

        if input[0].shape[1] == 1:
            return

        hidden_states = input[0]
        bsz, q_len, _ = hidden_states.size()
        attention_mask = deepcopy(input[1])
        position_ids = input[2]

        attention_mask[:,:,:,self.start_idx:self.end_idx] += delta_attention

        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

        cos, sin = module.rotary_emb(value_states, position_ids)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = self.repeat_kv(key_states, module.num_key_value_groups)
        value_states = self.repeat_kv(value_states, module.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)

        # if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)

        output_before_mlp = deepcopy(attn_output)

        if attn_output.size() != (bsz, module.num_heads, q_len, module.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, module.num_heads, q_len, module.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = module.o_proj(attn_output)

        assert (attn_output.shape == output[0].shape)

        if delta_attention !=0 :
            assert (attn_output!=output[0]).any(), "When changing the attention with delta != 0, the outputs must be different"

        else:
            assert (attn_output==output[0]).all(), "When using delta_attention = 0, the outputs must be equal"

        if self.save_internal_params:
            layer_dict = {
                # "query": query_states,
                # "key": key_states,
                "value": value_states.reshape(-1,4096).to("cpu"),#.mean(dim= 1),
                "output_before_mlp" : output_before_mlp.to("cpu"),
                "attention" : attn_weights.to("cpu"), #.mean(dim=1),#.to("cpu"),
                "avg_attention_heads" : attn_weights.mean(dim = 1).to("cpu"),
                "raw_embedding": input[0].to("cpu"),
                "modified_embedding" : attn_output.to("cpu"),
                # "QK^T": qkT
            }
            self.internal_parameters.append(layer_dict)
        
        return attn_output, None, output[2]

