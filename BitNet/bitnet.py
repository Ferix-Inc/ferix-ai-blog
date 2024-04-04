from typing import List, Optional, Tuple, Union
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention,
    LlamaMLP, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM,
)
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from bitlinear import BitLinear


class BitNetAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        if self.config.pretraining_tp > 1:
            raise ValueError(f"pretraining_tp must be <= 1 (got `pretraining_tp`: {self.config.pretraining_tp}).")


class BitNetFlashAttention2(LlamaFlashAttention2):
     def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size, bias=config.attention_bias)


class BitNetSdpaAttention(LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size, bias=config.attention_bias)


BITNET_ATTENTION_CLASSES = {
    "eager": BitNetAttention,
    "flash_attention_2": BitNetFlashAttention2,
    "sdpa": BitNetSdpaAttention,
}


class BitNetMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = BitLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        if self.config.pretraining_tp > 1:
            raise ValueError(f"pretraining_tp must be <= 1 (got `pretraining_tp`: {self.config.pretraining_tp}).")


class BitNetDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = BITNET_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = BitNetMLP(config)
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()


class BitNetModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [BitNetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class BitNetForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = BitNetModel(config)
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

