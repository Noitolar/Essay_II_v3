import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
import transformers.models.modernbert.modeling_modernbert as modernbert_components

from typing import Optional, Union, Tuple


def eager_attention_forward_without_rope(
        module: modernbert_components.ModernBertAttention,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor,
        sliding_window_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        local_attention: Tuple[int, int],
        bs: int,
        dim: int,
        output_attentions: Optional[bool] = False,
        **_kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    # cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    # query, key = modernbert_components.apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return attn_output, attn_weights
    return (attn_output,)


def sdpa_attention_forward_without_rope(
        module: modernbert_components.ModernBertAttention,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor,
        sliding_window_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        local_attention: Tuple[int, int],
        bs: int,
        dim: int,
        **_kwargs,
) -> Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    # cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    # query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (
        nnfunc.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=attention_mask,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


MODERNBERT_ATTENTION_FUNCTION_WITHOUT_ROPE = {
    # "flash_attention_2": flash_attention_forward,
    "eager": eager_attention_forward_without_rope,
    "sdpa": sdpa_attention_forward_without_rope,
}


class ModernBertAttentionWithoutRoPE(nn.Module):
    def __init__(self, config: modernbert_components.ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        # rope_theta = config.global_rope_theta
        # max_position_embeddings = config.max_position_embeddings
        # if self.local_attention != (-1, -1):
        #     if config.local_rope_theta is not None:
        #         rope_theta = config.local_rope_theta
        #     max_position_embeddings = config.local_attention

        # if config._attn_implementation == "flash_attention_2":
        #     self.rotary_emb = modernbert_components.ModernBertUnpaddedRotaryEmbedding(
        #         dim=self.head_dim, max_seqlen=max_position_embeddings, base=rope_theta
        #     )
        # else:
        #     self.rotary_emb = modernbert_components.ModernBertRotaryEmbedding(config=config, dim=self.head_dim, base=rope_theta)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: Optional[bool] = False,
            **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION_WITHOUT_ROPE[self.config._attn_implementation](
            self,
            qkv=qkv,
            # rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted
