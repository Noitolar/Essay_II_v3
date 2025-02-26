import json
import typing as tp
import transformers as tfm


def load_modernbert_json_config(
        json_file_path: str,
        vocab_size: int = -1,
        atten_impl: tp.Literal["flash_attention_2", "eager", "sdpa"] = "eager",
        mask_token_id: int = -1,
        cls_token_id: int = -1,
        sep_token_id: int = -1,
        pad_token_id: int = -1,
        unk_token_id: int = -1,
        emb_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = -1,
        max_position_embeddings: int = 8192,
        local_attention: int = 256,
) -> tfm.ModernBertConfig:
    with open(json_file_path) as f:
        assert atten_impl != "flash_attention_2", "[!] FLASH ATTN 2 NOT INSTALLED."
        data: dict = json.load(f)
        data["vocab_size"] = vocab_size
        data["_attn_implementation"] = atten_impl
        data["mask_token_id"] = mask_token_id
        data["cls_token_id"] = cls_token_id
        data["bos_token_id"] = cls_token_id
        data["sep_token_id"] = sep_token_id
        data["eos_token_id"] = sep_token_id
        data["pad_token_id"] = pad_token_id
        data["unk_token_id"] = unk_token_id
        data["hidden_size"] = emb_dim
        data["num_attention_heads"] = num_heads
        data["num_hidden_layers"] = num_layers
        data["max_position_embeddings"] = max_position_embeddings
        data["local_attention"] = local_attention
        cfg = tfm.ModernBertConfig(**data)
        # cfg._attn_implementation = atten_impl
        return cfg
