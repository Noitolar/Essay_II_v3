import torch
import torch.nn as nn
import transformers.models.modernbert.modeling_modernbert as modernbert_components
import code_03_nnmodel.nnmodel_01_attn_without_rope as attn_without_rope

from typing import Optional, Tuple

#
# class CustomizedModernBertLayer(modernbert_components.ModernBertPreTrainedModel):
class CustomizedModernBertLayer(modernbert_components.ModernBertEncoderLayer):
    def __init__(
            self,
            config: modernbert_components.ModernBertConfig,
            layer_id: int,
            remove_rope: bool,
    ):
        super().__init__(config, layer_id)
        self.config = config
        self.layer_id = layer_id
        self.remove_rope = remove_rope
        if remove_rope:
            self.attn = attn_without_rope.ModernBertAttentionWithoutRoPE(config, layer_id)


if __name__ == "__main__":
    import os

    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    torch.set_printoptions(linewidth=1000)
    import code_01_utils.utils_02_confg_loader as ut2

    num_days = 5
    num_ticks = 24
    num_nodes = 9
    num_fetures = 17

    emb_dim = 32

    config = ut2.load_modernbert_json_config(
        json_file_path="code_00_configs/modernbert_config_dropout.json",
        atten_impl="eager",
        vocab_size=num_fetures,
        emb_dim=emb_dim,
        num_heads=2,
    )

    block = CustomizedModernBertLayer(
        config=config,
        layer_id=1,
        remove_rope=True,
    )

    # data = torch.randn(size=(num_days, num_ticks, num_nodes, emb_dim))
    # data = data.reshape(num_days * num_ticks, num_nodes, emb_dim)
    # attn_mask = torch.ones(size=(num_days * num_ticks, num_nodes))
    #
    # print(data.shape)
    #
    # data = block.forward(inputs_embeds=data, attention_mask=attn_mask)
    #
    # print(data.shape)
