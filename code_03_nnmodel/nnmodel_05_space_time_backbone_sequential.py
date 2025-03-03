import torch
import torch.nn as nn
import transformers.models.modernbert.modeling_modernbert as modernbert_components
import typing as tp

import code_03_nnmodel.nnmodel_03_space_time_backbone_base as nn3
import code_03_nnmodel.nnmodel_02_cutomized_modernbert_layer as nn2


class SpaceTimeBackboneSequential(nn3.SpaceTimeBackboneBase):
    def __init__(
            self,
            # structure
            modernbert_config: modernbert_components.ModernBertConfig,
            num_layers_time: int,
            num_layers_space: int,

            # data shapes
            num_nodes: int,
            num_ticks: int,
            num_users: int,
            num_sub_graphs: int,
            is_multi_variant: bool,
            task_type: tp.Literal["pretrain", "finetune_dtvu", "finetune_dtu"],
    ):
        super().__init__(
            modernbert_config=modernbert_config,
            num_layers_time=num_layers_time,
            num_layers_space=num_layers_space,
            num_nodes=num_nodes,
            num_ticks=num_ticks,
            num_users=num_users,
            num_sub_graphs=num_sub_graphs,
            is_multi_variant=is_multi_variant,
            task_type=task_type,
        )

        self.layers = nn.ModuleList()
        assert modernbert_config.num_hidden_layers == num_layers_time + num_layers_space
        for index in range(num_layers_time + num_layers_space):
            if index % 2 == 0:
                self.layers.add_module(
                    name=f"sequential_layer_{index:02d}_space",
                    module=nn2.CustomizedModernBertLayer(config=modernbert_config, layer_id=index, remove_rope=True),
                    # module=nn2.CustomizedModernBertLayer(config=modernbert_config, layer_id=index, remove_rope=False),
                )
            else:
                self.layers.add_module(
                    name=f"sequential_layer_{index:02d}_time",
                    module=nn2.CustomizedModernBertLayer(config=modernbert_config, layer_id=index, remove_rope=False)
                )
        self.post_init()

    def forward(
            self,
            x: torch.Tensor,
            attn_mask_v: torch.Tensor | None = None,
            attn_mask_g: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            enable_v2g: bool = False,
    ):
        d, t, v, u, e, g = self.shape_check(x, attn_mask_v, attn_mask_g)
        if self.task_type == "pretrain":
            x, loss, targets = self.forward_pretrain(x, attn_mask_v, attn_mask_g, targets, enable_v2g, d, t, v, u, e, g)
        elif self.task_type == "finetune_dtvu":
            x, loss, targets = self.forward_fintune_dtvu(x, attn_mask_v, attn_mask_g, targets, enable_v2g, d, t, v, u, e, g)
        elif self.task_type == "finetune_dtu":
            assert enable_v2g is False, "[!] 微调的dtu模式下没有g维度"
            x, loss, targets = self.forward_fintune_dtu(x, attn_mask_v, targets, d, t, v, u, e)
        else:
            raise NotImplementedError

        return x, loss, targets

    def forward_pretrain(
            self,
            x: torch.Tensor,
            attn_mask_v: torch.Tensor,
            attn_mask_g: torch.Tensor,
            targets: torch.Tensor,
            enable_v2g: bool,
            d: int,
            t: int,
            v: int,
            u: int,
            e: int,
            g: int
    ):
        assert x.shape == torch.Size([d, t, v])
        assert attn_mask_v.shape == torch.Size([d, t, v])
        assert attn_mask_g.shape == torch.Size([d, t, g])
        assert targets.shape == torch.Size([d, t, v])
        # ======================================================
        # ======================================================
        x = self.embeddings(x)
        assert x.shape == torch.Size([d, t, v, e])
        # ======================================================
        # ======================================================
        if enable_v2g:
            x = torch.einsum("dtve->dtev", x)
            x = self.v2g(x)
            x = torch.einsum("dteg->dtge", x)
            attn_mask = attn_mask_g
            o = g
        else:
            attn_mask = attn_mask_v
            o = v
        # ======================================================
        # ======================================================
        x = torch.einsum("dtoe->dote", x)
        x = x.reshape(d * o, t, e)
        attn_mask = torch.einsum("dto->dot", attn_mask)
        attn_mask = attn_mask.reshape(d * o, t)
        # ======================================================
        # ======================================================
        x = self.forward_sequential_layers(x, attn_mask, d, t, o, e)
        # x = self.forward_sequential_layers_space_then_time(x, attn_mask, d, t, o, e)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * o, t, e])
        x = self.final_norm(x)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * o, t, e])
        x = x.reshape(d, o, t, e)
        # ======================================================
        # ======================================================
        if enable_v2g:
            x = torch.einsum("dgte->dteg", x)
            x = self.g2v(x)
            x = torch.einsum("dtev->dtve", x)
        else:
            x = torch.einsum("dvte->dtve", x)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d, t, v, e]), x.shape
        x = self.classifier(x)
        assert x.shape == torch.Size([d, t, v, u]), x.shape
        assert targets.shape == torch.Size([d, t, v]), targets.shape
        # ======================================================
        # ======================================================
        x = x.reshape(d * t * v, u)
        targets = targets.reshape(d * t * v)
        loss = self.loss_func(x, targets)
        return x, loss, targets

    def forward_fintune_dtvu(
            self,
            x: torch.Tensor,
            attn_mask_v: torch.Tensor,
            attn_mask_g: torch.Tensor,
            targets: torch.Tensor,
            enable_v2g: bool,
            d: int,
            t: int,
            v: int,
            u: int,
            e: int,
            g: int
    ):
        assert x.shape == torch.Size([d, t, v, u])
        assert attn_mask_v.shape == torch.Size([d, t, v])
        assert attn_mask_g.shape == torch.Size([d, t, g])
        assert targets.shape == torch.Size([d, t, u - 2])
        # ======================================================
        # ======================================================
        x = self.embeddings(x)
        assert x.shape == torch.Size([d, t, v, e])
        # ======================================================
        # ======================================================
        if enable_v2g:
            x = torch.einsum("dtve->dtev", x)
            x = self.v2g(x)
            x = torch.einsum("dteg->dtge", x)
            attn_mask = attn_mask_g
            o = g
        else:
            attn_mask = attn_mask_v
            o = v
        # ======================================================
        # ======================================================
        x = torch.einsum("dtoe->dote", x)
        x = x.reshape(d * o, t, e)
        attn_mask = torch.einsum("dto->dot", attn_mask)
        attn_mask = attn_mask.reshape(d * o, t)
        # ======================================================
        # ======================================================
        x = self.forward_sequential_layers(x, attn_mask, d, t, o, e)
        # x = self.forward_sequential_layers_time_only(x, attn_mask, d, t, o, e)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * o, t, e])
        x = self.final_norm(x)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * o, t, e])
        x = x.reshape(d, o, t, e)
        # ======================================================
        # ======================================================
        if enable_v2g:
            x = torch.einsum("dgte->dteg", x)
            x = self.g2v(x)
            x = torch.einsum("dtev->dtve", x)
        else:
            x = torch.einsum("dvte->dtve", x)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d, t, v, e]), x.shape
        x = self.classifier(x)
        assert x.shape == torch.Size([d, t, v, u]), x.shape
        # ======================================================
        # ======================================================
        x = x[:, :, :, 2:]  # 删除mask-user和fake-user的维度
        x = torch.einsum("dtvu->dtuv", x)
        x = self.multi_variant_norm(x) # TODO: 这个LayerNorm是否能起到正面作用？
        assert x.shape == torch.Size([d, t, u - 2, v]), x.shape
        assert targets.shape == torch.Size([d, t, u - 2]), targets.shape
        # ======================================================
        # ======================================================
        x = x.reshape(d * t * (u - 2), v)
        targets = targets.reshape(d * t * (u - 2))
        loss = self.loss_func(x, targets)
        return x, loss, targets

    def forward_fintune_dtu(
            self,
            x: torch.Tensor,
            attn_mask_u: torch.Tensor,
            targets: torch.Tensor,
            d: int,
            t: int,
            v: int,
            u: int,
            e: int,
    ):
        assert x.shape == torch.Size([d, t, u])
        assert attn_mask_u.shape == torch.Size([d, t, u])
        assert targets.shape == torch.Size([d, t, u])
        # ======================================================
        # ======================================================
        x = self.embeddings(x)
        assert x.shape == torch.Size([d, t, u, e])
        # ======================================================
        # ======================================================
        x = torch.einsum("dtue->dute", x)
        x = x.reshape(d * u, t, e)
        attn_mask_u = torch.einsum("dtu->dut", attn_mask_u)
        attn_mask_u = attn_mask_u.reshape(d * u, t)
        # ======================================================
        # ======================================================
        x = self.forward_sequential_layers(x, attn_mask_u, d, t, u, e)
        # x = self.forward_sequential_layers_time_only(x, attn_mask_u, d, t, u, e)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * u, t, e])
        x = self.final_norm(x)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * u, t, e])
        x = x.reshape(d, u, t, e)
        # ======================================================
        # ======================================================
        x = torch.einsum("dute->dtue", x)
        x = self.classifier(x)
        assert x.shape == torch.Size([d, t, u, v]), x.shape
        assert targets.shape == torch.Size([d, t, u]), targets.shape
        # ======================================================
        # ======================================================
        x = x.reshape(d * t * u, v)
        targets = targets.reshape(d * t * u)
        loss = self.loss_func(x, targets)
        return x, loss, targets

    def forward_sequential_layers(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor,
            d: int,
            t: int,
            o: int,
            e: int,
    ):
        for name, layer in self.layers.named_children():
            if "space" in name:
                # assert layer.remove_rope is True
                position_ids = torch.arange(o, device=x.device).unsqueeze(0)
                x, attn_mask = self.do_t_e__2__dt_o_e(x, attn_mask, d, t, o, e)
                altered_attn_mask, sliding_window_mask = self.update_attention_mask(attn_mask)
                x = layer(
                    x,
                    attention_mask=altered_attn_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                )[0]
            elif "time" in name:
                # assert layer.remove_rope is False
                position_ids = torch.arange(t, device=x.device).unsqueeze(0)
                x, attn_mask = self.dt_o_e__2__do_t_e(x, attn_mask, d, t, o, e)
                altered_attn_mask, sliding_window_mask = self.update_attention_mask(attn_mask)
                x = layer(
                    x,
                    attention_mask=altered_attn_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                )[0]
            else:
                raise NotImplementedError
        return x

    def forward_sequential_layers_space_then_time(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor,
            d: int,
            t: int,
            o: int,
            e: int,
    ):
        space_layers = [layer for name, layer in self.layers.named_children() if "space" in name]
        time_layers = [layer for name, layer in self.layers.named_children() if "time" in name]
        # ======================================================
        position_ids = torch.arange(o, device=x.device).unsqueeze(0)
        x, attn_mask = self.do_t_e__2__dt_o_e(x, attn_mask, d, t, o, e)
        altered_attn_mask, sliding_window_mask = self.update_attention_mask(attn_mask)
        for space_layer in space_layers:
            x = space_layer(
                x,
                attention_mask=altered_attn_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                output_attentions=False,
            )[0]
        # ======================================================
        position_ids = torch.arange(t, device=x.device).unsqueeze(0)
        x, attn_mask = self.dt_o_e__2__do_t_e(x, attn_mask, d, t, o, e)
        altered_attn_mask, sliding_window_mask = self.update_attention_mask(attn_mask)
        for time_layer in time_layers:
            x = time_layer(
                x,
                attention_mask=altered_attn_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                output_attentions=False,
            )[0]
        # ======================================================
        return x


    def forward_sequential_layers_time_only(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor,
            d: int,
            t: int,
            o: int,
            e: int,
    ):
        # space_layers = [layer for name, layer in self.layers.named_children() if "space" in name]
        time_layers = [layer for name, layer in self.layers.named_children() if "time" in name]
        # ======================================================
        # position_ids = torch.arange(o, device=x.device).unsqueeze(0)
        # x, attn_mask = self.do_t_e__2__dt_o_e(x, attn_mask, d, t, o, e)
        # altered_attn_mask, sliding_window_mask = self.update_attention_mask(attn_mask)
        # for space_layer in space_layers:
        #     x = space_layer(
        #         x,
        #         attention_mask=altered_attn_mask,
        #         sliding_window_mask=sliding_window_mask,
        #         position_ids=position_ids,
        #         output_attentions=False,
        #     )[0]
        # ======================================================
        position_ids = torch.arange(t, device=x.device).unsqueeze(0)
        # x, attn_mask = self.dt_o_e__2__do_t_e(x, attn_mask, d, t, o, e)
        altered_attn_mask, sliding_window_mask = self.update_attention_mask(attn_mask)
        for time_layer in time_layers:
            x = time_layer(
                x,
                attention_mask=altered_attn_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                output_attentions=False,
            )[0]
        # ======================================================
        return x

# if __name__ == "__main__":
#     import os
#     import code_01_utils.utils_02_confg_loader as ut2
#
#     os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
#     torch.set_printoptions(linewidth=1000)
#
#     d, t, v, u, e = 5, 24, 17, 9, 32
#     cfg = ut2.load_modernbert_json_config(
#         json_file_path="code_00_configs/modernbert_config_dropout.json",
#         num_layers=4,
#         num_heads=2,
#         emb_dim=e
#     )
#     model = SpaceTimeBackboneSequential(
#         modernbert_config=cfg,
#         num_layers_time=2,
#         num_layers_space=2,
#         num_nodes=v,
#         num_ticks=t,
#         num_users=u,
#         is_multi_variant=True,
#     )
