import torch
import torch.nn as nn
import typing as tp

import code_03_nnmodel.nnmodel_04_adapter as nn4
import code_03_nnmodel.nnmodel_05_space_time_backbone_sequential as nn5


class AdapterInSequential(nn.Module):
    def __init__(
            self,
            task_type: tp.Literal["finetune_dtvu", "finetune_dtu"],
            emb_dim_adapter: int,
            backbone: nn5.SpaceTimeBackboneSequential,
    ):
        super().__init__()
        self.task_type = task_type
        self.backbone = backbone
        self.emb_dim_backbone = self.backbone.emb_dim_backbone
        self.emb_dim_adapter = emb_dim_adapter
        # ===========================================================================
        self.adapters = nn.ModuleList()
        self.lambda_gates = list()
        for index in range(self.backbone.num_layers_time + self.backbone.num_layers_space):
            self.lambda_gates.append(nn.Parameter(torch.randn(size=(1,))))
            if index % 2 == 0:
                self.adapters.add_module(
                    name=f"sequential_adapter_{index:02d}_space",
                    module=nn4.Adapter(backbone_dim=self.emb_dim_backbone, adapter_dim=self.emb_dim_adapter),
                )
            else:
                self.adapters.add_module(
                    name=f"sequential_adapter_{index:02d}_time",
                    module=nn4.Adapter(backbone_dim=self.emb_dim_backbone, adapter_dim=self.emb_dim_adapter),
                )

    def forward(
            self,
            x: torch.Tensor,
            attn_mask_v: torch.Tensor | None = None,
            attn_mask_g: torch.Tensor | None = None,
            targets: torch.Tensor | None = None,
            enable_v2g: bool = False,
    ):
        d, t, v, u, e, g = self.backbone.shape_check(x, attn_mask_v, attn_mask_g)
        if self.task_type == "pretrain":
            assert False, "适配器模式仅在微调任务启用"
        elif self.task_type == "finetune_dtvu":
            x, loss, targets = self.forward_adapter_fintune_dtvu(x, attn_mask_v, attn_mask_g, targets, enable_v2g, d, t, v, u, e, g)
        elif self.task_type == "finetune_dtu":
            assert enable_v2g is False, "[!] 微调的dtu模式下没有g维度"
            x, loss, targets = self.forward_adapter_fintune_dtu(x, attn_mask_v, targets, d, t, v, u, e)
        else:
            raise NotImplementedError

        return x, loss, targets

    def forward_adapter_fintune_dtvu(
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
        x = self.backbone.embeddings(x)
        assert x.shape == torch.Size([d, t, v, e])
        # ======================================================
        # ======================================================
        if enable_v2g:
            x = torch.einsum("dtve->dtev", x)
            x = self.backbone.v2g(x)
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
        x = self.xxxxx(x, attn_mask, d, t, o, e)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * o, t, e])
        x = self.backbone.final_norm(x)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * o, t, e])
        x = x.reshape(d, o, t, e)
        # ======================================================
        # ======================================================
        if enable_v2g:
            x = torch.einsum("dgte->dteg", x)
            x = self.backbone.g2v(x)
            x = torch.einsum("dtev->dtve", x)
        else:
            x = torch.einsum("dvte->dtve", x)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d, t, v, e]), x.shape
        x = self.backbone.classifier(x)
        assert x.shape == torch.Size([d, t, v, u]), x.shape
        # ======================================================
        # ======================================================
        x = x[:, :, :, 2:]  # 删除mask-user和fake-user的维度
        x = torch.einsum("dtvu->dtuv", x)
        x = self.backbone.multi_variant_norm(x)
        assert x.shape == torch.Size([d, t, u - 2, v]), x.shape
        assert targets.shape == torch.Size([d, t, u - 2]), targets.shape
        # ======================================================
        # ======================================================
        x = x.reshape(d * t * (u - 2), v)
        targets = targets.reshape(d * t * (u - 2))
        loss = self.backbone.loss_func(x, targets)
        return x, loss, targets

    def forward_adapter_fintune_dtu(
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
        x = self.backbone.embeddings(x)
        assert x.shape == torch.Size([d, t, u, e])
        # ======================================================
        # ======================================================
        x = torch.einsum("dtue->dute", x)
        x = x.reshape(d * u, t, e)
        attn_mask_u = torch.einsum("dtu->dut", attn_mask_u)
        attn_mask_u = attn_mask_u.reshape(d * u, t)
        # ======================================================
        # ======================================================
        x = self.xxxxx(x, attn_mask_u, d, t, u, e)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * u, t, e])
        x = self.backbone.final_norm(x)
        # ======================================================
        # ======================================================
        assert x.shape == torch.Size([d * u, t, e])
        x = x.reshape(d, u, t, e)
        # ======================================================
        # ======================================================
        x = torch.einsum("dute->dtue", x)
        x = self.backbone.classifier(x)
        assert x.shape == torch.Size([d, t, u, v]), x.shape
        assert targets.shape == torch.Size([d, t, u]), targets.shape
        # ======================================================
        # ======================================================
        x = x.reshape(d * t * u, v)
        targets = targets.reshape(d * t * u)
        loss = self.backbone.loss_func(x, targets)
        return x, loss, targets

    def forward_sequential_layers_with_adapter(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor,
            d: int,
            t: int,
            o: int,
            e: int,
    ):
        for layer_index, (name, layer) in enumerate(self.layers.named_children()):
            if "space" in name:
                position_ids = torch.arange(o, device=x.device).unsqueeze(0)
                x, attn_mask = self.do_t_e__2__dt_o_e(x, attn_mask, d, t, o, e)
                altered_attn_mask, sliding_window_mask = self.update_attention_mask(attn_mask)
                x = self.forward_layer_with_adapter(
                    x=x,
                    layer_index=layer_index,
                    layer=layer,
                    attention_mask=altered_attn_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                )
            # ======================================================
            elif "time" in name:
                position_ids = torch.arange(t, device=x.device).unsqueeze(0)
                x, attn_mask = self.dt_o_e__2__do_t_e(x, attn_mask, d, t, o, e)
                altered_attn_mask, sliding_window_mask = self.update_attention_mask(attn_mask)
                x = self.forward_layer_with_adapter(
                    x=x,
                    layer_index=layer_index,
                    layer=layer,
                    attention_mask=altered_attn_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                )
            else:
                raise NotImplementedError
        return x

    def forward_layer_with_adapter(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor,
            sliding_window_mask: torch.Tensor,
            position_ids: torch.Tensor,
            layer: nn.Module,
            layer_index: int,
    ):
        attn_outputs = layer.attn(
            layer.attn_norm(x),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=None,
            max_seqlen=None,
            output_attentions=False,
        )
        x = x + attn_outputs[0]
        # ======================================================
        x_mlp_normed = layer.mlp_norm(x)
        mlp_output = layer.mlp(x_mlp_normed)
        # ======================================================
        lambda_gate = self.lambda_gates[layer_index].to(x.device)
        adapter_output = self.adapters[layer_index](mlp_output)
        mlp_output = lambda_gate * adapter_output + (1 - lambda_gate) * mlp_output
        # ======================================================
        x = x + mlp_output
        # ======================================================
        return x

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     sliding_window_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     cu_seqlens: Optional[torch.Tensor] = None,
    #     max_seqlen: Optional[int] = None,
    #     output_attentions: Optional[bool] = False,
    # ) -> torch.Tensor:
    #     attn_outputs = self.attn(
    #         self.attn_norm(hidden_states),
    #         attention_mask=attention_mask,
    #         sliding_window_mask=sliding_window_mask,
    #         position_ids=position_ids,
    #         cu_seqlens=cu_seqlens,
    #         max_seqlen=max_seqlen,
    #         output_attentions=output_attentions,
    #     )
    #
    #     hidden_states = hidden_states + attn_outputs[0]
    #     mlp_output = (
    #         self.compiled_mlp(hidden_states)
    #         if self.config.reference_compile
    #         else self.mlp(self.mlp_norm(hidden_states))
    #     )
    #     hidden_states = hidden_states + mlp_output
    #
    #     return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted
