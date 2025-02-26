import torch
import torch.nn as nn
import transformers.activations as activations
import transformers.models.modernbert.modeling_modernbert as modernbert_components
import typing as tp

import code_03_nnmodel.nnmodel_02_cutomized_modernbert_layer as nn2
import code_03_nnmodel.nnmodel_05_space_time_backbone_sequential as nn5


class SideBranchesSequential(nn.Module):
    def __init__(
            self,
            task_type: tp.Literal["finetune_dtvu", "finetune_dtu"],
            branch_modernbert_config: modernbert_components.ModernBertConfig,
            backbone: nn5.SpaceTimeBackboneSequential,
    ):
        super().__init__()
        self.task_type = task_type
        self.branch_modernbert_config = branch_modernbert_config
        self.backbone = backbone
        self.emb_dim_branch = branch_modernbert_config.hidden_size
        self.num_heads_branch = branch_modernbert_config.num_attention_heads
        # ===========================================================================
        self.branches = nn.ModuleList()
        self.compressors = nn.ModuleList([nn.Linear(self.backbone.emb_dim_backbone, self.emb_dim_branch)])  # 有layer+1个compressor
        # ===========================================================================
        for index in range(self.backbone.num_layers_time + self.backbone.num_layers_space):
            self.compressors.append(nn.Linear(self.backbone.emb_dim_backbone, self.emb_dim_branch))
            if index % 2 == 0:
                self.branches.add_module(
                    name=f"sequential_branch_{index:02d}_space",
                    module=nn2.CustomizedModernBertLayer(config=branch_modernbert_config, layer_id=index, remove_rope=True),
                )
            else:
                self.branches.add_module(
                    name=f"sequential_branch_{index:02d}_time",
                    module=nn2.CustomizedModernBertLayer(config=branch_modernbert_config, layer_id=index, remove_rope=False)
                )
        # ===========================================================================
        self.decompressor = nn.Sequential(
            nn.Linear(self.emb_dim_branch, self.emb_dim_branch),
            activations.ACT2FN[self.branch_modernbert_config.classifier_activation],
            nn.LayerNorm(self.emb_dim_branch),
            nn.Linear(self.emb_dim_branch, self.backbone.emb_dim_backbone),
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
            assert False, "侧枝模式仅在微调任务启用"
        elif self.task_type == "finetune_dtvu":
            x, loss, targets = self.forward_fintune_dtvu(x, attn_mask_v, attn_mask_g, targets, enable_v2g, d, t, v, u, e, g)
        elif self.task_type == "finetune_dtu":
            assert enable_v2g is False, "[!] 微调的dtu模式下没有g维度"
            x, loss, targets = self.forward_fintune_dtu(x, attn_mask_v, targets, d, t, v, u, e)
        else:
            raise NotImplementedError

        return x, loss, targets

    def forward_sequential_branches(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor,
            d: int,
            t: int,
            o: int,
            e: int,
    ):
        x_branch = x.clone()
        x_branch = self.compressors[0](x_branch)
        assert x_branch.shape == torch.Size([d * o, t, self.emb_dim_branch]), x_branch.shape

        for layer_index, ((layer_name, layer), (branch_name, branch)) in enumerate(zip(self.backbone.layers.named_children(), self.branches.named_children())):
            if "space" in layer_name:
                assert "space" in branch_name
                # ===========================================================================
                position_ids = torch.arange(o, device=x.device).unsqueeze(0)
                # ===========================================================================
                x, attn_mask = self.backbone.do_t_e__2__dt_o_e(x, attn_mask, d, t, o, e)
                altered_attn_mask, sliding_window_mask = self.backbone.update_attention_mask(attn_mask)
                # ===========================================================================
                x = layer(
                    x,
                    attention_mask=altered_attn_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                )[0]
                # ===========================================================================
                x_branch, _ = self.backbone.do_t_e__2__dt_o_e(x_branch, None, d, t, o, self.emb_dim_branch)
                x_branch = x_branch + self.compressors[layer_index + 1](x)
                # ===========================================================================
                x_branch = branch(
                    x_branch,
                    attention_mask=altered_attn_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                )[0]
                # ===========================================================================
            elif "time" in layer_name:
                assert "time" in branch_name
                # ===========================================================================
                position_ids = torch.arange(t, device=x.device).unsqueeze(0)
                # ===========================================================================
                x, attn_mask = self.backbone.dt_o_e__2__do_t_e(x, attn_mask, d, t, o, e)
                altered_attn_mask, sliding_window_mask = self.backbone.update_attention_mask(attn_mask)
                # ===========================================================================
                x = layer(
                    x,
                    attention_mask=altered_attn_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                )[0]
                # ===========================================================================
                x_branch, _ = self.backbone.dt_o_e__2__do_t_e(x_branch, None, d, t, o, self.emb_dim_branch)
                x_branch = x_branch + self.compressors[layer_index + 1](x)
                # ===========================================================================
                x_branch = branch(
                    x_branch,
                    attention_mask=altered_attn_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                )[0]
                # ===========================================================================
            else:
                raise NotImplementedError

        x_branch = self.decompressor(x_branch)
        assert x_branch.shape == torch.Size([d * o, t, self.backbone.emb_dim_backbone])

        return x_branch


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
        x = self.forward_sequential_branches(x, attn_mask, d, t, o, e)
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
        x = self.forward_sequential_branches(x, attn_mask_u, d, t, u, e)
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