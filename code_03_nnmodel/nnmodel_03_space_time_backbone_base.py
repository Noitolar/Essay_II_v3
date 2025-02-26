import torch
import torch.nn as nn
import transformers.activations as activations
import transformers.models.modernbert.modeling_modernbert as modernbert_components
import typing as tp


class SpaceTimeBackboneBase(modernbert_components.ModernBertPreTrainedModel):
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
        super().__init__(config=modernbert_config)
        # =======================================================
        # =======================================================
        self.modernbert_config = modernbert_config
        self.num_layers_time = num_layers_time
        self.num_layers_space = num_layers_space
        self.num_nodes = num_nodes
        self.num_ticks = num_ticks
        self.num_users = num_users
        self.is_multi_variant = is_multi_variant
        # =======================================================
        # =======================================================
        self.task_type = task_type
        self.num_sub_graphs = num_sub_graphs  # 子图维度，用于对节点维度进行降维
        # =======================================================
        # =======================================================
        if self.task_type == "pretrain":
            # 添加mask状态
            assert is_multi_variant is False
            self.num_users += 1
            # =======================================================
        elif self.task_type == "finetune_dtvu":
            # 添加mask状态和fake状态
            assert is_multi_variant is True
            self.num_users += 2
            # =======================================================
        elif self.task_type == "finetune_dtu":
            # 添加mask状态（finetune_dtu的状态是节点）
            assert is_multi_variant is True
            self.num_nodes += 1
            # =======================================================
        else:
            raise NotImplementedError
        # 根据不同的任务，嵌入方式有所不同
        # =======================================================
        # =======================================================
        self.emb_dim_backbone = modernbert_config.hidden_size
        self.embeddings = nn.Sequential(
            nn.Identity(),
            nn.Dropout(modernbert_config.embedding_dropout),
            nn.LayerNorm(self.emb_dim_backbone, eps=modernbert_config.norm_eps, bias=modernbert_config.norm_bias),
        )
        if self.task_type == "pretrain":
            self.embeddings[0] = nn.Embedding(self.num_users, self.emb_dim_backbone)
        elif self.task_type == "finetune_dtvu":
            self.embeddings[0] = nn.Linear(self.num_users, self.emb_dim_backbone)
        elif self.task_type == "finetune_dtu":
            self.embeddings[0] = nn.Embedding(self.num_nodes, self.emb_dim_backbone)
        # =======================================================
        # =======================================================
        self.v2g = nn.Linear(num_nodes, num_sub_graphs)
        self.g2v = nn.Linear(num_sub_graphs, num_nodes)
        # =======================================================
        # =======================================================
        self.final_norm = nn.LayerNorm(self.emb_dim_backbone, eps=modernbert_config.norm_eps, bias=modernbert_config.norm_bias)
        self.multi_variant_norm = nn.LayerNorm(self.num_nodes, eps=modernbert_config.norm_eps, bias=modernbert_config.norm_bias)
        self.loss_func = nn.CrossEntropyLoss()
        # =======================================================
        # =======================================================
        if self.task_type == "pretrain" or self.task_type == "finetune_dtvu":
            num_classes = self.num_users
        elif self.task_type == "finetune_dtu":
            num_classes = self.num_nodes
        else:
            raise NotImplementedError
        # =======================================================
        # =======================================================
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim_backbone, self.emb_dim_backbone, bias=modernbert_config.classifier_bias),
            activations.ACT2FN[modernbert_config.classifier_activation],
            nn.LayerNorm(self.emb_dim_backbone, eps=modernbert_config.norm_eps, bias=modernbert_config.norm_bias),
            nn.Dropout(modernbert_config.classifier_dropout),
            nn.Linear(self.emb_dim_backbone, num_classes, bias=modernbert_config.classifier_bias),
        )

    def shape_check(
            self,
            x: torch.Tensor,
            attn_mask_v: torch.Tensor,
            attn_mask_g: torch.Tensor | None = None
    ):
        d = x.shape[0]
        t = self.num_ticks
        v = self.num_nodes
        u = self.num_users
        e = self.emb_dim_backbone
        g = self.num_sub_graphs
        # =======================================================
        # =======================================================
        if self.task_type == "pretrain":
            assert x.shape == torch.Size([d, t, v])
            assert attn_mask_v.shape == torch.Size([d, t, v])
            assert attn_mask_g.shape == torch.Size([d, t, g])
        elif self.task_type == "finetune_dtvu":
            assert x.shape == torch.Size([d, t, v, u])
            assert attn_mask_v.shape == torch.Size([d, t, v])
            assert attn_mask_g.shape == torch.Size([d, t, g])
        elif self.task_type == "finetune_dtu":
            assert x.shape == torch.Size([d, t, u])
            assert attn_mask_v.shape == torch.Size([d, t, u]), attn_mask_v.shape
            assert attn_mask_g is None
        else:
            raise NotImplementedError
        return d, t, v, u, e, g

    @staticmethod
    def do_t_e__2__dt_o_e(
            x: torch.Tensor,
            attn_mask: torch.Tensor | None,
            d: int,
            t: int,
            o: int,
            e: int,
    ):
        assert x.shape == torch.Size([d * o, t, e]), x.shape
        assert attn_mask is None or attn_mask.shape == torch.Size([d * o, t]), attn_mask.shape
        x = x.reshape(d, o, t, e)
        x = torch.einsum("dote->dtoe", x)
        x = x.reshape(d * t, o, e)
        # =======================================================
        # =======================================================
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(d, o, t)
            attn_mask = torch.einsum("dot->dto", attn_mask)
            attn_mask = attn_mask.reshape(d * t, o)
        # =======================================================
        # =======================================================
        return x, attn_mask

    @staticmethod
    def dt_o_e__2__do_t_e(
            x: torch.Tensor,
            attn_mask: torch.Tensor | None,
            d: int,
            t: int,
            o: int,
            e: int,
    ):
        assert x.shape == torch.Size([d * t, o, e])
        assert attn_mask is None or attn_mask.shape == torch.Size([d * t, o])
        x = x.reshape(d, t, o, e)
        x = torch.einsum("dtoe->dote", x)
        x = x.reshape(d * o, t, e)
        # =======================================================
        # =======================================================
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(d, t, o)
            attn_mask = torch.einsum("dto->dot", attn_mask)
            attn_mask = attn_mask.reshape(d * o, t)
        # =======================================================
        # =======================================================
        return x, attn_mask

    def update_attention_mask(
            self,
            attention_mask: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        global_attention_mask = modernbert_components._prepare_4d_attention_mask(attention_mask, self.dtype)
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        distance = torch.abs(rows - rows.T)
        window_mask = ((distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_mask.device))
        sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)
        return global_attention_mask, sliding_window_mask
