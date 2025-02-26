import torch
import typing as tp
import numpy as np


class MaskManipulator:
    def __init__(
            self,
            num_ticks: int,
            num_nodes: int,
            num_users: int,
            num_sub_graphs: int,
    ):
        self.d = None
        self.t = num_ticks
        self.v = num_nodes
        self.u0 = num_users
        self.u1 = self.u0 + 1
        self.u2 = self.u0 + 2
        self.g = num_sub_graphs

    def mask(
            self,
            x: torch.Tensor,
            method: tp.Literal["random", "space", "time", "time_span", "time_pred"],
            task_type: tp.Literal["pretrain", "finetune_dtvu", "finetune_dtu"],
            targets: torch.Tensor,
            mark_ratio: float | None = None,
            history_ratio: float | None = None,
            preds_ratio: float | None = None,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.d = x.shape[0]
        if task_type == "finetune_dtvu":
            assert method in ["time", "time_span", "time_pred"]
            assert x.shape == torch.Size([self.d, self.t, self.v, self.u2]), x.shape
            assert targets.shape == torch.Size([self.d, self.t, self.u0]), targets.shape
        elif task_type == "finetune_dtu":
            assert method in ["time", "time_span", "time_pred"]
            assert x.shape == torch.Size([self.d, self.t, self.u0]), x.shape
            assert targets.shape == torch.Size([self.d, self.t, self.u0]), targets.shape
        elif task_type == "pretrain":
            assert x.shape == torch.Size([self.d, self.t, self.v]), x.shape
            assert targets.shape == torch.Size([self.d, self.t, self.v]), targets.shape
        else:
            raise NotImplementedError

        attn_mask_v, attn_mask_g = self.gen_attn_mask(task_type, method, history_ratio, preds_ratio)
        x, marked_matrix, marked_targets = self.get_marked_data(x, method, task_type, targets, mark_ratio, history_ratio, preds_ratio)

        # 后处理掩码矩阵，方便计算ACC指标
        if task_type == "pretrain":
            marked_matrix = marked_matrix.flatten()
            assert marked_matrix.shape == torch.Size([self.d * self.t * self.v]), marked_matrix.shape
        elif task_type == "finetune_dtvu":
            marked_matrix = marked_matrix[:, :, 0, 0].unsqueeze(dim=-1)
            marked_matrix = marked_matrix.repeat(1, 1, self.u0)
            marked_matrix = marked_matrix.flatten()
            assert marked_matrix.shape == torch.Size([self.d * self.t * self.u0]), marked_matrix.shape
        elif task_type == "finetune_dtu":
            marked_matrix = marked_matrix.flatten()
            assert marked_matrix.shape == torch.Size([self.d * self.t * self.u0]), marked_matrix.shape
        else:
            raise NotImplementedError

        return x, attn_mask_v, attn_mask_g, marked_matrix, marked_targets

    def gen_attn_mask(
            self,
            task_type: tp.Literal["pretrain", "finetune_dtvu", "finetune_dtu"],
            method: tp.Literal["random", "space", "time", "time_span", "time_pred"],
            history_ratio: float | None = None,
            preds_ratio: float | None = None,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # ==========================================================
        # ==========================================================
        if task_type == "pretrain" or task_type == "finetune_dtvu":
            dim_v = self.v
            dim_g = self.g
        elif task_type == "finetune_dtu":
            dim_v = self.u0
            dim_g = None
        else:
            raise NotImplementedError
        # ==========================================================
        # ==========================================================
        if method in ["random", "space", "time", "time_span"]:
            attn_mask_v = torch.ones(size=(self.d, self.t, dim_v))
            attn_mask_g = torch.ones(size=(self.d, self.t, dim_g)) if dim_g is not None else None
        elif method == "time_pred":
            assert history_ratio + preds_ratio <= 1.0
            history_t = int(history_ratio * self.t)
            preds_t = int(preds_ratio * self.t)
            pad_t = self.t - history_t - preds_t
            attn_mask_v = torch.concat([
                torch.ones(size=(self.d, history_t + preds_t, dim_v)),
                torch.zeros(size=(self.d, pad_t, dim_v)),
            ], dim=1)
            attn_mask_g = torch.concat([
                torch.ones(size=(self.d, history_t + preds_t, dim_g)),
                torch.zeros(size=(self.d, pad_t, dim_g)),
            ], dim=1) if dim_g is not None else None
        else:
            raise NotImplementedError
        # ==========================================================
        # ==========================================================
        return attn_mask_v, attn_mask_g

    def get_marked_data(
            self,
            x: torch.Tensor,
            method: tp.Literal["random", "space", "time", "time_span", "time_pred"],
            task_type: tp.Literal["pretrain", "finetune_dtvu", "finetune_dtu"],
            targets: torch.Tensor,
            mark_ratio: float | None = None,
            history_ratio: float | None = None,
            preds_ratio: float | None = None,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if method in ["random", "space", "time", "time_span"]:
            assert mark_ratio is not None
        elif method == "time_pred":
            assert history_ratio is not None and preds_ratio is not None
        else:
            raise NotImplementedError

        if task_type == "pretrain":
            assert targets.shape == torch.Size([self.d, self.t, self.v])
        elif task_type == "finetune_dtvu" or task_type == "finetune_dtu":
            assert targets.shape == torch.Size([self.d, self.t, self.u0])
        else:
            raise NotImplementedError

        if method == "random":
            assert task_type == "pretrain"
            marked_matrix = torch.bernoulli(torch.full(size=(self.d, self.t, self.v), fill_value=mark_ratio)).bool()
            targets[~marked_matrix] = -100
            x[marked_matrix] = 0  # 如果有100个离散值，那么0代表mask，1~100代表数值

        elif method == "space":
            assert task_type == "pretrain"
            marked_matrix = torch.bernoulli(torch.full(size=(self.d, self.v), fill_value=mark_ratio)).bool()
            marked_matrix = marked_matrix.unsqueeze(dim=1).repeat(1, self.t, 1)
            targets[~marked_matrix] = -100
            x[marked_matrix] = 0  # 如果有100个离散值，那么0代表mask，1~100代表数值

        elif method == "time":
            marked_matrix = torch.bernoulli(torch.full(size=(self.d, self.t), fill_value=mark_ratio)).bool()
            x, marked_matrix, targets = self.utils_dt_mask_transfer(x, task_type, marked_matrix, targets)

        elif method == "time_span":
            t_mid = np.random.randint(low=0, high=self.t)
            t_span = int(mark_ratio * self.t)
            t_start = max(0, t_mid - t_span // 2)
            t_end = min(t_mid + t_span // 2, self.t)
            marked_matrix = torch.full(size=(self.d, self.t), fill_value=False)
            marked_matrix[:, t_start:t_end] = True
            x, marked_matrix, targets = self.utils_dt_mask_transfer(x, task_type, marked_matrix, targets)

        elif method == "time_pred":
            t_history = int(history_ratio * self.t)
            t_preds = int(preds_ratio * self.t)
            marked_matrix = torch.full(size=(self.d, self.t), fill_value=False)
            marked_matrix[:, t_history:t_history + t_preds] = True
            x, marked_matrix, targets = self.utils_dt_mask_transfer(x, task_type, marked_matrix, targets)

        else:
            raise NotImplementedError

        return x, marked_matrix, targets

    def utils_dt_mask_transfer(
            self,
            x: torch.Tensor,
            task_type: tp.Literal["pretrain", "finetune_dtvu", "finetune_dtu"],
            dt_mask: torch.Tensor,
            targets: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if task_type == "pretrain":
            marked_matrix = dt_mask.unsqueeze(dim=2).repeat(1, 1, self.v)
            targets[~marked_matrix] = -100
            x[marked_matrix] = 0
        elif task_type == "finetune_dtvu":
            marked_matrix = dt_mask.unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, self.v, self.u2)
            tmp = dt_mask.unsqueeze(dim=2).repeat(1, 1, self.u0)
            targets[~tmp] = -100
            mark_tensor = torch.zeros(size=(1, 1, 1, self.u2,))
            mark_tensor[0, 0, 0, 0] = 1
            mark_tensor = mark_tensor.repeat(self.d, self.t, self.v, 1)
            x = x.float()
            x[marked_matrix] = mark_tensor[marked_matrix]
        elif task_type == "finetune_dtu":
            marked_matrix = dt_mask.unsqueeze(dim=2).repeat(1, 1, self.u0)
            targets[~marked_matrix] = -100
            x[marked_matrix] = 0
        else:
            raise NotImplementedError

        return x, marked_matrix, targets


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1000)

    # nd = 1
    # nt = 10
    # nv = 12
    # ne = 256
    # nu = 5

    # data_pretrain = torch.randint(low=1, high=nu+1, size=(nd, nt, nv))
    # target_pretrain = data_pretrain.clone()
    #
    # mm = MaskManipulator(
    #     num_ticks=nt,
    #     num_nodes=nv,
    #     num_users=nu,
    #     num_emb_dims=ne,
    # )
    #
    # print(data_pretrain)
    #
    # data_pretrain, attn_maskk, mk_matrix, target_pretrain = mm.mask(
    #     x=data_pretrain,
    #     method="random",
    #     task_type="pretrain",
    #     targets=target_pretrain,
    #     mark_ratio=0.5
    # )
    #
    # print(data_pretrain)
    # print(target_pretrain)
    # print(mk_matrix.int())

    # nd = 1
    # nt = 10
    # nv = 4
    # nu = 5
    #
    # data_finetune = torch.randint(low=0, high=2, size=(nd, nt, nv, nu + 1))
    # xxx = torch.zeros(size=(nd, nt, nv, 1))
    # data_finetune = torch.concat([xxx, data_finetune], dim=-1)
    # target_finetune = torch.randint(low=0, high=nu, size=(nd, nt, nu))
    #
    # mm = MaskManipulator(
    #     num_ticks=nt,
    #     num_nodes=nv,
    #     num_users=nu,
    # )
    #
    # # print(data_finetune)
    #
    # data_finetune, attn_finetune, mk_matrix, target_finetune = mm.mask(
    #     x=data_finetune,
    #     method="time_pred",
    #     task_type="finetune",
    #     targets=target_finetune,
    #     mark_ratio=0.5,
    #     history_ratio=0.4,
    #     preds_ratio=0.3,
    # )
    #
    # # print(data_finetune)
    # print(target_finetune)
    # # print(mk_matrix.int())
