import torch
import torch.nn as nn
import torch.utils.data as tdata
import torch.optim as optim
import typing as tp
import random
import rich.progress as prgs

import code_03_nnmodel.nnmodel_07_mask_manipulator as nn7

import code_04_trainer.trainer_02_cache as tr1
import code_04_trainer.trainer_03_metric as tr2
import code_04_trainer.trainer_04_logger as tr3


class MyTrainer:
    def __init__(
            self,
            model: nn.Module,
            task_type: tp.Literal["pretrain", "finetune_dtvu", "finetune_dtu"],
            mask_manipulator: nn7.MaskManipulator,
            optimizer: optim.Optimizer,
            scheduler: tp.Any,
            scheduler_policy: tp.Literal["epoch", "step", None],
            device: torch.device | str,
            log_path: str | None = None,
    ):
        self.model = model.to(device)
        self.task_type = task_type
        self.mask_manipulator = mask_manipulator

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_policy = scheduler_policy
        if self.scheduler is None:
            assert scheduler_policy is None
        else:
            assert scheduler_policy is not None

        self.device = device
        self.log_path = log_path

        if self.task_type == "pretrain":
            num_classes = self.mask_manipulator.u1
        elif self.task_type == "finetune_dtvu":
            num_classes = self.mask_manipulator.v
        elif self.task_type == "finetune_dtu":
            num_classes = self.mask_manipulator.v + 1
        else:
            num_classes = self.mask_manipulator.v

        self.metric = tr2.MetricsCalculator(num_classes=num_classes)
        self.cache = tr1.TrainerCache()
        self.logger = tr3.TrainerLogger(log_path=log_path)

        if task_type == "pretrain":
            self.mask_method_prob = {
                "random": 0.2,
                "space": 0.3,
                "time": 0.2,
                "time_span": 0.1,
                "time_pred": 0.2,
            }
        elif task_type == "finetune_dtvu" or task_type == "finetune_dtu":
            self.mask_method_prob = {
                "time_span": 0.3,
                "time_pred": 0.7,
            }

    def train_step(
            self,
            x: torch.Tensor,
            targets: torch.Tensor,
            clip_grad_norm_factor: float | None = None,
            enable_v2g: bool = False,
    ):
        self.model.train()
        self.optimizer.zero_grad()

        mask_method = random.choices(
            population=list(self.mask_method_prob.keys()),
            weights=list(self.mask_method_prob.values()),
            k=1
        )[0]

        if mask_method in ["random", "space", "time", "time_span"]:
            mask_ratio = random.uniform(0.2, 0.4)
            history_ratio = None
            preds_ratio = None
        elif mask_method == "time_pred":
            mask_ratio = None
            history_ratio = random.uniform(0.3, 0.59)
            preds_ratio = random.uniform(0.2, 0.39)
        else:
            raise NotImplementedError

        x, attn_mask_v, attn_mask_g, marked_matrix, targets = self.mask_manipulator.mask(
            x=x,
            method=mask_method,
            task_type=self.task_type,
            targets=targets,
            mark_ratio=mask_ratio,
            history_ratio=history_ratio,
            preds_ratio=preds_ratio,
        )

        x = x.to(self.device)
        attn_mask_v = attn_mask_v.to(self.device)
        attn_mask_g = attn_mask_g.to(self.device) if attn_mask_g is not None else None
        marked_matrix = marked_matrix.to(self.device)
        targets = targets.to(self.device)

        preds, loss, targets = self.model(
            x=x,
            attn_mask_v=attn_mask_v,
            attn_mask_g=attn_mask_g,
            targets=targets,
            enable_v2g=enable_v2g,
        )

        loss.backward()

        if clip_grad_norm_factor is not None:
            nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=clip_grad_norm_factor
            )

        self.optimizer.step()

        if self.scheduler is not None and self.scheduler_policy == "step":
            self.scheduler.step()

        return preds, targets, marked_matrix, loss

    def train_epoch(
            self,
            loader: tdata.DataLoader,
            epoch_index: int,
            clip_grad_norm_factor: float | None = None,
            enable_v2g: bool = False,
    ):
        self.model.train()
        self.cache.clear_cache()

        for step_index, (x, targets) in prgs.track(sequence=enumerate(loader), total=len(loader), description=f"[TRN] epoch_{epoch_index:02d}"):
            preds, targets, marked_matrix, loss = self.train_step(
                x=x,
                targets=targets,
                clip_grad_norm_factor=clip_grad_norm_factor,
                enable_v2g=enable_v2g
            )

            self.cache.update_cache(
                preds=preds,
                targets=targets,
                loss=loss,
                marked_matrix=marked_matrix
            )

        if self.scheduler is not None and self.scheduler_policy == "epoch":
            self.scheduler.step()

        result_dict = {
            "trn/loss": self.cache.loss_cache / self.cache.num_samples_cache,
            "trn/acc@01": self.metric.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=1),
            "trn/acc@05": self.metric.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=5),
            "trn/acc@10": self.metric.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=10),
            "trn/acc@20": self.metric.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=20),
        }

        self.logger.info("\n")
        self.logger.info(f"[TRN] epoch_{epoch_index:02d}")
        self.logger.info("=============================================")
        for k, v in result_dict.items():
            self.logger.info(f"{k} = {v:.4f}")
        self.logger.info("\n")

        return result_dict

    @torch.no_grad()
    def validate_step_dynamic(
            self,
            x: torch.Tensor,
            targets: torch.Tensor,
            history_step: int | None = None,
            preds_step: int | None = None,
            enable_v2g: bool = False,
    ):
        self.model.eval()

        if self.task_type == "pretrain":
            mask_method = random.choices(
                population=list(self.mask_method_prob.keys()),
                weights=list(self.mask_method_prob.values()),
                k=1
            )[0]
            if mask_method in ["random", "space", "time", "time_span"]:
                mask_ratio = random.uniform(0.2, 0.4)
                history_ratio = None
                preds_ratio = None
            elif mask_method == "time_pred":
                mask_ratio = None
                history_ratio = random.uniform(0.3, 0.59)
                preds_ratio = random.uniform(0.2, 0.39)
            else:
                raise NotImplementedError
        elif self.task_type == "finetune_dtvu" or self.task_type == "finetune_dtu":
            mask_method = "time_pred"
            assert history_step is not None and preds_step is not None
            assert history_step + preds_step <= self.mask_manipulator.t
            history_ratio = history_step / self.mask_manipulator.t + 0.001
            preds_ratio = preds_step / self.mask_manipulator.t + 0.001
            mask_ratio = None
        else:
            raise NotImplementedError

        x, attn_mask_v, attn_mask_g, marked_matrix, targets = self.mask_manipulator.mask(
            x=x,
            method=mask_method,
            task_type=self.task_type,
            targets=targets,
            mark_ratio=mask_ratio,
            history_ratio=history_ratio,
            preds_ratio=preds_ratio,
        )

        x = x.to(self.device)
        attn_mask_v = attn_mask_v.to(self.device)
        attn_mask_g = attn_mask_g.to(self.device) if attn_mask_g is not None else None
        marked_matrix = marked_matrix.to(self.device)
        targets = targets.to(self.device)

        preds, loss, targets = self.model(
            x=x,
            attn_mask_v=attn_mask_v,
            attn_mask_g=attn_mask_g,
            targets=targets,
            enable_v2g=enable_v2g,
        )

        return preds, targets, marked_matrix, loss

    @torch.no_grad()
    def validate_epoch_dynamic(
            self,
            loader: tdata.DataLoader,
            epoch_index: int,
            history_step: tuple | None = None,
            preds_step: int | None = None,
            enable_v2g: bool = False,
    ):
        self.model.eval()
        self.cache.clear_cache()

        if self.task_type == "pretrain":
            val_type = "val_pretrain"
        elif self.task_type == "finetune_dtvu" or self.task_type == "finetune_dtu":
            assert history_step is not None and preds_step is not None
            # val_type = f"val_{self.task_type}_hist({history_step})_pred({preds_step})"
            # val_type = f"val_{self.task_type}_steps_{preds_step}"
            val_type = f"val_finetune_dtvu_steps_{preds_step}" # 懒得改了
            assert len(history_step) == len(loader), f"{len(history_step)}, {len(loader)}"
        else:
            raise NotImplementedError

        for step_index, (x, targets) in prgs.track(sequence=enumerate(loader), total=len(loader), description=f"[{val_type.upper()}] epoch_{epoch_index:02d}"):
            preds, targets, marked_matrix, loss = self.validate_step_dynamic(
                x=x,
                targets=targets,
                history_step=history_step[step_index],
                preds_step=preds_step,
                enable_v2g=enable_v2g,
            )

            self.cache.update_cache(
                preds=preds,
                targets=targets,
                loss=loss,
                marked_matrix=marked_matrix
            )

        result_dict = {
            f"{val_type}/loss": self.cache.loss_cache / self.cache.num_samples_cache,
            f"{val_type}/acc@01": self.metric.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=1),
            f"{val_type}/acc@05": self.metric.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=5),
            f"{val_type}/acc@10": self.metric.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=10),
            f"{val_type}/acc@20": self.metric.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=20),
        }

        self.logger.info("\n")
        self.logger.info(f"[{val_type.upper()}] epoch_{epoch_index:02d}")
        self.logger.info("=============================================")
        for k, v in result_dict.items():
            self.logger.info(f"{k} = {v:.4f}")
        self.logger.info("\n")

        return result_dict
