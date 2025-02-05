import torch
import torch.nn as nn
import torch.utils.data as tdata
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
import typing as tp
import argparse

import code_03_nnmodel.nnmodel_00_unist_core as nn0
import code_04_trainer.trainer_01_cache as t1
import code_04_trainer.trainer_02_metrics as t2
import code_04_trainer.trainer_03_logger as t3


class UniTrainer:
    def __init__(
            self,
            data_shapes: argparse.Namespace,
            model: nn0.UniST,
            optimizer: optim.Optimizer,
            scheduler,
            device: torch.device,
            log_path: str | None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.cache = t1.TrainerCache()
        self.metrics_calculator = t2.MetricsCalculator(num_labels=data_shapes.raw_x * data_shapes.raw_y)
        self.logger = t3.TrainerLogger(log_path=log_path)

    def train_step(
            self,
            x: torch.Tensor,  # 👈 <n>, <c>, <raw_t>, <raw_x>, <raw_y>
            x_timestamp: torch.Tensor,  # 👈 <n>, <raw_t>, <channel_time>
            x_period: torch.Tensor,  # 👈 <n>, <p>period, <c>, <raw_t_future>, <raw_x>, <raw_y>
            use_timestamp_embeddings: bool,
            mask_ratio: float,
            mask_mode: tp.Literal["random", "tube", "block", "frame", "temporal"],
            encode_mode: tp.Literal["forward", "backward"],
            prompt_mode: tp.Literal[None, "s", "p", "c", "sp", "pc", "sc", "spc"],
            clip_grad_norm_factor: float | None,
    ):
        self.model.train()
        self.optimizer.zero_grad()

        x = x.to(self.device)
        x_timestamp = x_timestamp.to(self.device)
        x_period = x_period.to(self.device)

        (
            (
                preds_masked,
                targets_masked,
                loss_masked
            ),
            (_, _, _)
        ) = self.model.forward(
            x=x,
            x_period=x_period,
            x_timestamp=x_timestamp,
            use_timestamp_embeddings=use_timestamp_embeddings,
            mask_ratio=mask_ratio,
            mask_mode=mask_mode,
            encode_mode=encode_mode,
            prompt_mode=prompt_mode,
        )

        loss_masked.backward()
        if clip_grad_norm_factor is not None:
            nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=clip_grad_norm_factor
            )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return preds_masked, targets_masked, loss_masked

    @torch.no_grad()
    def validate_step(
            self,
            x: torch.Tensor,  # 👈 <n>, <c>, <raw_t>, <raw_x>, <raw_y>
            x_timestamp: torch.Tensor,  # 👈 <n>, <raw_t>, <channel_time>
            x_period: torch.Tensor,  # 👈 <n>, <p>period, <c>, <raw_t_future>, <raw_x>, <raw_y>
            use_timestamp_embeddings: bool,
            mask_ratio: float,
            mask_mode: tp.Literal["random", "tube", "block", "frame", "temporal"],
            encode_mode: tp.Literal["forward", "backward"],
            prompt_mode: tp.Literal[None, "s", "p", "c", "sp", "pc", "sc", "spc"],
    ):
        self.model.eval()

        x = x.to(self.device)
        x_timestamp = x_timestamp.to(self.device)
        x_period = x_period.to(self.device)

        (
            (
                preds_masked,
                targets_masked,
                loss_masked
            ),
            (_, _, _)
        ) = self.model.forward(
            x=x,
            x_period=x_period,
            x_timestamp=x_timestamp,
            use_timestamp_embeddings=use_timestamp_embeddings,
            mask_ratio=mask_ratio,
            mask_mode=mask_mode,
            encode_mode=encode_mode,
            prompt_mode=prompt_mode,
        )

        return preds_masked, targets_masked, loss_masked

    def train_epoch(
            self,
            train_loader: tdata.DataLoader,
            num_epochs: int,
            use_timestamp_embeddings: bool,
            mask_ratio: float,
            mask_mode: tp.Literal["random", "tube", "block", "frame", "temporal"],
            encode_mode: tp.Literal["forward", "backward"],
            prompt_mode: tp.Literal[None, "s", "p", "c", "sp", "pc", "sc", "spc"],
            clip_grad_norm_factor: float | None,
    ):
        self.cache.clear_cache()
        for num_steps, (x, x_timestamp, x_period) in enumerate(train_loader):
            preds_masked, targets_masked, loss_masked = self.train_step(
                x=x,
                x_timestamp=x_timestamp,
                x_period=x_period,
                use_timestamp_embeddings=use_timestamp_embeddings,
                mask_ratio=mask_ratio,
                mask_mode=mask_mode,
                encode_mode=encode_mode,
                prompt_mode=prompt_mode,
                clip_grad_norm_factor=clip_grad_norm_factor,
            )
            self.cache.update_cache(preds_masked, targets_masked, loss_masked)

        result_dict = {
            "trn/loss": self.cache.loss_cache / self.cache.num_samples_cache,
            "trn/acc@1": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=1),
            "trn/acc@5": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=5),
            "trn/acc@10": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=10),
            "trn/acc@20": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=20),
            "trn/acc@50": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=50),
        }

        self.logger.info("\n")
        self.logger.info(f"trn @ epoch_{num_epochs:02d}")
        self.logger.info("=============================================")
        for k, v in result_dict.items():
            self.logger.info(f"{k} = {v:.4f}")

        return result_dict

    @torch.no_grad()
    def validate_epoch(
            self,
            validate_loader: tdata.DataLoader,
            num_epochs: int,
            use_timestamp_embeddings: bool,
            mask_ratio: float,
            mask_mode: tp.Literal["random", "tube", "block", "frame", "temporal"],
            encode_mode: tp.Literal["forward", "backward"],
            prompt_mode: tp.Literal[None, "s", "p", "c", "sp", "pc", "sc", "spc"],
    ):
        self.cache.clear_cache()
        for num_steps, (x, x_timestamp, x_period) in enumerate(validate_loader):
            preds_masked, targets_masked, loss_masked = self.validate_step(
                x=x,
                x_timestamp=x_timestamp,
                x_period=x_period,
                use_timestamp_embeddings=use_timestamp_embeddings,
                mask_ratio=mask_ratio,
                mask_mode=mask_mode,
                encode_mode=encode_mode,
                prompt_mode=prompt_mode,
            )
            self.cache.update_cache(preds_masked, targets_masked, loss_masked)

        result_dict = {
            "val/loss": self.cache.loss_cache / self.cache.num_samples_cache,
            "val/acc@1": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=1),
            "val/acc@5": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=5),
            "val/acc@10": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=10),
            "val/acc@20": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=20),
            "val/acc@50": self.metrics_calculator.topk_accuracy(self.cache.preds_cache, self.cache.targets_cache, topk=50),
        }

        self.logger.info("\n")
        self.logger.info(f"val @ epoch_{num_epochs:02d}")
        self.logger.info("=============================================")
        for k, v in result_dict.items():
            self.logger.info(f"{k} = {v:.4f}")

        return result_dict


if __name__ == "__main__":
    import argparse
    import code_03_nnmodel.nnmodel_00_unist_core as nn0

    my_shapes = argparse.Namespace()

    my_shapes.kernel_time = 2
    my_shapes.kernel_space = 4
    my_shapes.p = 5

    my_shapes.channel_time = 1
    my_shapes.channel_variant = 17

    my_shapes.raw_t = 8
    my_shapes.raw_t_history = 6
    my_shapes.raw_t_future = 2
    my_shapes.raw_x = 8
    my_shapes.raw_y = 8

    my_shapes.t = my_shapes.raw_t // my_shapes.kernel_time
    my_shapes.x = my_shapes.raw_x // my_shapes.kernel_space
    my_shapes.y = my_shapes.raw_y // my_shapes.kernel_space
    my_shapes.t_history = my_shapes.raw_t_history // my_shapes.kernel_time
    my_shapes.t_future = my_shapes.raw_t_future // my_shapes.kernel_time

    n = 77
    emb_size = 128

    num_classes = my_shapes.raw_x * my_shapes.raw_y
    xxx = torch.nn.functional.one_hot(
        torch.randint(
            low=0,
            high=num_classes,
            size=(n * my_shapes.channel_variant * my_shapes.raw_t,)
        ),
        num_classes=num_classes,
    ).reshape(
        n,
        my_shapes.channel_variant,
        my_shapes.raw_t,
        my_shapes.raw_x,
        my_shapes.raw_y,
    ).float()

    xxx_period = torch.nn.functional.one_hot(
        torch.randint(
            low=0,
            high=num_classes,
            size=(n * my_shapes.p * my_shapes.channel_variant * my_shapes.raw_t,)
        ),
        num_classes=num_classes,
    ).reshape(
        n,
        my_shapes.p,
        my_shapes.channel_variant,
        my_shapes.raw_t,
        my_shapes.raw_x,
        my_shapes.raw_y,
    ).float()

    xxx_ts = torch.tensor(range(my_shapes.raw_t)).reshape(1, -1, my_shapes.channel_time).repeat(n, 1, 1)

    print(xxx.shape)
    print(xxx_ts.shape)
    print(xxx_period.shape)

    dataset = torch.utils.data.TensorDataset(
        xxx,
        xxx_ts,
        xxx_period
    )
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=False)

    model = nn0.UniST(
        encoder_depth=8,
        encoder_heads=4,
        decoder_depth=8,
        decoder_heads=4,

        data_shapes=my_shapes,
        emb_dim=emb_size,

        memory_dim_time=512,
        memory_dim_space=512,
        num_conv_layers_parallel=3,
        num_attn_heads=4,
        num_attn_layers_sequential=1,
        space_prompt_implement_id="impl-1"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    trainer = UniTrainer(
        data_shapes=my_shapes,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cuda:0"),
        log_path=None
    )

    eps = 20
    for idx in range(eps):
        result = trainer.train_epoch(
            train_loader=loader,
            num_epochs=idx,
            use_timestamp_embeddings=True,
            mask_ratio=0.25,
            mask_mode="frame",
            encode_mode="forward",
            # prompt_mode="spc",
            prompt_mode=None,
            clip_grad_norm_factor=0.05
        )

    # for idx in range(eps):
    #     result = trainer.validate_epoch(
    #         validate_loader=loader,
    #         num_epochs=idx,
    #         use_timestamp_embeddings=True,
    #         mask_ratio=0.25,
    #         mask_mode="frame",
    #         encode_mode="forward",
    #         prompt_mode="spc",
    #         # clip_grad_norm_factor=0.05
    #     )
