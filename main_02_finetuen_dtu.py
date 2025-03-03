import torch
import torch.utils.data as tdata
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.tensorboard as tsb
import transformers as tfm
import typing as tp
import time
import os
import collections
import random

import code_01_utils.utils_02_confg_loader as ut2
import code_03_nnmodel.nnmodel_05_space_time_backbone_sequential as nn5
import code_03_nnmodel.nnmodel_06_space_time_backbone_parallel as nn6
import code_03_nnmodel.nnmodel_07_mask_manipulator as nn7
import code_04_trainer.trainer_01_core as tr1


def main_finetuen_dtu(
        # 小样本
        ex_num_samples: int,
        ex_num_users: int,

        # 任务类型
        task_type: tp.Literal["pretrain", "finetune_dtvu", "finetune_dtu"] = "finetune_dtu",

        # dataset
        dataset_path: str = "data_02_preprocessed_data/YJMob100K/p2_tensor/DTU_TARGETS_DURATION_40_DAY_14_53_UID_80_SPAN_01.pth",
        batch_size: int = 1,
        val_batch_size: int = 1,

        # model
        model_type: tp.Literal["sequential", "parallel"] = "sequential",
        device: str = "cpu",
        config_path: str = "code_00_configs/modernbert_config.json",
        global_seed: int | None = None,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        num_sub_graphs: int = 128,

        # trainer
        tensorboard_dir: str | None = None,
        learning_rate: float = 1e-3,
        scheduler_policy: tp.Literal["epoch", "step", None] = None,
        weight_decay: float = 1e-2,
        clip_grad_norm_factor: float | None = None,
        enable_v2g: bool = False,
        num_epochs: int = 40,

        # eval
        history_step: int | tuple = 12,
        # preds_step: int = 3,
):
    assert task_type == "finetune_dtu"

    # load dataset
    finetune_data = torch.load(dataset_path)
    num_samples, num_ticks, num_users = finetune_data.shape
    num_nodes = int(torch.max(finetune_data).item())

    # 小样本
    tfm.set_seed(888)
    finetune_data = finetune_data[torch.randperm(num_samples), :, :]
    num_samples = ex_num_samples
    num_users = ex_num_users
    finetune_data = finetune_data[:num_samples, :, :num_users]

    # 划分数据集
    size_trn = int(0.8 * num_samples)

    # static_val_set params
    if isinstance(history_step, int):
        history_steps = [history_step for _ in range(num_epochs)]
    elif isinstance(history_step, tuple):
        random.seed(42)
        assert batch_size == 1 and val_batch_size == 1
        tmp = [int(num_ticks * random.uniform(*history_step)) for _ in range(num_samples - size_trn)]
        history_steps = [tmp for _ in range(num_epochs)]
    else:
        raise TypeError("[!] history_step must be int or tuple")

    # cuda seed
    if global_seed is not None:
        tfm.set_seed(global_seed)
    exp_time = time.strftime("%Y%m%d-%H%M%S")

    # data loader
    trn_set = tdata.TensorDataset(finetune_data[:size_trn], finetune_data[:size_trn].clone())
    val_set = tdata.TensorDataset(finetune_data[size_trn:], finetune_data[size_trn:].clone())
    trn_loader = tdata.DataLoader(trn_set, batch_size=batch_size, shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=val_batch_size, shuffle=False)
    # =======================================================
    # =======================================================
    if tensorboard_dir is not None:
        dataset_name = dataset_path.replace("data_02_preprocessed_data/YJMob100K/p2_tensor/", "")[:-4]
        os.makedirs(f"{tensorboard_dir}/{dataset_name}/EXP-{exp_time}-{model_type.upper()}", exist_ok=True)
        tsb_writer = tsb.SummaryWriter(log_dir=f"{tensorboard_dir}/{dataset_name}/EXP-{exp_time}-{model_type.upper()}")
    else:
        dataset_name = None
        tsb_writer = None
    # =======================================================
    # =======================================================
    # models
    modernbert_cfg = ut2.load_modernbert_json_config(
        json_file_path=config_path,
        num_layers=num_layers,
        num_heads=num_heads,
        emb_dim=embed_dim,
    )
    # =======================================================
    # =======================================================
    if model_type == "sequential":
        model_class = nn5.SpaceTimeBackboneSequential
    elif model_type == "parallel":
        model_class = nn6.SpaceTimeBackboneParallel
    else:
        raise NotImplementedError
    model = model_class(
        modernbert_config=modernbert_cfg,
        num_layers_space=num_layers // 2,
        num_layers_time=num_layers // 2,
        num_nodes=num_nodes,
        num_ticks=num_ticks,
        num_users=num_users,
        num_sub_graphs=num_sub_graphs,
        is_multi_variant=True,
        task_type=task_type,
    )
    # =======================================================
    # =======================================================
    mask_manipulator = nn7.MaskManipulator(
        num_nodes=num_nodes,
        num_ticks=num_ticks,
        num_users=num_users,
        num_sub_graphs=num_sub_graphs,
    )
    # =======================================================
    # =======================================================
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # =======================================================
    # =======================================================
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=8,
    ) if scheduler_policy is not None else None
    # =======================================================
    # =======================================================
    trainer = tr1.MyTrainer(
        model=model,
        task_type=task_type,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_policy=scheduler_policy,
        mask_manipulator=mask_manipulator,
        device=device,
        log_path=f"{tensorboard_dir}/{dataset_name}/EXP-{exp_time}-{model_type.upper()}/exp.log" if tensorboard_dir is not None else None,
    )
    # =======================================================
    # =======================================================
    # training
    best_acc_record = collections.defaultdict(float)
    # =======================================================
    # =======================================================
    for epoch_index in range(num_epochs):
        epoch_record = dict(
            trn_record=trainer.train_epoch(
                loader=trn_loader,
                epoch_index=epoch_index,
                clip_grad_norm_factor=clip_grad_norm_factor,
                enable_v2g=enable_v2g,
            ),
            val_record_step1=trainer.validate_epoch_dynamic(
                loader=val_loader,
                epoch_index=epoch_index,
                history_step=history_steps[epoch_index],
                preds_step=1,
                enable_v2g=enable_v2g,
            ),
            val_record_step2=trainer.validate_epoch_dynamic(
                loader=val_loader,
                epoch_index=epoch_index,
                history_step=history_steps[epoch_index],
                preds_step=3,
                enable_v2g=enable_v2g,
            ),
        )
        # =======================================================
        if tensorboard_dir is None:
            continue
        # =======================================================
        # tensorboard
        for record_key, record_value in epoch_record.items():
            for metric_key, metric_value in record_value.items():
                tsb_writer.add_scalar(metric_key, metric_value, epoch_index)
                if "val" in metric_key and "acc" in metric_key:
                    best_metric_key = metric_key.replace("/", "_best/")
                    if metric_value > best_acc_record[best_metric_key]:
                        best_acc_record[best_metric_key] = metric_value
                        # save_path = os.path.join(
                        #     tensorboard_path.replace("record_01_log", "data_04_checkpoint"),
                        #     best_metric_key.replace("/", "_").replace("@", "_") + ".pkl",
                        # )
                        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        # torch.save(model.state_dict(), save_path)
                    tsb_writer.add_scalar(best_metric_key, best_acc_record[best_metric_key], epoch_index)
            tsb_writer.flush()


if __name__ == "__main__":
    import functools

    main_finetuen_dtu_partial = functools.partial(
        main_finetuen_dtu,
        task_type="finetune_dtu",
        dataset_path="data_02_preprocessed_data/YJMob100K/p2_tensor/DTU_TARGETS_DURATION_20_DAY_07_26_UID_20_SPAN_01.pth",
        batch_size=1,
        val_batch_size=1,
        model_type="sequential",
        device="cuda:0",
        config_path="code_00_configs/modernbert_config_dropout.json",
        # global_seed=42,
        embed_dim=128,
        num_heads=4,
        num_layers=12,
        tensorboard_dir="record_01_tensorboard",
        learning_rate=5e-4,
        weight_decay=1e-4,
        scheduler_policy="epoch",
        clip_grad_norm_factor=0.1,
        num_epochs=64,
        history_step=(0.4, 0.8),
        # preds_step=2,
    )

    main_finetuen_dtu_partial(global_seed=0, ex_num_samples=20, ex_num_users=5)
    main_finetuen_dtu_partial(global_seed=1, ex_num_samples=20, ex_num_users=5)
    main_finetuen_dtu_partial(global_seed=2, ex_num_samples=20, ex_num_users=5)
