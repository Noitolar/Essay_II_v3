import torch
import torch.utils.data as tdata

import code_01_utils.utils_02_confg_loader as ut2
import code_03_nnmodel.nnmodel_05_space_time_backbone_sequential as nn5
import code_03_nnmodel.nnmodel_07_mask_manipulator as nn7
import code_03_nnmodel.nnmodel_08_side_branches_sequential as nn8

if __name__ == "__main__":
    pretrain_data = torch.load("data_02_preprocessed_data/YJMob100K/p2_tensor/DTU_TARGETS_DURATION_40_DAY_14_53_UID_80_SPAN_01.pth", )
    num_samples, num_ticks, num_users = pretrain_data.shape
    num_nodes = int(torch.max(pretrain_data).item())
    num_sub_graphs = 128
    size_trn = int(0.8 * num_samples)

    trn_set = tdata.TensorDataset(pretrain_data[:size_trn], pretrain_data[:size_trn].clone())
    val_set = tdata.TensorDataset(pretrain_data[size_trn:], pretrain_data[size_trn:].clone())

    trn_loader = tdata.DataLoader(trn_set, batch_size=1, shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=4, shuffle=False)

    backbone = nn5.SpaceTimeBackboneSequential(
        num_layers_space=2,
        num_layers_time=2,
        num_nodes=num_nodes,
        num_ticks=num_ticks,
        num_users=num_users,
        num_sub_graphs=num_sub_graphs,
        is_multi_variant=True,
        task_type="finetune_dtu",
        modernbert_config=ut2.load_modernbert_json_config(
            json_file_path="code_00_configs/modernbert_config_dropout.json",
            num_layers=4,
            num_heads=4,
            emb_dim=128
        ),
    )

    model = nn8.SideBranchesSequential(
        task_type="finetune_dtu",
        backbone=backbone,
        branch_modernbert_config=ut2.load_modernbert_json_config(
            json_file_path="code_00_configs/modernbert_config_dropout.json",
            emb_dim=32,
            num_heads=2
        ),
    )

    mask_manipulator = nn7.MaskManipulator(
        num_nodes=num_nodes,
        num_ticks=num_ticks,
        num_users=num_users,
        num_sub_graphs=num_sub_graphs,
    )

    # device = "cuda:0"
    device = "cpu"
    model = model.to(device)

    import time

    print("[+] START TRAINING")
    ctr = time.perf_counter()

    for batch, targets in trn_loader:
        batch, attn_mask_v, attn_mask_g, marked_matrix, targets = mask_manipulator.mask(
            x=batch,
            method="time",
            task_type="finetune_dtu",
            targets=targets,
            mark_ratio=0.4
        )

        batch = batch.to(device)
        attn_mask_v = attn_mask_v.to(device)
        attn_mask_g = attn_mask_g.to(device) if attn_mask_g is not None else None
        marked_matrix = marked_matrix.to(device)
        targets = targets.to(device)

        preds, loss, targets = model.forward(
            x=batch,
            attn_mask_v=attn_mask_v,
            attn_mask_g=attn_mask_g,
            targets=targets,
            enable_v2g=False,
            # enable_v2g=True,
        )

        print(preds.shape)
        print(targets.shape)
        print(marked_matrix.shape)
        print(f"memory: {torch.cuda.memory_allocated(device) / (2 ** 30):.2f} GB")
        break

    print(f"time: {time.perf_counter() - ctr:.2f}s")
