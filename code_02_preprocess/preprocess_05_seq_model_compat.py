import torch
import pandas as pd
import datasets as ds
import random
import collections

import code_03_nnmodel.nnmodel_07_mask_manipulator as nn7


def tensor_to_csv(
        dtu_targets_path: str,
        to_csv: str
):
    dtu_targets = torch.load(dtu_targets_path) - 1
    d, t, u = dtu_targets.shape
    dtu_targets = torch.einsum("dtu->udt", dtu_targets)
    # dtu_targets = dtu_targets.reshape(u, d * t)

    df_data = collections.defaultdict(list)
    for uid, seq1 in enumerate(dtu_targets):
        for d, seq2 in enumerate(seq1):
            for t, node in enumerate(seq2):
                df_data["uid"].append(f"uid_{uid}")
                df_data["t"].append(t)
                df_data["bsid"].append(f"bsid_{node:04d}")
                df_data["trjid"].append(f"uid_{uid}_d_{d}")
        df = pd.DataFrame.from_dict(df_data)
        df.to_csv(to_csv, index=False)

# def preprocess_05_seq_model_compat(
#         dtu_target_path: str,
#         to_dir: str
# ):
#     dtu_targets = torch.load(dtu_target_path) - 1
#     d, t, u = dtu_targets.shape
#     v = int(torch.max(dtu_targets).item())
#     g = 114514
#
#     data_
#
#
#     trn_size = int(d * 0.8)
#     trn_set = dtu_targets[:trn_size]
#     val_set = dtu_targets[trn_size:]
#
#     random.seed(42)
#     hist_steps = [int(t * random.uniform(0.3, 0.6)) for _ in range(d)]
#
#     marked_input_ids_list = list()
#     attn_mask_list = list()
#     targets_list = list()
#     marked_matrix_list = list()
#
#     mask_manipulator = nn7.MaskManipulator(t, v, u, g)
#
#     for item in dtu_targets:
#         item = item.unsqueeze(0)
#         mask_manipulator.mask(
#             item,
#             tar
#         )


if __name__ == "__main__":
    import os

    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    tensor_to_csv(
        dtu_targets_path="data_02_preprocessed_data/YJMob100K/p2_tensor/DTU_TARGETS_DURATION_40_DAY_14_53_UID_80_SPAN_01.pth",
        to_csv="data_02_preprocessed_data/YJMob100K/p5_compat/records.csv",
    )
