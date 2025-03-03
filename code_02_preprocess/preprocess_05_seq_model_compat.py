import torch
import pandas as pd
import datasets as ds
import collections
import random
import os
import code_01_utils.utils_03_tokenizer_loader as ut3


def tensor_to_csv(
        dtu_targets_path: str,
        cache_dir: str,
        applyed_users: int
):
    dtu_targets = torch.load(dtu_targets_path)
    days, ticks, users = dtu_targets.shape
    assert applyed_users <= users
    dtu_targets = dtu_targets[:, :, torch.randperm(users)]
    dtu_targets = dtu_targets[:, :, :applyed_users]
    dtu_targets = torch.einsum("dtu->udt", dtu_targets)

    cache_dir = f"{cache_dir}/USER_APPLYED_{applyed_users:02d}"
    os.makedirs(cache_dir, exist_ok=True)

    df_data = collections.defaultdict(list)
    for uid, seq1 in enumerate(dtu_targets):
        for d, seq2 in enumerate(seq1):
            for t, node in enumerate(seq2):
                df_data["uid"].append(f"u{uid:03d}")
                df_data["time"].append(t)
                df_data["x"].append(0)
                df_data["y"].append(0)
                df_data["bsid"].append(f"b{node:03d}")
                df_data["trjid"].append(f"u{uid:03d}d{d:02d}")

    df = pd.DataFrame.from_dict(df_data)
    df.to_csv(f"{cache_dir}/records.csv", index=False)

    vocabs = ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]"]
    nodes = sorted(df["bsid"].unique().tolist())
    with open(f"{cache_dir}/vocab.txt", "w", encoding="utf-8") as f:
        for token in vocabs + nodes:
            f.write(f"{token}\n")

    result = collections.defaultdict(list)
    for trjid, group in df.groupby("trjid"):
        # print(len(group["bsid"]))
        result["sentence"].append(" ".join(group["bsid"].tolist()))
    result_df = pd.DataFrame(result)

    huggingface_dataset = ds.Dataset.from_pandas(result_df)
    huggingface_dataset_dict = huggingface_dataset.train_test_split(train_size=0.8, test_size=0.2)

    tokenizer = ut3.load_tokenizer(f"{cache_dir}/vocab.txt")
    # mask_manipulator = nn7.MaskManipulator(t, v, u, g)

    input_ids = tokenizer(
        list(huggingface_dataset_dict["test"]["sentence"]),
        padding=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )["input_ids"]
    num_samples_test = len(huggingface_dataset_dict["test"]["sentence"])

    random.seed(888)
    mask_matrix_pred_01 = list()
    attn_mask_pred_01 = list()
    mask_matrix_pred_03 = list()
    attn_mask_pred_03 = list()
    for _ in range(num_samples_test):
        mask_p1 = torch.zeros(size=(1, 2 + ticks))
        mask_p3 = torch.zeros(size=(1, 2 + ticks))
        attn_p1 = torch.zeros(size=(1, 2 + ticks))
        attn_p3 = torch.zeros(size=(1, 2 + ticks))
        target_index = 1 + int(random.uniform(0.4, 0.8) * ticks)
        mask_p1[:, target_index] = 1
        mask_p3[:, target_index:target_index + 3] = 1
        attn_p1[:, :target_index + 1] = 1
        attn_p3[:, :target_index + 3] = 1
        mask_matrix_pred_01.append(mask_p1)
        mask_matrix_pred_03.append(mask_p3)
        attn_mask_pred_01.append(attn_p1)
        attn_mask_pred_03.append(attn_p3)
    mask_matrix_pred_01 = torch.concat(mask_matrix_pred_01, dim=0).bool()
    mask_matrix_pred_03 = torch.concat(mask_matrix_pred_03, dim=0).bool()
    attn_mask_pred_01 = torch.concat(attn_mask_pred_01, dim=0).long()
    attn_mask_pred_03 = torch.concat(attn_mask_pred_03, dim=0).long()

    input_ids_pred_01 = input_ids.clone()
    input_ids_pred_03 = input_ids.clone()
    targets_pred_01 = input_ids.clone()
    targets_pred_03 = input_ids.clone()

    # print(input_ids_pred_01.shape)
    # print(mask_matrix_pred_01.shape)
    #
    # print(input_ids_pred_01[11])
    # print(mask_matrix_pred_01[11].int())
    # print(attn_mask_pred_01[11].int())
    # print(mask_matrix_pred_03[11].int())
    # print(attn_mask_pred_03[11].int())
    # exit()

    input_ids_pred_01[mask_matrix_pred_01] = 0  # [MASK] == 0
    input_ids_pred_03[mask_matrix_pred_03] = 0

    targets_pred_01[~mask_matrix_pred_01] = -100
    targets_pred_03[~mask_matrix_pred_03] = -100

    # print(input_ids_pred_01[11])
    # print(targets_pred_01[11])
    # print(mask_matrix_pred_01[11].int())
    # print(attn_mask_pred_01[11].int())
    #
    # print(input_ids_pred_03[11])
    # print(targets_pred_03[11])
    # print(mask_matrix_pred_03[11].int())
    # print(attn_mask_pred_03[11].int())
    # exit()

    data_pred_01 = dict(
        marked_input_ids=input_ids_pred_01,
        marked_matrix=mask_matrix_pred_01,
        targets=targets_pred_01,
        attn_mask=attn_mask_pred_01,
    )

    data_pred_03 = dict(
        marked_input_ids=input_ids_pred_03,
        marked_matrix=mask_matrix_pred_03,
        targets=targets_pred_03,
        attn_mask=attn_mask_pred_03,
    )

    huggingface_dataset_dict["test_pred_01"] = ds.Dataset.from_dict(data_pred_01)
    huggingface_dataset_dict["test_pred_03"] = ds.Dataset.from_dict(data_pred_03)
    huggingface_dataset_dict.save_to_disk(dataset_dict_path=cache_dir)


if __name__ == "__main__":
    import os

    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    torch.set_printoptions(linewidth=10000)
    tensor_to_csv(
        dtu_targets_path="data_02_preprocessed_data/YJMob100K/p2_tensor/DTU_TARGETS_DURATION_20_DAY_07_26_UID_20_SPAN_01.pth",
        cache_dir="data_02_preprocessed_data/YJMob100K/p5_compat",
        applyed_users=5
    )
