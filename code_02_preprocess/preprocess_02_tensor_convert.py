import pandas as pd
import numpy as np
import torch
import os


def preprocess_02_tensor_convert(
        from_csv: str | pd.DataFrame,
        to_pth: str = None,
        node_col_id: str = "bsid_50",
        user_col_id: str = "uid",
        day_span: int = 1,
        return_tensor: bool = False,
):
    if to_pth is None:
        to_pth = from_csv.replace("p1_filtered", "p2_tensor").replace(".csv", ".pth")

    os.makedirs(os.path.dirname(to_pth), exist_ok=True)
    df = from_csv if isinstance(from_csv, pd.DataFrame) else pd.read_csv(from_csv)

    users = df[user_col_id].unique()
    days = df["d"].unique()
    ticks = df["t"].unique()
    nodes = df[node_col_id].unique()

    num_users = len(users)
    num_days = len(days)
    num_ticks = len(ticks)
    num_nodes = len(nodes)

    print(f"shape: (u={num_users},d={num_days},t={num_ticks},v={num_nodes})")

    node_map = {node: index for index, node in enumerate(nodes)}
    user_map = {user: index for index, user in enumerate(users)}
    day_map = {day: index for index, day in enumerate(days)}

    df[node_col_id] = df[node_col_id].map(node_map)
    df[user_col_id] = df[user_col_id].map(user_map)
    df["d"] = df["d"].map(day_map)

    expected_array = np.zeros(shape=(
        num_users,
        num_days,
        num_ticks,
        num_nodes,
    ), dtype=int)

    for uid, d, t, x, y, bsid_200, bsid_100, bsid_50 in df.itertuples(index=False, name=None):
        expected_array[uid, d, t, bsid_50] = 1

    result_tensor = torch.from_numpy(expected_array)
    result_tensor = torch.einsum("udtv->dtvu", result_tensor)
    # result_tensor = result_tensor[:, :, :, torch.randperm(result_tensor.shape[-1])]
    # result_tensor = result_tensor[:, :, :, :100]

    # 相互错开1天，然后在t维度上合并
    if day_span > 1:
        tmp_list = []
        index_start = 0
        index_end = num_days - day_span + 1
        while index_start < day_span:
            tmp_list.append(result_tensor[index_start:index_end])
            index_start += 1
            index_end += 1
        result_tensor = torch.concat(tmp_list, dim=1)

    # fake user: 在其他用户状态全0时为1
    # mask user: 默认为0，在生成mask的时候，如果对应的(d, t, v)坐标被选中，则将整个状态向量替换为([1],[0],[0,0,,,0])
    fake_user_record = (1 - torch.sum(result_tensor, dim=-1).bool().int()).unsqueeze(dim=-1)
    mask_user_record = torch.zeros_like(fake_user_record)
    # print(fake_user_record.shape)
    # print(result_tensor.shape)

    attached_tensor = torch.concat([mask_user_record, fake_user_record, result_tensor], dim=-1)
    # print(attached_tensor.shape)

    target_tensor = torch.einsum("dtvu->dtuv", result_tensor)
    target_tensor = target_tensor.argmax(dim=-1)
    dtu_target_tensor = 1 + target_tensor  # 留一个位置给mask

    torch.save(attached_tensor, to_pth)
    torch.save(target_tensor, to_pth.replace("p2_tensor/", "p2_tensor/TARGETS_"))
    torch.save(dtu_target_tensor, to_pth.replace("p2_tensor/", "p2_tensor/DTU_TARGETS_"))

    # 保存一下节点信息
    coordinates = df[["bsid_50", "x", "y"]].drop_duplicates(subset="bsid_50").reset_index(drop=True)
    coordinates.to_csv(to_pth.replace("p2_tensor/", "p2_tensor/REFERENCE_").replace(".pth", ".csv"), index=False)

    # coordinates = df[["bsid_100", "x", "y"]].drop_duplicates(subset="bsid_100").reset_index(drop=True)
    # coordinates.to_csv(to_pth.replace(".pth", "_coordinates_100.csv"), index=False)
    #
    # coordinates = df[["bsid_200", "x", "y"]].drop_duplicates(subset="bsid_200").reset_index(drop=True)
    # coordinates.to_csv(to_pth.replace(".pth", "_coordinates_200.csv"), index=False)

    if return_tensor:
        return attached_tensor, target_tensor


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    torch.set_printoptions(linewidth=1000)

    filename = "DURATION_40_DAY_14_53_UID_80"
    for span in (1,):
        preprocess_02_tensor_convert(
            from_csv=f"data_02_preprocessed_data/YJMob100K/p1_filtered/{filename}.csv",
            to_pth=f"data_02_preprocessed_data/YJMob100K/p2_tensor/{filename}_SPAN_{span:02d}.pth",
            day_span=span,
        )

    filename = "DURATION_20_DAY_07_26_UID_20"
    # for span in (1, 2, 3, 4):
    for span in (1,):
        preprocess_02_tensor_convert(
            from_csv=f"data_02_preprocessed_data/YJMob100K/p1_filtered_strict/{filename}.csv",
            to_pth=f"data_02_preprocessed_data/YJMob100K/p2_tensor/{filename}_SPAN_{span:02d}.pth",
            day_span=span,
        )
