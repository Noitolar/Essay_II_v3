import pandas as pd
import numpy as np
import torch
import os
import rich.progress as richprogress


def preprocess_03_pretrain_data_generate(
        from_csv: str | pd.DataFrame,
        to_pth_dir: str,
        reference_csv: str,
):
    to_pth_dir = to_pth_dir.replace("p3_pretrain_data", f"p3_pretrain_data/{os.path.basename(reference_csv)[:-4]}")
    os.makedirs(to_pth_dir, exist_ok=True)
    df = from_csv if isinstance(from_csv, pd.DataFrame) else pd.read_csv(from_csv, usecols=["d", "t", "x", "y"])
    df["x"] = df["x"] - 1
    df["y"] = df["y"] - 1

    ref_df = pd.read_csv(reference_csv)
    ref_coordintes = [50 * (x // 4) + y // 4 for bsid_50, x, y in ref_df.itertuples(index=False, name=None)]

    bsid_set_50 = set(ref_coordintes)
    num_nodes_50 = len(bsid_set_50)

    node_map_50 = {node: index for index, node in enumerate(bsid_set_50)}
    num_days = df["d"].nunique()

    expected_array_50 = np.zeros(shape=(num_days, 24, num_nodes_50))
    expected_array_50_grid = np.zeros(shape=(num_days, 24, 50, 50))

    for d, t, x, y in richprogress.track(df.itertuples(index=False, name=None), total=len(df), description="[=] making arrays"):
        x_50 = x // 4
        y_50 = y // 4
        t = t // 2  # 48 ticks -> 24 ticks

        try:
            coordinate_50 = node_map_50[50 * x_50 + y_50]
            expected_array_50[d][t][coordinate_50] += 1
            expected_array_50_grid[d][t][x_50][y_50] += 1
        except KeyError:
            continue

    src_name = os.path.basename(from_csv)[:-4].upper()
    torch.save(torch.from_numpy(expected_array_50), f"{to_pth_dir}/{src_name}_ARRAY_50.pth")
    torch.save(torch.from_numpy(expected_array_50_grid), f"{to_pth_dir}/{src_name}_GRIDS_50.pth")


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    torch.set_printoptions(linewidth=1000)

    # preprocess_03_pretrain_data_generate(
    #     from_csv="data_01_dataset/YJMob100K/yjmob100k-dataset-test.csv",
    #     to_pth_dir="data_02_preprocessed_data/YJMob100K/p3_pretrain_data/test",
    #     reference_csv="data_02_preprocessed_data/YJMob100K/p2_tensor/40days_20records/span_1/day_11_to_50_num80_coordinates_50.csv"
    # )

    for src_level in [1000, 2000, 5000, 10000, 20000, 50000, 125000]:
        preprocess_03_pretrain_data_generate(
            from_csv=f"data_01_dataset/YJMob100K/YJMOB_{src_level}.csv",
            to_pth_dir="data_02_preprocessed_data/YJMob100K/p3_pretrain_data",
            reference_csv="data_02_preprocessed_data/YJMob100K/p2_tensor/REFERENCE_DURATION_40_DAY_14_53_UID_80_SPAN_01.csv"
        )
