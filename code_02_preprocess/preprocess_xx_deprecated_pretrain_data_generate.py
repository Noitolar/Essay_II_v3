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
    to_pth_dir = to_pth_dir.replace("p3_pretrain_data", f"p3_pretrain_data/ref_{os.path.basename(reference_csv)[:-4]}")
    os.makedirs(to_pth_dir, exist_ok=True)
    df = from_csv if isinstance(from_csv, pd.DataFrame) else pd.read_csv(from_csv)
    ref_df = pd.read_csv(reference_csv)
    ref_coordintes = [(x, y) for bsid_50, x, y in ref_df.itertuples(index=False, name=None)]

    # bsid_set_200 = set()
    # bsid_set_100 = set()
    bsid_set_50 = set()
    for u, d, t, x, y in richprogress.track(df.itertuples(index=False, name=None), total=len(df), description="[=] adding bsid"):
        if (x, y) not in ref_coordintes:
            continue

        x_200 = x - 1
        y_200 = y - 1
        # x_100 = x_200 // 2
        # y_100 = y_200 // 2
        x_50 = x_200 // 4
        y_50 = y_200 // 4

        # bsid_set_200.add(200 * x_200 + y_200)
        # bsid_set_100.add(100 * x_100 + y_100)
        bsid_set_50.add(50 * x_50 + y_50)
        # bsid_set_200.add(f"x{x_200}y{y_200}")
        # bsid_set_100.add(f"x{x_100}y{y_100}")
        # bsid_set_50.add(f"x{x_50}y{y_50}")

    # node_map_200 = {node: index for index, node in enumerate(bsid_set_200)}
    # node_map_100 = {node: index for index, node in enumerate(bsid_set_100)}
    node_map_50 = {node: index for index, node in enumerate(bsid_set_50)}

    # num_nodes_200 = len(bsid_set_200)
    # num_nodes_100 = len(bsid_set_100)
    num_nodes_50 = len(bsid_set_50)
    num_days = df["d"].nunique()

    # expected_array_200 = np.zeros(shape=(num_days, 24, num_nodes_200))
    # expected_array_100 = np.zeros(shape=(num_days, 24, num_nodes_100))
    expected_array_50 = np.zeros(shape=(num_days, 24, num_nodes_50))

    # expected_array_200_grid = np.zeros(shape=(num_days, 24, 200, 200))
    # expected_array_100_grid = np.zeros(shape=(num_days, 24, 100, 100))
    expected_array_50_grid = np.zeros(shape=(num_days, 24, 50, 50))

    for u, d, t, x, y in richprogress.track(df.itertuples(index=False, name=None), total=len(df), description="[=] making arrays"):
        if (x, y) not in ref_coordintes:
            continue

        x_200 = x - 1
        y_200 = y - 1
        # x_100 = x_200 // 2
        # y_100 = y_200 // 2
        x_50 = x_200 // 4
        y_50 = y_200 // 4

        t = t // 2 # 48 ticks -> 24 ticks

        # coordinate_200 = node_map_200[200 * x_200 + y_200]
        # coordinate_100 = node_map_100[100 * x_100 + y_100]
        coordinate_50 = node_map_50[50 * x_50 + y_50]

        # expected_array_200[d][t][coordinate_200] += 1
        # expected_array_100[d][t][coordinate_100] += 1
        expected_array_50[d][t][coordinate_50] += 1

        # expected_array_200_grid[d][t][x_200][y_200] += 1
        # expected_array_100_grid[d][t][x_100][y_100] += 1
        expected_array_50_grid[d][t][x_50][y_50] += 1

    # torch.save(torch.from_numpy(expected_array_200), f"{to_pth_dir}/array_200.pth")
    # torch.save(torch.from_numpy(expected_array_100), f"{to_pth_dir}/array_100.pth")
    torch.save(torch.from_numpy(expected_array_50), f"{to_pth_dir}/array_50.pth")

    # torch.save(torch.from_numpy(expected_array_200_grid), f"{to_pth_dir}/array_200_grid.pth")
    # torch.save(torch.from_numpy(expected_array_100_grid), f"{to_pth_dir}/array_100_grid.pth")
    torch.save(torch.from_numpy(expected_array_50_grid), f"{to_pth_dir}/array_50_grid.pth")


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    torch.set_printoptions(linewidth=1000)

    # preprocess_03_pretrain_data_generate(
    #     from_csv="data_01_dataset/YJMob100K/yjmob100k-dataset-test.csv",
    #     to_pth_dir="data_02_preprocessed_data/YJMob100K/p3_pretrain_data/test",
    #     reference_csv="data_02_preprocessed_data/YJMob100K/p2_tensor/40days_20records/span_1/day_11_to_50_num80_coordinates_50.csv"
    # )

    preprocess_03_pretrain_data_generate(
        from_csv="data_01_dataset/YJMob100K/yjmob100k-dataset-merged.csv",
        to_pth_dir="data_02_preprocessed_data/YJMob100K/p3_pretrain_data/merged",
        reference_csv="data_02_preprocessed_data/YJMob100K/p2_tensor/40days_20records/span_1/day_11_to_50_num80_coordinates_50.csv"
    )
