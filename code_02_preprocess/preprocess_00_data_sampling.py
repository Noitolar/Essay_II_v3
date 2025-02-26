import pandas as pd
import os


def multi_level_sampling(
        from_csv: pd.DataFrame | str,
        to_dir: str
):
    df = from_csv if isinstance(from_csv, pd.DataFrame) else pd.read_csv(from_csv)
    num_uids = 125000
    print(f"num_uids = {num_uids}")
    splits = dict(
        YJMOB_50000=50000,
        YJMOB_20000=20000,
        YJMOB_10000=10000,
        YJMOB_5000=5000,
        YJMOB_2000=2000,
        YJMOB_1000=1000,
        # YJMOB_500=500,
        # YJMOB_200=200,
        # YJMOB_100=100,
    )

    for name, value in splits.items():
        df = df[df.uid < value].reset_index(drop=True)
        df.to_csv(f"{to_dir}/{name}.csv", index=False)


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    multi_level_sampling(
        from_csv="data_01_dataset/YJMob100K/YJMOB_125000.csv",
        to_dir="data_01_dataset/YJMob100K"
    )
