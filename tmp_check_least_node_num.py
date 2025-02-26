import os
import pandas as pd


if __name__ == "__main__":
    target_dir = "data_02_preprocessed_data/YJMob100K/p1_filtered"
    for filename in os.listdir(target_dir):
        df = pd.read_csv(os.path.join(target_dir, filename))
        print(f"{filename}: {df.bsid_50.nunique()}")