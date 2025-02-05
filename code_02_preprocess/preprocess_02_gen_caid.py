import pandas as pd
import os
import rich.progress as richprogress


def p2_gen_caid(
        from_csv: str,
        to_csv: str,
        uid_start_from_zero: bool = False,
        day_start_from_zero: bool = False,
        time_start_from_zero: bool = False,
        xy_start_from_zero: bool = True,
        time_interval: int| None = None,
):
    df = pd.read_csv(from_csv, dtype=int)

    if uid_start_from_zero:
        uid_dict = {u: index for index, u in enumerate(df.uid.unique())}
        df["uid"] = df["uid"].apply(lambda u: uid_dict.get(u))

    if day_start_from_zero:
        day_start = df.at[0, "d"]
        df["d"] = df["d"] - day_start

    if time_start_from_zero:
        assert time_interval is not None
        time_start = df.at[0, "t"]
        df["t"] = df["t"].apply(lambda t: (t - time_start) // time_interval)

    if xy_start_from_zero:
        df["x"] = df["x"] - 1
        df["y"] = df["y"] - 1

    # df["caid"] = df.apply(lambda row: f"{row.x:03d}_{row.y:03d}", axis=1)
    # df["caid_idx"] = df.apply(lambda row: row.x * 200 + row.y, axis=1)
    #
    # df["caid_100"] = df.apply(lambda row: f"{row.x // 2:03d}_{row.y // 2:03d}", axis=1)
    # df["caid_idx_100"] = df.apply(lambda row: row.x // 2 * 100 + row.y // 2, axis=1)
    #
    # df["caid_50"] = df.apply(lambda row: f"{row.x // 4:03d}_{row.y // 4:03d}", axis=1)
    # df["caid_idx_50"] = df.apply(lambda row: row.x // 4 * 50 + row.y // 4, axis=1)
    #
    # df["caid_25"] = df.apply(lambda row: f"{row.x // 8:03d}_{row.y // 8:03d}", axis=1)
    # df["caid_idx_25"] = df.apply(lambda row: row.x // 8 * 25 + row.y // 8, axis=1)

    df["caid"] = df.apply(lambda row: f"x{row.x:03d}y{row.y:03d}", axis=1)
    df["caid_idx"] = df.apply(lambda row: row.x * 200 + row.y, axis=1)

    df["caid_100"] = df.apply(lambda row: f"x{row.x // 2:03d}y{row.y // 2:03d}", axis=1)
    df["caid_idx_100"] = df.apply(lambda row: row.x // 2 * 100 + row.y // 2, axis=1)

    df["caid_50"] = df.apply(lambda row: f"x{row.x // 4:03d}y{row.y // 4:03d}", axis=1)
    df["caid_idx_50"] = df.apply(lambda row: row.x // 4 * 50 + row.y // 4, axis=1)

    df["caid_25"] = df.apply(lambda row: f"x{row.x // 8:03d}y{row.y // 8:03d}", axis=1)
    df["caid_idx_25"] = df.apply(lambda row: row.x // 8 * 25 + row.y // 8, axis=1)

    df.to_csv(to_csv, index=False)


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v2")

    # from_dir = f"data_02_preprocessed_data/YJMob100K/p1_filtered/12days_7_22"
    from_dir = f"data_02_preprocessed_data/YJMob100K/p1_filtered/30days_7_22"
    to_dir = from_dir.replace("/p1_filtered/", "/p2_caid_added/")

    os.makedirs(to_dir, exist_ok=True)

    progress = richprogress.track(os.listdir(from_dir), description=f"[+] generating caid from dir {from_dir}")
    for file_name in progress:
        if file_name.endswith(".csv"):
            p2_gen_caid(
                from_csv=f"{from_dir}/{file_name}",
                to_csv=f"{to_dir}/{file_name}",
                uid_start_from_zero=True,
                day_start_from_zero=True,
                time_start_from_zero=True,
                time_interval=2
            )
            # break
