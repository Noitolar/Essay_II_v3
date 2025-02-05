import pandas as pd
import typing as tp
import multiprocessing as mp
import functools
import rich.progress as richprogress
import os


def p1_quality_filter(
        from_csv: str | pd.DataFrame,
        to_csv: str,
        valid_d_range: tuple = (0, 31, 1),
        valid_t_range: tuple = (12, 43, 2),
        min_records_threshold: int = 12,
        num_workers: int = 8,
):
    """
    每天
    6:00, 7:00, ..., 21:00
    一共16个时间点
    缺失一定数量则拒绝该用户的所有数据
    """

    if isinstance(from_csv, str):
        df = pd.read_csv(from_csv, dtype=int)
    elif isinstance(from_csv, pd.DataFrame):
        df = from_csv
    else:
        raise TypeError("[!] <from_csv> must be str or pd.DataFrame")

    df = df[df["d"].isin(range(*valid_d_range))]
    df = df[df["t"].isin(range(*valid_t_range))]

    workers = mp.Pool(num_workers)
    task = functools.partial(
        p1_task,
        num_days=len(range(*valid_d_range)),
        valid_t_range=valid_t_range,
        min_records_threshold=min_records_threshold,
        debug=False
    )
    uid_and_groups = df.groupby("uid")
    progress = richprogress.track(uid_and_groups, description="[+] p1_quality_filter")

    results = list(workers.imap(task, progress))

    workers.close()
    workers.join()

    results = pd.concat(results).sort_values(by=["uid", "d", "t"]).reset_index(drop=True)

    tmp = results["uid"].unique()
    # print(f"有效UID共{len(tmp)}个：{tmp}")
    print(f"有效UID共{len(tmp)}个")
    to_csv = to_csv.replace(".csv", f"_num{len(tmp)}.csv")

    results.to_csv(to_csv, index=False)


def p1_task(
        uid_and_group: tp.Tuple[int, pd.DataFrame],
        num_days: int,
        valid_t_range: tuple,
        min_records_threshold: int,
        debug: bool
)->pd.DataFrame|None:
    uid, group = uid_and_group
    filled_groups = list()

    group_by_d = group.groupby("d")
    if len(group_by_d) < num_days:
        return None

    for d, sub_group in group.groupby("d"):
        sub_group.reset_index(inplace=True, drop=True)
        length = len(sub_group)
        max_records_threshold = len(range(*valid_t_range))

        if length < min_records_threshold:
            if debug:
                print(f"[!] BAD UID --- uid:{uid}, d:{d}, len:{len(sub_group)}")
            return None
        elif length > max_records_threshold:
            raise NotImplementedError
        elif len(sub_group) == max_records_threshold:
            filled_groups.append(sub_group)
            continue
        elif min_records_threshold <= length < max_records_threshold:
            tmp = pd.DataFrame(
                [
                    {
                        "uid": uid,
                        "d": d,
                        "t": t
                    } for t in range(*valid_t_range)
                ]
            )
            sub_group = pd.merge(sub_group, tmp, how="right", on=["uid", "d", "t"])
            sub_group.ffill(axis=0, inplace=True)
            sub_group.bfill(axis=0, inplace=True)
            filled_groups.append(sub_group)
            continue
        else:
            raise NotImplementedError

    return pd.concat(filled_groups).reset_index(drop=True)


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v2")

    # g = pd.read_csv("test.csv")
    # p1_task(
    #     ("46", g),
    #     (14, 45, 2),
    #     12,
    #     False
    # )

    main_df = pd.read_csv("data_01_dataset/YJMob100K/yjmob100k-dataset-merged.csv", dtype=int)

    valid_d_range_start = 0
    num_days = 30
    valid_d_range_end = num_days + valid_d_range_start

    valid_t_range = (14, 45, 2)  # 7:00 ~ 22:00, 16 records
    min_records_threshold = 12  # none less than 12 records

    target_dir = f"data_02_preprocessed_data/YJMob100K/p1_filtered/{num_days}days_7_22"
    os.makedirs(target_dir, exist_ok=True)

    while valid_d_range_end <= 74:
        p1_quality_filter(
            # from_csv="data_01_dataset/YJMob100K/yjmob100k-dataset-merged.csv",
            from_csv=main_df,
            to_csv=f"{target_dir}/day{valid_d_range_start:02d}_to_day{valid_d_range_end - 1:02d}.csv",
            valid_d_range=(valid_d_range_start, valid_d_range_end, 1),
            valid_t_range=valid_t_range,
            min_records_threshold=min_records_threshold,
            num_workers=mp.cpu_count() - 1,
        )

        # early stop
        if valid_d_range_start >= 20:
            break

        valid_d_range_start += 1
        valid_d_range_end += 1
