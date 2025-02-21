import pandas as pd
import numpy as np
import typing as tp
import multiprocessing as mp
import functools
import rich.progress as richprogress
import os
import random


# import warnings
# warnings.filterwarnings('error')


def p1_merge_48_to_24(
        from_csv: str | pd.DataFrame,
        to_csv: str,
        min_num_records_per_day: int = 16,
        valid_d_range: tuple = (0, 31, 1),
        num_workers: int = 8,
):
    """
    原始数据是每天0~47个时刻
    融合每2个个时刻
        如果两个时刻都有值，则随机选择
        如果只有一个值就选择这个
        如果都没有值，则利用相邻的上一个/下一个时刻进行填充
            如果上一个/下一个时刻都是空的，则为空
    """

    if isinstance(from_csv, pd.DataFrame):
        df = from_csv
    elif isinstance(from_csv, str):
        df = pd.read_csv(from_csv)
    else:
        raise TypeError("[!] from_csv must be DataFrame or str")

    df = df[df["d"].isin(range(*valid_d_range))]
    workers = mp.Pool(num_workers)
    task = functools.partial(
        p1_task,
        min_num_days=len(range(*valid_d_range)),
        min_num_records_per_day=min_num_records_per_day,
    )

    uid_and_groups = df.groupby("uid")
    progress = richprogress.track(uid_and_groups, description="[+] p1_merge_48_to_24")
    results = [result for result in workers.imap(task, progress) if result is not None]
    workers.close()
    workers.join()

    results = pd.concat(results).sort_values(by=["uid", "d", "t"]).reset_index(drop=True)
    tmp = results["uid"].unique()
    print(f"有效UID共{len(tmp)}个")
    to_csv = to_csv.replace(".csv", f"_num{len(tmp)}.csv")
    results.to_csv(to_csv, index=False)


def p1_task(
        uid_and_group: tp.Tuple[int, pd.DataFrame],
        min_num_days: int,
        min_num_records_per_day: int,
) -> pd.DataFrame | None:
    uid, group = uid_and_group
    results = list()

    # 如果该用户记录中的天数小于指定天数
    # 则丢弃该用户的记录
    group_by_d = group.groupby("d")
    if len(group_by_d) < min_num_days:
        return None

    for d, sub_group in group_by_d:
        sub_group.reset_index(inplace=True, drop=True)

        list_48 = [(None, None) for _ in range(48)]
        for _, _, t, x, y in sub_group.itertuples(index=False, name=None):
            assert 0 <= t <= 47
            list_48[t] = (x, y)

        list_24 = [(None, None) for _ in range(24)]
        for index_a, index_b in [(x, x + 1) for x in range(0, len(list_48), 2)]:
            record_a, record_b = list_48[index_a], list_48[index_b]
            if record_a[0] is not None and record_b[0] is not None:
                list_24[index_a // 2] = random.choice([record_a, record_b])
            elif record_a[0] is not None and record_b[0] is None:
                list_24[index_a // 2] = record_a
            elif record_b[0] is not None and record_a[0] is None:
                list_24[index_a // 2] = record_b
            elif record_a[0] is None and record_b[0] is None:
                tmp = (None, None)
                if index_a == 0:
                    tmp = list_48[index_b + 1]
                elif index_b == 47:
                    tmp = list_48[index_a - 1]
                else:
                    pre_record = list_48[index_a - 1]
                    post_record = list_48[index_b + 1]
                    if pre_record[0] is not None and post_record[0] is not None:
                        tmp = random.choice([pre_record, post_record])
                    elif pre_record[0] is not None and post_record[0] is None:
                        tmp = pre_record
                    elif post_record[0] is not None and pre_record[0] is None:
                        tmp = post_record
                    elif post_record[0] is None and pre_record[0] is None:
                        pass
                    else:
                        raise NotImplementedError
                list_24[index_a // 2] = tmp
            else:
                raise NotImplementedError

        # 如果当天的记录数量低于阈值
        # 抛弃该用户
        list_not_none = [x for x in list_24 if x[0] is not None]

        if len(list_not_none) < min_num_records_per_day:
            return None
        # 高于阈值则进行前后的补全
        else:
            df_today = pd.DataFrame([
                {
                    "uid": uid,
                    "d": d,
                    "t": t,
                    "x": x,
                    "y": y,
                    "bsid_200": f"x{x:03d}y{y:03d}" if x is not None else None,
                    "bsid_100": f"x{x // 2:03d}y{y // 2:03d}" if x is not None else None,
                    "bsid_50": f"x{x // 4:03d}y{y // 4:03d}" if x is not None else None,
                } for t, (x, y) in enumerate(list_24)
            ])

            # 如果用户在任何一天经过的不同位置节点(bsid_200级别)数量小于3
            # 则放弃该用户
            if df_today["bsid_200"].nunique() < 3:
                return None

            df_today.ffill(axis=0, inplace=True)
            df_today.bfill(axis=0, inplace=True)
            results.append(df_today)
            continue

    # try:
    #     xxx = pd.concat(results)
    # except FutureWarning:
    #     for v in results:
    #         print(v)
    #     exit()

    return pd.concat(results).reset_index(drop=True)


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    # main_df = pd.read_csv("data_01_dataset/YJMob100K/yjmob100k-dataset-test.csv", dtype=int)
    main_df = pd.read_csv("data_01_dataset/YJMob100K/yjmob100k-dataset-merged.csv", dtype=int)

    min_num_records = 20
    d_start = 0
    d_duration = 40
    d_end = d_start + d_duration

    target_dir = f"data_02_preprocessed_data/YJMob100K/p1_filtered/{d_duration}days_{min_num_records}records"
    os.makedirs(target_dir, exist_ok=True)

    while d_end < 75:
        p1_merge_48_to_24(
            from_csv=main_df,
            to_csv=f"{target_dir}/day_{d_start:02d}_to_{d_end - 1:02d}.csv",
            min_num_records_per_day=min_num_records,
            valid_d_range=(d_start, d_end, 1),
            num_workers=8,
            # num_workers=1,
        )

        if d_start > 20:
            break

        d_start += 1
        d_end += 1
