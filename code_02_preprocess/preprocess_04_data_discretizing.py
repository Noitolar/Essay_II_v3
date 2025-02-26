import torch
import torch.nn.functional as nnfunc
import os
import typing as tp


def preprocess_04_data_discretizing(
        from_pth: str | torch.Tensor,
        to_pth: str | None = None,
        num_bins: int = 100,
        scaler_type: tp.Literal["minmax", "norm"] = "norm",
        day_span: int = 1,
        return_tensors: bool = True,
):
    if to_pth is None:
        to_pth = from_pth.replace("p3_pretrain_data", "p4_pretrain_data_discretized")
    os.makedirs(os.path.dirname(to_pth), exist_ok=True)

    data_tensor = from_pth if isinstance(from_pth, torch.Tensor) else torch.load(from_pth)

    # data_tensor = torch.tensor([
    #     [1, 2, 6, 17],
    #     [2, 3, 1, 1],
    #     [3, 1, 2, 3]
    # ]).float()
    #
    # xxx = nnfunc.normalize(data_tensor)
    # yyy = nn.LayerNorm(data_tensor.shape[-1])(data_tensor)
    #
    # print(xxx)
    # print(yyy)
    # exit()

    # 相互错开1天，然后在t维度上合并
    num_days = data_tensor.shape[0]
    if day_span > 1:
        tmp_list = []
        index_start = 0
        index_end = num_days - day_span + 1
        while index_start < day_span:
            tmp_list.append(data_tensor[index_start:index_end])
            index_start += 1
            index_end += 1
        data_tensor = torch.concat(tmp_list, dim=1)

    if scaler_type == "minmax":
        min_value = torch.min(data_tensor)
        max_value = torch.max(data_tensor)
        data_tensor = (data_tensor - min_value) / (max_value - min_value)
    elif scaler_type == "norm":
        data_tensor = nnfunc.normalize(data_tensor)

    # 0表示mask，1~bin表示实际数值
    data_tensor = 1 + (data_tensor * (num_bins - 1)).long()
    # print(data_tensor.shape)
    # print(data_tensor[0][0])

    # data_tensor = nnfunc.one_hot(data_tensor, num_bins)
    # print(data_tensor.shape)

    torch.save(data_tensor, to_pth)

    if return_tensors:
        return data_tensor


if __name__ == "__main__":
    os.chdir(r"C:\Users\MTX\Documents\Code\Python3\Essay_II_v3")
    torch.set_printoptions(linewidth=1000)

    for day_span in (1, 2, 3, 4):
        from_dir = "data_02_preprocessed_data/YJMob100K/p3_pretrain_data/REFERENCE_DURATION_40_DAY_14_53_UID_80_SPAN_01/"
        for filename in os.listdir(from_dir):

            if "GRIDS" in filename:
                continue

            preprocess_04_data_discretizing(
                from_pth=f"{from_dir}/{filename}",
                to_pth=from_dir.replace("p3_pretrain_data", "p4_pretrain_data_discretized") + filename.replace(".pth", f"_SPAN_{day_span:02d}.pth"),
                scaler_type="norm",
                day_span=day_span,
            )