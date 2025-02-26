import torch

torch.set_printoptions(linewidth=10000)
data = torch.load("data_02_preprocessed_data/YJMob100K/p4_pretrain_data_discretized/test/array_50_grid.pth")

print(data.shape)
for x in data[54,12]:
    print(x)
# print(data[54,12])
