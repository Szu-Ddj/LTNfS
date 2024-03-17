import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        # RGCN卷积层
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations=num_relations)

    def forward(self, x, batched_edge_index, batched_edge_type):
        # 通过两个RGCN层传递数据
        x = self.conv1(x, batched_edge_index, batched_edge_type)
        x = torch.relu(x)
        x = self.conv2(x, batched_edge_index, batched_edge_type)
        return x

# 示例参数
in_channels = 768  # 输入特征维度
out_channels = 768 # 输出特征维度
num_relations = 3  # 关系类型数量
batch_size = 2     # 批处理大小

# 创建模型实例
model = RGCN(in_channels, out_channels, num_relations)

# 示例输入数据
num_nodes = 7
x = torch.randn(batch_size, num_nodes, in_channels)  # 节点特征
batched_edge_index = torch.tensor([[[0, 1, 1, 2], [1, 0, 2, 1]], [[3, 4, 4, 5], [4, 3, 5, 4]]], dtype=torch.long)  # 边索引
batched_edge_type = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 0]], dtype=torch.long)  # 边类型

# 前向传播
print(batched_edge_index.size(), batched_edge_type.size())
out = model(x, batched_edge_index, batched_edge_type)
print(out, out.size())
# from torch.utils.data import DataLoader
# import torch

# # 假设您已经有了一个数据集
# dataset = ... # 您的数据集，每个元素包含x, edge_index, edge_type

# # 使用torch.utils.data.DataLoader
# loader = DataLoader(dataset, batch_size=2, shuffle=True)

# for data in loader:
#     x_batch = torch.cat([d['x'] for d in data], dim=0)
#     edge_index_batch = torch.cat([d['edge_index'] for d in data], dim=1)
#     edge_type_batch = torch.cat([d['edge_type'] for d in data], dim=0)

#     # 创建一个batch向量来标识每个节点的原始图索引
#     batch = torch.cat([torch.full_like(d['x'][:, 0], i) for i, d in enumerate(data)])

#     # 更新edge_index_batch以反映合并后的节点索引
#     # 这里需要一个额外的步骤来调整edge_index_batch

#     # 将处理后的数据输入模型
#     output = model(x_batch, edge_index_batch, edge_type_batch, batch)
#     # 接下来进行模型训练或评估等