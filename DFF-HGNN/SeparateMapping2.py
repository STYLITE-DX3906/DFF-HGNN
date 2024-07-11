# 使用信息熵作为衡量标准
import torch
import torch.nn as nn

class SeparateMappingModule(nn.Module):
    def __init__(self, input_dim, output_dim,feat_drop):
        super(SeparateMappingModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.informative_mapping = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh()
        )

        self.redundant_mapping = nn.Sequential(
            nn.Linear(input_dim - output_dim, output_dim),
            nn.Tanh()
        )



    def forward(self, input_features):

        entropy_values, sorted_indices = calculate_entropy(input_features)

        informative_dimensions = input_features[:, sorted_indices[:self.output_dim]]
        informative_mapped = self.informative_mapping(informative_dimensions)

        redundant_dimensions = input_features[:, sorted_indices[self.output_dim:]]
        redundant_mapped = self.redundant_mapping(redundant_dimensions)

        output_features = informative_mapped + redundant_mapped

        return output_features



def calculate_entropy(input_features):
    # Calculate entropy for each column
    entropy_values = torch.tensor([calculate_column_entropy(col) for col in input_features.T])

    # Sort indices based on entropy in descending order
    sorted_indices = torch.argsort(entropy_values, descending=True)

    return entropy_values, sorted_indices

def calculate_column_entropy(column):
    # Assume column is a 1D tensor
    probabilities = column.softmax(dim=0)
    log_probabilities = torch.log2(probabilities)
    entropy = -torch.sum(probabilities * log_probabilities)

    return entropy

# def calculate_column_entropy(column):
#     # 确保 column 是整数类型
#     column = column.long()
#
#     # 获取当前列中的最大值作为 minlength 的参考
#     max_value = torch.max(column).item() + 1
#
#     # 统计每个值的频数
#     value_counts = torch.bincount(column, minlength=max_value)
#
#     # 计算每个值出现的概率
#     probabilities = value_counts / value_counts.sum()
#
#     # 避免概率为0的情况导致log运算出错
#     probabilities[probabilities == 0] = 1e-9
#
#     # 计算基于真实频数的概率分布的信息熵
#     log_probabilities = torch.log2(probabilities)
#     entropy = -torch.sum(probabilities * log_probabilities)
#
#     return entropy

# def calculate_column_entropy(column):
#     # 确保 column 是整数类型（如果已经是整数则忽略这一步）
#     column = column.long()
#
#     # 统计每个离散值的频数
#     value_counts = torch.bincount(column)
#
#     # 计算非零频率出现的概率
#     nonzero_probabilities = value_counts[value_counts.nonzero(as_tuple=True)]
#     probabilities = nonzero_probabilities / nonzero_probabilities.sum()
#
#     # 避免概率为0的情况导致log运算出错
#     probabilities[probabilities == 0] = 1e-9
#
#     # 计算基于真实频数的概率分布的信息熵
#     log_probabilities = torch.log2(probabilities)
#     entropy = -torch.sum(probabilities * log_probabilities)
#
#     return entropy

# def calculate_column_entropy(column, num_bins=10):
#     # 计算直方图并归一化以得到概率分布
#     hist = torch.histc(column, bins=num_bins, min=column.min(), max=column.max())
#     probabilities = hist / torch.sum(hist)
#
#     # 避免对零概率取对数，使用小数避免除零错误
#     probabilities = probabilities[probabilities > 0]
#     log_probabilities = torch.log2(probabilities)
#
#     # 计算信息熵
#     entropy = -torch.sum(probabilities * log_probabilities)
#     return entropy