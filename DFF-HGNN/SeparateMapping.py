# 使用方差作为信息量的衡量标准 复刻RFR思路
import torch
import torch.nn as nn
from sklearn.feature_selection import mutual_info_regression
import numpy as np

class SeparateMappingModule(nn.Module):
    def __init__(self, input_dim, output_dim,feat_drop):
        super(SeparateMappingModule, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear mapping for informative dimensions
        self.informative_mapping = nn.Linear(output_dim, output_dim)
        # self.informative_mapping = nn.Sequential(
        #     nn.Linear(output_dim, output_dim),
        #     # nn.ReLU()
        #     # nn.LeakyReLU()
        #     # nn.Tanh()
        #     nn.Sigmoid()
        #     # nn.ELU()
        #     # nn.GELU()
        #     # nn.Hardswish()
        #
        # )

        # Non-linear mapping for redundant dimensions
        self.redundant_mapping = nn.Sequential(
            nn.Linear(input_dim - output_dim, output_dim),
            # nn.ReLU()
            # nn.LeakyReLU()
            # nn.Tanh()
            # nn.Sigmoid()
            nn.ELU()
            # nn.GELU()
            # nn.Hardswish()

        )
        # self.redundant_mapping = nn.Linear(input_dim - output_dim, output_dim)


        # self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # self.predict = nn.Linear(2 * output_dim, output_dim)

        # if feat_drop > 0:
        #     self.feat_drop = nn.Dropout(feat_drop)
        # else:
        #     self.feat_drop = lambda x: x


    def forward(self, input_features):

        # # Calculate mean and variance for each dimension
        # # mean_values = torch.mean(input_features, dim=0)
        variance_values, sorted_indices = torch.sort(torch.var(input_features, dim=0), descending=True)

        # # Select top d dimensions and perform linear mapping
        informative_dimensions = input_features[:, sorted_indices[:self.output_dim]]
        informative_mapped = self.informative_mapping(informative_dimensions)

        # # Select top d dimensions and perform linear mapping 保持原特征矩阵中的维度次序不变
        # informative_dimensions = input_features[:, sorted_indices[:self.output_dim]]
        # original_order_indices = torch.argsort(sorted_indices[:self.output_dim])
        # informative_dimensions = informative_dimensions[:, original_order_indices]
        # informative_mapped = self.informative_mapping(informative_dimensions)

        # Select remaining (m-d) dimensions and perform non-linear mapping
        redundant_dimensions = input_features[:, sorted_indices[self.output_dim:]]
        redundant_mapped = self.redundant_mapping(redundant_dimensions)

        # Hadamard product of the two mapped matrices
        # output_features = informative_mapped + redundant_mapped
        output_features = informative_mapped * redundant_mapped
        # output_features =  torch.cat([informative_mapped, redundant_mapped], dim=1)
        # output_features = self.feat_drop(self.predict(output_features))



        return output_features


