import torch
import torch.nn as nn
import torch.nn.functional as F
from AS import AS_encoder
# from SeparateMapping import SeparateMappingModule
from SeparateMapping2 import SeparateMappingModule


from AS import Agg_relation_attribute


class RAS_HGN(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, relation_dim_list, sample_rate, nei_num, attn_drop, feat_drop, outsize):
        super(RAS_HGN, self).__init__()

        self.hidden_dim = hidden_dim

# 特征映射增强方法
        self.fc_list1 = nn.ModuleList([SeparateMappingModule(feats_dim, hidden_dim, feat_drop) for feats_dim in feats_dim_list])

        self.fc_list2 = nn.ModuleList([SeparateMappingModule(relation_dim, hidden_dim, feat_drop) for relation_dim in relation_dim_list])

# dropout
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

# 定义AS实例
        self.AS = AS_encoder(hidden_dim, sample_rate, nei_num, attn_drop, feat_drop)

        self.predict2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.predict = nn.Linear(hidden_dim, outsize)

        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)


    def forward(self, feats, nei_index, relation_features):

# 节点特征转换 + dropoout
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list1[i](feats[i]))))

        relation_features_all = []
        for i in range(len(relation_features)):
            relation_features_all.append(F.elu(self.feat_drop(self.fc_list2[i](relation_features[i]))))

# 进行AS操作

        target_features = self.alpha * h_all[0] + (1 - self.alpha) * relation_features_all[0]
        print("目标节点更新前 自身的融合特征中属性特征所占权重：", self.alpha.data.cpu().numpy())
        print("目标节点更新前 自身的融合特征中关系特征所占权重：", (1-self.alpha).data.cpu().numpy())

        z_AS = self.AS(h_all, nei_index, relation_features_all, target_features)

# 融合自身特征和邻居特征

        h = torch.cat([target_features, z_AS], dim=1)

        h = self.feat_drop(self.predict2(h))

        return self.feat_drop(self.predict(h)), h
