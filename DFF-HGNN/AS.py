import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Type_GAT import HeteroFeatureAggregationModule



class AS_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop, feat_drop):
        super(AS_encoder, self).__init__()
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.intra_att = nn.ModuleList([intra_att(hidden_dim, attn_drop, feat_drop) for _ in range(nei_num)])
        self.inter_att = inter_att(hidden_dim,attn_drop)

    def forward(self, nei_h, nei_index, relation_features, target_features):
        embeds = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]

            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).cuda()
            one_type_emb = F.elu(self.intra_att[i](sele_nei, nei_h[i + 1], nei_h[0], relation_features[i + 1], relation_features[0]))
            embeds.append(one_type_emb)
        z_mc = self.inter_att(embeds,target_features)
        return z_mc

class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop, feat_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.att2 = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att2.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.feat_drop = nn.Dropout(feat_drop)
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()
        self.predict_refer = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.predict_refer.weight, gain=1.414)
        self.predict_refer2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.predict_refer2.weight, gain=1.414)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.agg_relation_attribute = Agg_relation_attribute(hidden_dim, attn_drop)

    def forward(self, nei, h, h_refer, relation_features, relation_features_refer):
        emb_attribute_relation = []
# 属性特征注意力机制聚合邻居节点的属性特征
        nei_emb = F.embedding(nei, h)
        # nei_emb = self.feat_drop(self.predict_nei2(nei_emb))
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        h_refer = self.feat_drop(self.predict_refer(h_refer))
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        all_emb = F.normalize(all_emb, p=1, dim=-1)
        attn_curr = self.attn_drop(self.att)
        att_attribute = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att_attribute = self.softmax(att_attribute)
        emb_attribute = (att_attribute * nei_emb).sum(dim=1)
        emb_attribute_relation.append(emb_attribute)
# 结构特征注意力机制聚合邻居节点的结构特征
        nei_emb = F.embedding(nei, relation_features)
        # nei_emb = self.feat_drop(self.predict_nei(nei_emb))
        relation_features_refer = torch.unsqueeze(relation_features_refer, 1)
        relation_features_refer = relation_features_refer.expand_as(nei_emb)
        relation_features_refer = self.feat_drop(self.predict_refer2(relation_features_refer))
        all_emb = torch.cat([relation_features_refer, nei_emb], dim=-1)
        all_emb = F.normalize(all_emb, p=1, dim=-1)
        attn_curr2 = self.attn_drop(self.att2)
        att_relation = self.leakyrelu(all_emb.matmul(attn_curr2.t()))
        att_relation = self.softmax(att_relation)
        emb_relation = (att_relation * nei_emb).sum(dim=1)
        emb_attribute_relation.append(emb_relation)
# 属性特征和结构特征融合机制
        emb = self.agg_relation_attribute(emb_attribute_relation)
        return emb

class Agg_relation_attribute(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Agg_relation_attribute, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("每种类型的邻居 在类内聚合过程中 属性特征和关系特征的注意力系数  ", beta.data.cpu().numpy())  # 输出属性特征和关系特征的各自的注意力系数
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.heterofeatureaggregationmodule = HeteroFeatureAggregationModule(hidden_dim, attn_drop)
    def forward(self,embeds,target_features):
        z_mc = self.heterofeatureaggregationmodule(embeds,target_features)
        return z_mc