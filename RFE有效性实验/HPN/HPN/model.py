import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, APPNPConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        #print(beta)
        #beta=torch.tensor([[0.],[1.]]).to('cuda:0')
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class HANLayer(nn.Module):

    def __init__(self, num_meta_paths, hidden_size, k_layers, alpha, edge_drop, dropout):
        super(HANLayer, self).__init__()
        self.appnp_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            # self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,dropout, dropout, activation=F.elu))
            # 两层 alpha=0.03只能跑92
            self.appnp_layers.append(APPNP(k_layers=k_layers, alpha=alpha, edge_drop=edge_drop, dropout=dropout))
        self.semantic_attention = SemanticAttention(in_size=hidden_size)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.appnp_layers[i](h, g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, k_layers, alpha, edge_drop, dropout):
        super(HAN, self).__init__()
        # 投影层
        self.fc_trans=nn.Linear(in_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, hidden_size, k_layers, alpha, edge_drop, dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size, k_layers, alpha, edge_drop, dropout))
        self.predict = nn.Linear(hidden_size, out_size)

    def forward(self, g, h):
        h=self.fc_trans(h)
        #h = self.dropout(h)
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h),h

class APPNP(nn.Module):
    # 0.03 0.1 0.0
    # yelp
    def __init__(self, k_layers, alpha, edge_drop, dropout=0.6):
        super(APPNP, self).__init__()
        self.appnp = APPNPConv(k_layers, alpha, edge_drop)
        self.dropout = nn.Dropout(p=dropout)
        # self.dropout = dropout
        pass

    def forward(self, features, g):
        h = self.dropout(features)
        # h = F.dropout(features, self.dropout, training=self.training)
        h = self.appnp(g, h)
        return h