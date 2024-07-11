import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=2):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        #print(beta)
        #beta=torch.tensor([[0.],[1.0]]).to('cuda:0')
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class HANLayer(nn.Module):

    def __init__(self, num_meta_paths, in_size, out_size, dropout):
        super(HANLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for i in range(num_meta_paths):
            self.gcn_layers.append(GraphConv(in_size, out_size,activation=F.elu, weight=True))
        self.semantic_attention = SemanticAttention(in_size=out_size)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h,weight):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gcn_layers[i](g, h,edge_weight=weight[i]).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.fc_trans=nn.Linear(in_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, hidden_size, hidden_size, dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size ,hidden_size, dropout))
        self.predict = nn.Linear(hidden_size, out_size)

    def forward(self, g, h,eweight):
        h=self.fc_trans(h)
        h = self.dropout(h)
        for gnn in self.layers:
            h = self.dropout(h)
            h = gnn(g, h,eweight)
        return self.predict(h),h