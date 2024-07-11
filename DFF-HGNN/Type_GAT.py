import torch
import torch.nn as nn
import torch.nn.functional as F



class HeteroFeatureAggregationModule(nn.Module):
    def __init__(self, in_dim, attn_drop):
        super(HeteroFeatureAggregationModule, self).__init__()
        # Learnable parameters
        self.att_local_weights = nn.Parameter(torch.empty(size=(1, 2*in_dim)), requires_grad=True)
        self.att_global_weights1 = nn.Parameter(torch.empty(size=(1, 2*in_dim)), requires_grad=True)
        self.att_global_weights2 = nn.Parameter(torch.empty(size=(1, 1, 1)), requires_grad=True)
        nn.init.xavier_normal_(self.att_local_weights.data, gain=1.414)
        nn.init.xavier_normal_(self.att_global_weights1.data, gain=1.414)
        nn.init.xavier_normal_(self.att_global_weights2.data, gain=1.414)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, neighbor_features_list, target_features):
        # Concatenate neighbor features along the last dimension
        emb = torch.stack(neighbor_features_list, dim=0)
        h_refer = target_features.unsqueeze(0).expand_as(emb)
        all_emb = torch.cat([h_refer, emb], dim=-1)
        all_emb = F.normalize(all_emb, p=2, dim=-1)
        # Local attention calculation
        attn_curr_local = self.attn_drop(self.att_local_weights)
        att_local = self.leakyrelu(all_emb.matmul(attn_curr_local.t()))
        att_local = self.softmax(att_local)
        # Global attention calculation
        attn_global1 = self.attn_drop(self.att_global_weights1)
        attn_global2 = self.attn_drop(self.att_global_weights2)
        att_global2 = attn_global2.expand_as(all_emb)
        att_global = self.leakyrelu(att_global2.matmul(attn_global1.t()))
        att_global = self.softmax(att_global)
        # Combine local and global attention weights
        final_attentions = self.alpha * att_local + (1-self.alpha) * att_global

        # Weighted sum of neighbor features using the final attention weights
        weighted_neighbor_features = final_attentions * emb
        aggregated_features = weighted_neighbor_features.sum(dim=0)

        return aggregated_features