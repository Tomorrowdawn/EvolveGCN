from model.ReprModule import ReprModule, WeightedGAT
import torch.nn as nn
import torch
from dgl import DGLGraph
import torch
from torch.nn.functional import scaled_dot_product_attention, embedding
import warnings
def get_embeddings(embeddings, *indices):
    assert indices[0].dim() == 2
    # 在 embeddings 后面拼接一个全零的 embedding
    zero_embedding = torch.zeros(1, embeddings.shape[1], dtype=embeddings.dtype, device=embeddings.device)
    embeddings_with_zero = torch.cat([embeddings, zero_embedding], dim=0)

    # 使用索引操作获取对应的 embeddings
    result = []
    for indice in indices:
        result.append(embeddings_with_zero[indices])
    
    return result

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v = nn.Parameter(torch.rand(input_dim))
    def forward(self, query, key):
        q = self.q_proj(query)
        k = self.k_proj(key)
        output = scaled_dot_product_attention(q, k, self.v)
        return output

class ProposalPredictor(nn.Module):
    def __init__(self, in_feats, embedding_dim = 64):
        raise NotImplementedError("Deprecated.")
        self.repr = ReprModule(in_feats, embedding_dim)
        self.pooling = AttentionPooling(embedding_dim)
        self.cls = nn.Sequential(
            nn.Linear(embedding, 3),
            nn.Softmax(3)
        )
    def reset_weights(self):
        pass
    def forward(self, time_step, g, proposal):
        """
        time_step: 当前时间步
        
        g:图包含ndata['feat']和edata['weight'](E, 1)
        注意: edata['weight']应当是调用了EdgeWeightNorm¶的结果.
        
        proposal: 一个字典，包含'id','sponsors'和'cosponsors'。
        分别代表提案id(一个[B, 1] tensor)，发起者（一个[B, 1] tensor），协助发起者(一个[B, 4] tensor. 因为我们发现95%分位点就是4个协助发起者)
        如果不足4个协力, 则用-1填充。
        返回一个[B, N, 3]张量，表示每个节点yes/no/none的概率。
        
        """
        repr = self.repr(g)
        sponsors = proposal['sponsors']
        cosponsors = proposal['cosponsors']
        # 得到发起者的embedding
        sponsor_H, cosponsor_H = get_embeddings(repr, sponsors, cosponsors)
        H = torch.cat([sponsor_H, cosponsor_H], dim=0)
        attention_H = self.pooling(repr, H)
        return self.cls(attention_H)
    

class GATPredictor(nn.Module):
    def __init__(self, num_nodes, node_embedding_dim = None,
                 embedding_dim = 64, num_heads = 8, num_layers = 3):
        self.node_embeds = None
        if node_embedding_dim is None:
            in_feats = node_embedding_dim
            self.node_embeds = nn.Embedding(num_nodes, node_embedding_dim)          
        self.repr = WeightedGAT(in_feats, embedding_dim, num_heads, num_layers)
        self.pooling = AttentionPooling(embedding_dim)
        self.cls = nn.Sequential(
            nn.Linear(embedding_dim, 3),
            nn.Softmax(3)
        )
    def reset_weights(self):
        pass
    def forward(self, time_step, g, proposal):
        if self.node_embeds is None:
            inputs = g.ndata['feat']
        else:
            inputs = self.node_embeds
        edge_weights = g.edata['weight']
        repr = self.repr(g, inputs, edge_weights)##[N, H]
        sponsors = proposal['sponsors']
        cosponsors = proposal['cosponsors']
        # 得到发起者的embedding
        sponsor_H, cosponsor_H = get_embeddings(repr, sponsors, cosponsors)
        H = torch.cat([sponsor_H, cosponsor_H], dim=1)##[B, 5, H]
        repr = repr.unsqueeze(0).expand(H.shape[0], -1, H.shape[-1])
        attention_H = self.pooling(repr, H)##[B, N, H]
        return self.cls(attention_H)##[B, N, 3]