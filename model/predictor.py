from model.ReprModule import ReprModule
import torch.nn as nn
import torch
from dgl import DGLGraph
import torch
from torch.nn.functional import scaled_dot_product_attention, embedding

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
        self.repr = ReprModule(in_feats, embedding_dim)
        self.pooling = AttentionPooling(embedding_dim)
        self.cls = nn.Sequential(
            nn.Linear(embedding, 3),
            nn.Softmax(3)
        )
        
    def forward(self, g_list:list[DGLGraph], proposal, sampled_members = None):
        """
        
        g_list:一列图，每个图包含ndata['feat']和edata['weight'](E, 1)
        注意: edata['weight']应当是调用了EdgeWeightNorm¶的结果.
        proposal: 一个字典，包含'id','sponsors'和'cosponsors'。
        分别代表提案id(一个[B, 1] tensor)，发起者（一个[B, 1] tensor），协助发起者(一个[B, 4] tensor. 因为我们发现95%分位点就是4个协助发起者)
        如果不足4个协力, 则用-1填充。
        返回一个[B, N, 3]张量，表示每个节点yes/no/none的概率。
        
        sampled_members: 如果全计算的情况下计算量过大, 可以只计算sampled_members中的节点的预测结果.
            TODO: 还未实现.
        """
        repr = self.repr(g_list)
        sponsors = proposal['sponsors']
        cosponsors = proposal['cosponsors']
        # 得到发起者的embedding
        sponsor_H, cosponsor_H = get_embeddings(repr, sponsors, cosponsors)
        H = torch.cat([sponsor_H, cosponsor_H], dim=0)
        attention_H = self.pooling(repr, H)
        return self.cls(attention_H)