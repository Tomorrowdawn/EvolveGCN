#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####构建示例用作测试#####
import networkx as nx
import dgl
import torch
import numpy as np

# 创建一个空的无向图
G = nx.Graph()

# 添加节点
num_nodes = 15
#G.add_nodes_from(range(num_nodes))
nodes_with_id = [(i, {'node_id': 2*i}) for i in range(num_nodes)]
G.add_nodes_from(nodes_with_id)

# 添加边并赋予边权重
edges = [
    (0, 1, {'weight': 0.5}),
    (0, 2, {'weight': 0.8}),
    (1, 3, {'weight': 0.3}),
    (2, 3, {'weight': 0.9}),
    (3, 4, {'weight': 0.7}),
    (4, 5, {'weight': 0.6}),
    (4, 6, {'weight': 0.4}),
    (5, 7, {'weight': 0.2}),
    (5, 8, {'weight': 0.5}),
    (6, 8, {'weight': 0.1}),
    (7, 9, {'weight': 0.3}),
    (8, 9, {'weight': 0.6}),
    (9, 10, {'weight': 0.4}),
    (9, 11, {'weight': 0.7}),
    (10, 12, {'weight': 0.9}),
    (11, 12, {'weight': 0.8}),
    (12, 13, {'weight': 0.5}),
    (12, 14, {'weight': 0.3}),
]

G.add_edges_from(edges)

# 将NetworkX图转换为DGLGraph
dgl_G = dgl.from_networkx(G)

# 添加边权重特征
weights = torch.tensor([edata['weight'] for u, v, edata in G.edges(data=True) for _ in range(2)])
dgl_G.edata['weight'] = weights
#node_ids = torch.arange(num_nodes)*2  # 假设节点ID从0开始递增
#dgl_G.ndata['node_id'] = node_ids


# 输出DGL图的一些基本信息
print("DGL图的节点数量：", dgl_G.number_of_nodes())
print("DGL图的边数量：", dgl_G.number_of_edges())
print("DGL图的边权重：", dgl_G.edata['weight'])
#print("DGL图的节点ID：", dgl_G.ndata['node_id'])


# In[10]:


#####代码主体#####
import dgl
import networkx as nx
import torch

id_name = 'node_id'#节点id的名字未固定，可以在这里改
weight_name = 'weight'#边权重的名字，这里可以改

class RelaAnaly(object):
    def __init__(self,G):#G为DGL图
        self.G = G
        self.nodes_num = G.number_of_nodes()
        try:
            self.nx_G = G.to_networkx(node_attrs=[id_name], edge_attrs=[weight_name])
        except:
            self.nx_G = G.to_networkx(edge_attrs=[weight_name])
        self.nx_G_simple = nx.Graph(self.nx_G)
        
    def ChangeOutput(self, dict0):#可以将原字典（key值为节点序号）变为新字典（key值为id_name）
        ids = np.array(self.G.ndata[id_name])
        dict1 = {}
        for i in range(len(ids)):
            dict1[ids[i]] = dict0[i]
        return dict1
        
    def InDegree(self, node_id):#节点的入度,根据id_name
        node_index = (dgl_G.ndata[id_name] == node_id_to_find).nonzero().item()
        return self.G.in_degrees(node_index)
    
    def OutDegree(self, node_id):#节点的出度
        node_index = (dgl_G.ndata[id_name] == node_id_to_find).nonzero().item()
        return self.G.out_degrees(node_index)
    
    def InDegree_(self, node_index):#节点的入度,根据节点默认id
        return self.G.in_degrees(node_index)
    
    def OutDegree_(self, node_index):#节点的出度
        return self.G.out_degrees(node_index)
    
    def CenterDegree(self):#节点地位：度中心性 http://staff.ustc.edu.cn/~tongxu/socomp/slides/4.pdf
        return nx.degree_centrality(self.nx_G_simple)
    
    def CenterCloseness(self):#节点地位：紧密中心性
        return nx.closeness_centrality(self.nx_G_simple)
    
    def CenterBetweenness(self):#节点地位&弱连接(越高可能越大)：介数中心性
        #sorted(xxxx.items(), key=lambda x: x[1], reverse=True)可直接排序
        return nx.betweenness_centrality(self.nx_G_simple)
    
    def EffectiveSize(self):#结构洞判定依据：有效大小 
        #当某个节点的联系人相互连接，它的自我中心网络（以某个节点为中心的一级网络）就具有冗余性。节点的自我网络的有效大小就是它的关系中的非冗余部分的度量
        return nx.effective_size(self.nx_G_simple)
    
    def Constraint(self):#结构洞判定依据：约束
        #这一指标衡量某一节点被其社交网络所约束的程度。一个节点附近较高的约束意味着较高的网络密度和较低的结构洞数量
        return nx.constraint(self.nx_G_simple)
    
    def PageRank(self):#PageRank算法 http://staff.ustc.edu.cn/~tongxu/socomp/slides/4.pdf
        #可排序后作为结构洞判断标准
        return nx.pagerank(self.nx_G_simple)
 
    def get_holes(self, a = 1.0, b = 0.8, c = 0.75, d = 0.2):
        #结构洞获取参考,abcd为参数
        #返回的排在前面的节点指定为结构洞
        dict1 = nx.betweenness_centrality(self.nx_G_simple)
        m1 = max(dict1.values())+0.001
        dict2 = nx.effective_size(self.nx_G_simple)
        m2 = max(dict2.values())+0.001
        dict3 = nx.constraint(self.nx_G_simple)
        m3 = max(dict3.values())+0.001
        dict4 = nx.clustering(self.nx_G_simple)
        m4 = max(dict4.values())+0.001
        thedict = {}
        for i in range(self.nodes_num):
            thedict[i] = a*dict1[i]/m1 + b*dict2[i]/m2 - c*dict3[i]/m3 - d*dict4[i]/m4
        return sorted(thedict.items(), key=lambda x: x[1], reverse=True)

