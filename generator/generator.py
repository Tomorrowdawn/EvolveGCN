import pandas as pd
import numpy as np
import dgl
import torch
from dgl.nn.pytorch.conv import EdgeWeightNorm
from tqdm import tqdm
import time
from copy import deepcopy

class GraphGenerator():
    def __init__(self, proposals:pd.DataFrame, 
                 members:pd.DataFrame,
                 votes:pd.DataFrame):
        
        self.index = 0
        self.threshold = 0.5
        
        # 处理 proposals 部分
        self.proposals = proposals
        self.prepare_proposals()

        # 处理 members 部分
        self.members = members
        self.prepare_members()
        
        # 处理 votes 部分
        self.votes = votes
        self.prepare_votes()

        # meeting就按 102-116的编号 从0-14
        # bill 先按meeting_id 排序 再按字典序 全部转换成0开始的数字下标
        # member 直接按 id 字典序 全部转换成0开始的数字下标
        self.meeting2index = {}
        self.bill2index = {}
        self.member2index = {}
        self.meeting_num = 0
        self.bill_num = 0
        self.member_num = 0
        self.meeting_bill_member_tensor_voteResult = None
        self.meeting_bill_tensor_maxVoteCount = None

        self.prepare_meeting_bill_member_tensor()

        self.subgraph = None
        self.bills = None
        self.create_subgraph()
        self.modify_subgraph()
        self.create_bills()
        self.training = True

        pass   
    
    @staticmethod
    def variant_name():
        return "base"  
    
    def prepare_proposals(self):
        # 检查 proposals 是否包含 'bill_id' 列
        if 'bill_id' not in self.proposals.columns:
            print("警告：数据中不存在 'bill_id' 列。")
            print("列名：", self.proposals.columns)
        # 提取 meeting_id
        self.proposals['meeting_id'] = self.proposals['bill_id'].str.split('-').str.get(1) # 假设 meeting_id 总是在 bill_id 的第二部分
        pass
    
    def prepare_members(self):
        pass

    def prepare_votes(self):
        # 检查 votes 是否包含 'date' 列
        if 'date' in self.votes.columns:
            # 检查 'date' 列是否已转换为 datetime 类型
            if not pd.api.types.is_datetime64_any_dtype(self.votes['date']):
                self.votes['date'] = pd.to_datetime(self.votes['date'], utc=True)
        else:
            print("警告：数据中不存在 'date' 列。")
            print("列名：", self.votes.columns)
        # 检查 votes 是否按 'date' 列排序
        if not self.votes.empty:
            self.votes.sort_values(by='date', inplace=True) # 按 'date' 列排序
        # 检查 votes 是否包含 'meeting_id' 列
        if 'meeting_id' not in self.votes.columns and 'bill_name' in self.votes.columns:
            # 将 'meeting_id' 从 'bill_name' 中提取
            self.votes['meeting_id'] = self.votes['bill_name'].str.split('-').str.get(1) # 假设 meeting_id 总是在 bill_name 的第二部分
        pass

    def calculate_bill_vote_count(votes, meeting_id, bill_name, member_id):
        # 根据 meeting_id, bill_name, member_id 筛选 votes
        filtered_votes = [vote for vote in votes if vote["meeting_id"] == meeting_id and vote["bill_name"] == bill_name and vote["id"] == member_id]
        # 计算特定议案的 votes 个数
        bill_vote_count = len(filtered_votes)
        return bill_vote_count

    def prepare_meeting_bill_member_tensor(self):
        # 统计 meeting 数量
        self.meeting_num = self.votes['meeting_id'].nunique()
        # 统计 bill 数量
        # self.bill_num = self.proposals['bill_id'].nunique()
        self.bill_num = self.votes['bill_name'].nunique()

        # 初始化 meeting2index
        self.meeting2index = dict(zip(self.votes['meeting_id'].unique(), range(self.meeting_num)))
        # 初始化 bill2index, 先clone votes, 在 votes 副本 中找 bill, 按 meeting_id 排序, 再按bill_name字典序, 从0开始编号
        votes_clone = self.votes.copy()
        votes_clone.sort_values(by=['meeting_id', 'bill_name'], inplace=True)
        self.bill2index = dict(zip(votes_clone['bill_name'].unique(), range(self.bill_num)))
        # 初始化 member2index, 直接按 id 字典序, 从0开始编号
        # members 是一个 DataFrame，包含 'meeting' , 'subclub', 'members' 三列, 'members' 列是一个列表
        members_list = self.members['members'].tolist()
        all_members = [member for sublist in members_list for member in sublist] # 将所有成员放在一个列表中
        all_members = list(set(all_members)) # 去重
        all_members.sort() # 排序
        # 统计 member 数量
        self.member_num = len(all_members)
        self.member2index = dict(zip(all_members, range(self.member_num)))

        # 初始化 meeting_bill_member_tensor_voteResult
        self.meeting_bill_member_tensor_voteResult = np.full((self.meeting_num, self.bill_num, self.member_num), -np.inf) # 用于存储每个议案的投票结果,初始化为-inf
        # 初始化 meeting_bill_tensor_maxVoteCount
        self.meeting_bill_tensor_maxVoteCount = np.zeros((self.meeting_num, self.bill_num)) # 用于存储每次会议每个议案的最大投票次数,初始化为0

        # print("member2index: ", self.member2index)

        # members_list = self.members['id'].values
        # members_list = sorted([member for sublist in members_list for member in sublist])
        # self.member2index = dict(zip(members_list, range(self.member_num)))

        # 创建一个长为 member_num 的数组, 用于存储每个成员的投票次数
        member_vote_count = np.zeros(self.member_num) 

        # 填充 meeting_bill_member_tensor
        def process_row(row: pd.Series) -> None:
            meeting_idx = self.meeting2index[row[-1]] # meeting_id
            bill_idx = self.bill2index[row[0]] # bill_name
            # 如果 member 不在 member2index 中，跳过
            #if member not in self.member2index:
            #    return
            member_idx = self.member2index[row[3]] # id
            vote = row[-2] # vote
            vote2val = {
                'Y': 1,
                'N': -1,
                'NV': 0
            }
            self.meeting_bill_member_tensor_voteResult[meeting_idx, bill_idx, member_idx] = vote2val[vote]
            # 更新 meeting_bill_tensor_maxVoteCount
            member_vote_count[member_idx] += 1
            self.meeting_bill_tensor_maxVoteCount[meeting_idx, bill_idx] = np.max(member_vote_count)
        self.votes.loc[self.votes['id'].isin(self.member2index)].apply(
            process_row, axis=1, raw=True)

    def similarity_measure(self, u, v, meeting_index):
        m_copy = self.meeting_bill_member_tensor_voteResult
        mask = m_copy == -np.inf
        m_copy[mask] = 0
        # 假设议题A要被投票3次，甲3次赞成得到+3，乙3次赞成得到+3，类似的议题B要被投票1次，甲3次赞成得到+1，乙3次赞成得到+1, 那么这2个议题上甲和乙每次赞成的相似度应该是一样的
        # 通过 self.meeting_bill_tensor_maxVoteCount 平衡不同议题
        # 计算时防止分母为零
        max_vote_count = self.meeting_bill_tensor_maxVoteCount[meeting_index, :] # 每个议案的最大投票次数
        normalized_max_vote_count = np.where(max_vote_count == 0, 1, max_vote_count) # 防止分母为零

        # 计算标准化相似度分数
        # print("u: ", u)
        # print("v: ", v)
        # print("meeting_index: ", meeting_index)
        # normalized_scores = m_copy[meeting_index, :, u] * m_copy[meeting_index, :, v] / normalized_max_vote_count # 标准化相似度分数，可以考虑修改为其他标准化方法
        # 计算两个成员对于所有议案的标准化投票总和的相似度
        normalized_scores = u * v / normalized_max_vote_count
        similarity_score = np.sum(normalized_scores) # 计算相似度分数

        return similarity_score
        
        
    def create_subgraph(self):
        subgraphs = []  # 存储每次会议的子图

        m_copy = self.meeting_bill_member_tensor_voteResult
        #mask = m_copy == -np.inf
        #m_copy[mask] = 0

        # 对每次会议创建子图
        # for meeting_index in range(self.meeting_num):
        for meeting_index in tqdm(range(self.meeting_num), desc="Generating subgraphs"):
            start_time = time.time()

            # 提取每次会议的投票tensor
            votes_slice = deepcopy(m_copy[meeting_index, :, :])
            votes_slice[votes_slice==-np.inf] = 0
            
                        
            # 计算相似度矩阵（其尺寸应该是 num_members x num_members）
            # similarity_matrix = squareform(pdist(votes_slice.T, lambda u, v: self.similarity_measure(u, v, meeting_index)))
            
            max_vote_count = self.meeting_bill_tensor_maxVoteCount[meeting_index, :] # 每个议案的最大投票次数
            normalized_max_vote_count = np.where(max_vote_count == 0, 1, max_vote_count) # 防止分母为零
            # normalized_max_vote_count 从(8849,) 重塑为 (8849,1)
            normalized_max_vote_count = normalized_max_vote_count[:, np.newaxis]

            # 这样就可以保持行对行的除法操作，每个议题对每个成员进行归一化,  标准化议案贡献，使得每个议案对于相似度贡献相同 (members x bills)
            normalized_votes = votes_slice / normalized_max_vote_count

            # 利用矩阵乘法计算成员间的相似度 (members x members)
            # np.dot 对二维数组执行矩阵乘法，对于一维数组执行内积
            similarity_matrix = np.dot(normalized_votes, normalized_votes.T)
            
            # 根据相似度矩阵创建图
            g = dgl.DGLGraph()
            
            # 使用triu_indices函数获取上三角矩阵中的索引
            src_list, dst_list = np.triu_indices(self.member_num, k=1)  # k=1表示不包括对角线
            src_list = src_list.astype(np.int64)
            dst_list = dst_list.astype(np.int64)
            # print("src_list.shape before: ", src_list.shape)
            # print("dst_list.shape before: ", dst_list.shape)

            # 从这些索引中得到所有的边的权重，并过滤掉无穷大的权重
            edge_weights = similarity_matrix[src_list, dst_list]
            # print("edge_weights.shape: ", edge_weights.shape)
            finite_edges = ~np.isinf(edge_weights)
            # print("finite_edges.shape: ", finite_edges.shape)
            
            # 只保留有限权重的边
            src_list = src_list[finite_edges]
            dst_list = dst_list[finite_edges]
            edge_weights = edge_weights[finite_edges]
            # print("src_list.shape: ", src_list.shape)
            # print("dst_list.shape: ", dst_list.shape)
            # print("edge_weights.shape: ", edge_weights.shape)

            # 如果有边可以添加，那么转换权重到适当的类型并添加这些边
            if len(src_list) > 0:
                # 将权重从NumPy数组转换为PyTorch张量
                edge_weights_tensor = torch.from_numpy(edge_weights).float()

                # 一次性添加所有的边和它们的权重
                g.add_edges(src_list, dst_list, {'weight': edge_weights_tensor})
            
            # # 将NumPy数组转换为PyTorch张量
            votes_slice = m_copy[meeting_index, :, :]
            vote_data = torch.from_numpy(votes_slice.T).float()

            # # 输出有多少个节点
            # print("g.num_nodes(): ", g.num_nodes())
            # # 输出有多少个边
            # print("g.num_edges(): ", g.num_edges())
            # # 输出vote_data 的形状
            # print("vote_data.shape: ", vote_data.shape)
            
            # # 把每个成员的投票数据设置为节点的'data'特征
            g.ndata['vote'] = vote_data

            # 为了把每个成员的投票数据设置为节点的'data'特征， 遍历已有节点并设置'data'特征
            
            subgraphs.append(g)
        
        self.subgraph = subgraphs
        pass

    def modify_subgraph(self):
        updated_subgraphs = []

        # 去除权重小于阈值的边
        for g in self.subgraph:
            # 获取边的权重
            edge_weights = g.edata['weight']
            # 选择权重大于阈值的边
            mask = edge_weights > self.threshold
            # 保留权重大于阈值的边的子图
            # sub_g = g.edge_subgraph(mask, preserve_nodes=True)
            sub_g = g.edge_subgraph(mask, relabel_nodes=False)
            updated_subgraphs.append(sub_g)

        # 遍历每个有更新的子图，并为每个子图加上自循环边
        for idx, g in enumerate(updated_subgraphs):
            # 为每个子图加上自循环边
            g_with_self_loops = dgl.add_self_loop(g)
            # 更新列表中的图
            updated_subgraphs[idx] = g_with_self_loops

        # 使用EdgeWeightNorm对边权重进行归一化处理
        for g in updated_subgraphs:
            if g.num_edges() > 0:  # 确保图中有边
                edge_weights = g.edata['weight']
                norm = EdgeWeightNorm(norm='right')  # 选择归一化的方向，'right'表示按出度归一化
                normalized_weights = norm(g, edge_weights)
                g.edata['weight'] = normalized_weights

        # 更新self.subgraph
        self.subgraph = updated_subgraphs
    
    def create_bills(self):
        # 遍历 self.proposals ，创建 bills (是一个字典，包含 'sponsors' 和 'cosponsors' 两个键，每个键对应一个张量，sponsors和cosponsors的内容都是member index，形状是［bill_num,1］,［bill_num，4］)
        # cosponsors如果不足四个的话补-1
        # self.proposals 已经 通过函数 prepare_proposals 进行了排序等处理
        bills = {'sponsors': [], 'cosponsors': []}
        for idx, row in self.proposals.iterrows():
            # 提取 sponsors
            sponsor = row['sponsor']
            sponsor_index = self.member2index[sponsor]
            sponsor_tensor = torch.tensor([sponsor_index], dtype=torch.long)
            bills['sponsors'].append(sponsor_tensor)
            # 提取 cosponsors
            cosponsors = row['cosponsors']
            cosponsor_indices = [self.member2index[cosponsor] for cosponsor in cosponsors]
            # 如果 cosponsors 的数量不足 4，用 -1 填充
            while len(cosponsor_indices) < 4:
                cosponsor_indices.append(-1)
            # 如果 cosponsors 的数量超过 4，只保留前 4 个
            cosponsor_indices = cosponsor_indices[:4]
            cosponsor_tensor = torch.tensor(cosponsor_indices, dtype=torch.long)
            bills['cosponsors'].append(cosponsor_tensor)
        # 将 sponsors 和 cosponsors 转换为张量
        bills['sponsors'] = torch.stack(bills['sponsors'])
        bills['cosponsors'] = torch.stack(bills['cosponsors'])

        self.bills = bills
        
        pass
    
    def get_num_nodes(self):
        return self.member_num
    def train(self):
        self.training = True
    def eval(self):
        self.training = False
    def to(self, device):
        self.device = device
        for i in range(len(self.subgraph)):
            self.subgraph[i] = self.subgraph[i].to(device)
        self.bills['sponsors'] = self.bills['sponsors'].to(device)
        self.bills['cosponsors'] = self.bills['cosponsors'].to(device)
    def __iter__(self):
        self.index = 0
        return self
        
    def __next__(self):
        if self.training:
            offset = 0
            bound = 12
        else:
            offset = 12
            bound = 15
        if self.index + offset < bound:
            result = (self.subgraph[self.index + offset], self.bills)
            self.index += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.subgraph)
        
    def __getitem__(self, idx_or_slice):
        # 判断idx_or_slice的类型并返回相应的子图或子图列表
        if isinstance(idx_or_slice, slice):
            # 如果是切片对象，则返回一个范围内的子图列表
            return (self.subgraph[idx_or_slice], self.bills)
        elif isinstance(idx_or_slice, int):
            # 如果idx_or_slice是整数，我们返回那个索引处的子图
            if idx_or_slice < 0 or idx_or_slice >= len(self):
                raise IndexError("Index out of range")
            
            return (self.subgraph[idx_or_slice], self.bills)
        else:
            raise TypeError("Invalid argument type.")