import numpy as np
from .generator import GraphGenerator

class VariantGenerator(GraphGenerator):
    @staticmethod
    def variant_name(self):
        return "variant"
    
    def similarity_measure(self, meeting_index):
        # m_copy = self.meeting_bill_member_tensor_voteResult
        # mask = m_copy == -np.inf
        # m_copy[mask] = 0
        # # 假设议题A要被投票3次，甲3次赞成得到+3，乙3次赞成得到+3，类似的议题B要被投票1次，甲3次赞成得到+1，乙3次赞成得到+1, 那么这2个议题上甲和乙每次赞成的相似度应该是一样的
        # # 通过 self.meeting_bill_tensor_maxVoteCount 平衡不同议题
        # # 计算时防止分母为零
        # max_vote_count = self.meeting_bill_tensor_maxVoteCount[meeting_index, :] # 每个议案的最大投票次数
        # normalized_max_vote_count = np.where(max_vote_count == 0, 1, max_vote_count) # 防止分母为零

        # # 计算标准化相似度分数
        # # print("u: ", u)
        # # print("v: ", v)
        # # print("meeting_index: ", meeting_index)
        # # normalized_scores = m_copy[meeting_index, :, u] * m_copy[meeting_index, :, v] / normalized_max_vote_count # 标准化相似度分数，可以考虑修改为其他标准化方法
        # # 计算两个成员对于所有议案的标准化投票总和的相似度
        # normalized_scores = u * v / normalized_max_vote_count
        # similarity_score = np.sum(normalized_scores) # 计算相似度分数

        # return similarity_score
        
        m_copy = self.meeting_bill_member_tensor_voteResult
        mask = m_copy == -np.inf
        m_copy[mask] = 0
        # # 假设议题A要被投票3次，甲3次赞成得到+3，乙3次赞成得到+3，类似的议题B要被投票1次，甲3次赞成得到+1，乙3次赞成得到+1, 那么这2个议题上甲和乙每次赞成的相似度应该是一样的

        # 提取每次会议的投票tensor
        votes_slice = m_copy[meeting_index, :, :]
        
        max_vote_count = self.meeting_bill_tensor_maxVoteCount[meeting_index, :] # 每个议案的最大投票次数
        normalized_max_vote_count = np.where(max_vote_count == 0, 1, max_vote_count) # 防止分母为零
        # normalized_max_vote_count 从(8849,) 重塑为 (8849,1)
        normalized_max_vote_count = normalized_max_vote_count[:, np.newaxis]

        # 这样就可以保持行对行的除法操作，每个议题对每个成员进行归一化,  标准化议案贡献，使得每个议案对于相似度贡献相同 (members x bills)
        normalized_votes = votes_slice / normalized_max_vote_count
        similarity_matrix = np.dot(normalized_votes, normalized_votes.T)

        return similarity_matrix