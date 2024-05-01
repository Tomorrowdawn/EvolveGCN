from model.predictor import GATPredictor
from generator.generator import GraphGenerator
import torch
import dgl
from tqdm import tqdm
def get_random_batches(*matrices, batch_size):
    # 获取第一个矩阵的形状
    N, B = matrices[0].shape

    # 生成一个随机排列的索引
    random_indices = torch.randperm(B)

    # 对每个矩阵应用随机索引进行重新排列
    shuffled_matrices = [matrix[:, random_indices] if i == 0 else matrix[random_indices] for i, matrix in enumerate(matrices)]

    # 计算完整批次的数量
    num_full_batches = B // batch_size

    # 遍历完整批次
    for i in range(num_full_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        batches = [matrix[:, start_index:end_index] if j == 0 else matrix[start_index:end_index] for j, matrix in enumerate(shuffled_matrices)]
        yield batches

    # 处理剩余的数据（如果有）
    if B % batch_size != 0:
        start_index = num_full_batches * batch_size
        remaining_batches = [matrix[:, start_index:] if j == 0 else matrix[start_index:] for j, matrix in enumerate(shuffled_matrices)]
        yield remaining_batches

def find_valid_columns(votes):
    """
    找出矩阵中全部不是均为 -inf 的列的下标。

    参数：
    - votes: 一个形状为 [N, B] 的矩阵，包含 0、1、-1 或 -inf。

    返回值：
    - valid_indices: 一个包含合法列的向量.
    """
    # 创建一个与 votes 形状相同的张量，用于存储每个元素是否为 -inf
    is_neg_inf = votes != float('-inf')

    # 沿着第一个维度求和，得到每一列中 -inf 的数量
    neg_inf_count = is_neg_inf.sum(dim=0)

    # 找出 -inf 数量不等于 N 的列的下标
    valid_indices = torch.where(neg_inf_count != votes.shape[0])[0]

    return valid_indices


def train_model(gen:GraphGenerator, node_embedding_size = 64,
                epoches:int = 100, batch_size = 64,
                lr:float=1e-3, device='cuda',
                report_hook = None, eval_hook = None, eval_epoches = 10):
    N = gen.get_num_nodes()
    model = GATPredictor(N, node_embedding_size, num_heads=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epoches):
        for i, (g, bills) in enumerate(gen):
            ##g: 所有节点都在, 但某些边被mask掉
            votes:torch.Tensor= g.ndata['vote']##[N, B]矩阵, 填充为1,0,-1,-inf
            valid_mask = find_valid_columns(votes)
            votes = votes[:, valid_mask].to(device)
            sponsors = bills['sponsors'][valid_mask].to(device)
            cosponsors = bills['cosponsors'][valid_mask].to(device)
            for batch in get_random_batches(votes, sponsors, cosponsors,
                                            batch_size=batch_size):
                votes, sponsors, cosponsors = batch
                node_mask = (votes > -10)
                if node_mask.sum() < 1:
                    continue
                labels = votes[node_mask] + 1##0, 1, 2
                labels = labels.long()
                optimizer.zero_grad()
                g:dgl.DGLGraph
                proposal = {
                    'id':None,
                    'sponsors':sponsors, ##[B, 1]
                    'cosponsors':cosponsors ##[B, 4]
                }
                prediction:torch.Tensor = model(i, g, proposal)##[B, N, 3]
                prediction = prediction.transpose(0, 1)##[N, B, 3], fit to node_mask.
                loss = criterion(prediction[node_mask], labels)
                loss.backward()
                optimizer.step()
                if report_hook is not None:
                    report_hook(epoch, i, loss.item(), g,
                                proposal, prediction)
        if epoch > 0 and epoch % eval_epoches == 0 and eval_hook is not None:
            eval_hook(model, gen)
            model.train()
    return model