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
    is_valid = votes != float('-inf')

    # 沿着第一个维度求和，得到每一列中 -inf 的数量
    valid_count = is_valid.sum(dim=0)

    valid_indices = torch.where(valid_count > 0)[0]

    return valid_indices


def train_model(gen:GraphGenerator, node_embedding_size = 64,
                epoches:int = 100, batch_size = 32,
                lr:float=1e-3, device='cuda',
                report_hook = None, eval_hook = None, eval_epoches = 10):
    N = gen.get_num_nodes()
    print("num nodes: ", N)
    model = GATPredictor(N, node_embedding_size, num_heads=4)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    gen.train()
    for epoch in range(epoches):
        for i, (g, bills) in enumerate(gen):
            ##g: 所有节点都在, 但某些边被mask掉
            #g.to(device)
            votes:torch.Tensor= g.ndata['vote']##[N, B]矩阵, 填充为1,0,-1,-inf
            valid_mask = find_valid_columns(votes)
            votes = votes[:, valid_mask].to(device)
            #print("valid votes shape: ", votes.shape)
            sponsors = bills['sponsors'][valid_mask].to(device)
            cosponsors = bills['cosponsors'][valid_mask].to(device)
            for b_idx, batch in enumerate(get_random_batches(votes, sponsors, cosponsors,
                                            batch_size=batch_size)):
                
                votes, sponsors, cosponsors = batch
                #print("batch idx: ", b_idx)
                node_mask = (votes > -10)
                if node_mask.sum() < 1:
                    #print("skip")
                    continue
                labels = votes[node_mask] + 1##0, 1, 2
                #print(" batch labels :", labels.shape)
                labels = labels.long()
                optimizer.zero_grad()
                g:dgl.DGLGraph
                proposal = {
                    'id':None,
                    'sponsors':sponsors, ##[B, 1]
                    'cosponsors':cosponsors ##[B, 4]
                }
                logits:torch.Tensor = model(i, g, proposal)##[B, N, 3]
                logits = logits.transpose(0, 1)##[N, B, 3], fit to node_mask.
                loss = criterion(logits[node_mask], labels)
                loss.backward()
                optimizer.step()
                if report_hook is not None:
                    report_hook(epoch, i, b_idx, loss.item(), g,
                                proposal, logits[node_mask], labels)
        if epoch > 0 and epoch % eval_epoches == 0 and eval_hook is not None:
            gen.eval()
            eval_hook(model, gen)
            gen.train()
            model.train()
    return model

# 导入加载数据集的函数
from load_dataset import load_dataset, split_data

# 定义数据集所在的目录
dataset_dir = 'data'  # 替换为你的数据集目录

# 加载数据集
# 可选：如果split_data函数已定义完整，你可以用它来分割数据
# train_set, test_set = split_data(votes) # 假设是对votes进行分割

# 创建GraphGenerator实例
import os
import pickle
if os.path.exists(os.path.join('./data', 'gen.pkl')):
    with open(os.path.join('./data', 'gen.pkl'), 'rb') as f:
        gen = pickle.load(f)
else:
    cosponsors, members, votes = load_dataset(dataset_dir)
    gen = GraphGenerator(proposals=cosponsors, members=members, votes=votes)
    with open(os.path.join('./data', 'gen.pkl'), 'wb') as f:
        pickle.dump(gen, f)

# 实例化模型
# model = GATPredictor(gen.get_num_nodes(), node_embedding_size=64, num_heads=3)

# 设定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
gen.to(device)
# 进行训练的测试函数
def test_train_model():
    try:
        # 调用train_model函数
        reports = []
        def report_hook(epoch, timestep, b_idx, loss, g, proposal, logits, labels):
            acc = cal_accuracy(logits, labels)
            print(f"Epoch: {epoch}, Step: {timestep}, Batch: {b_idx}, Loss: {loss}, Acc: {acc}")
            reports.append({'epoch': epoch, 'timestep': timestep, 'b_idx': b_idx, 'loss': loss, 'acc': acc})
            pass
        trained_model = train_model(gen, node_embedding_size=16,
                                    epoches=100, batch_size=64,
                                    lr=1e-3, device=device,
                                    report_hook=report_hook, eval_hook=eval_hook,
                                    eval_epoches=10)
        print("训练成功！")
        torch.save(trained_model.state_dict(), './data/gat_predictor.pt')
        torch.save(reports, './data/reports.pt')
    except Exception as e:
        print(f'训练失败: {e}')
        raise e

import torch

def cal_accuracy(logits, labels):
    """
    计算准确率的函数。
    
    参数:
    logits -- 模型的输出，形状为 [batch_size, num_classes]
    labels -- 真实的标签，形状为 [batch_size]
    
    返回:
    accuracy -- 此批数据的准确率
    """
    # 获取最可能的类别索引
    _, predicted = torch.max(logits, dim=1)
    
    # 计算正确预测的数量
    correct = (predicted == labels).sum().item()
    
    # 计算准确率
    accuracy = correct / labels.size(0)
    
    return accuracy

# 定义一个简单的评估钩子函数
def eval_hook(model, gen):
    print("评估模型...")

# 运行测试
test_train_model()