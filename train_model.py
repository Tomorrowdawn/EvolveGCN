from model.predictor import GATPredictor
from generator.generator import GraphGenerator
import torch
import dgl

def train_model(gen:GraphGenerator, node_embedding_size = 64,
                epoches:int = 200, lr:float=1e-3, device='cuda',
                report_hook = None, eval_hook = None, eval_epoches = 10):
    N = gen.get_num_nodes()
    model = GATPredictor(N, node_embedding_size, num_heads=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epoches):
        for i, (g, bills) in enumerate(gen):
            ##g: 所有节点都在, 但某些边被mask掉
            g = dgl.add_self_loop(g)
            votes :torch.Tensor= g.ndata['vote'].to(device)##[N, B]矩阵, 填充为1,0,-1,-inf
            optimizer.zero_grad()
            g:dgl.DGLGraph
            node_mask = (votes > -10)
            labels = votes[node_mask] + 1##0, 1, 2
            labels = labels.long()
            proposal = {
                'id':None,
                'sponsors':bills['sponsors'], ##[B, 1]
                'cosponsors':bills['cosponsors'] ##[B, 4]
            }
            prediction:torch.Tensor = model(i, g, proposal)##[B, N, 3]
            prediction = prediction.transpose(0, 1)##[N, B, 3], fit to node_mask.
            loss = criterion(prediction[node_mask], labels)
            loss.backward()
            optimizer.step()
            if report_hook is not None:
                report_hook(epoch, loss.item())
        if epoch > 0 and epoch % eval_epoches == 0 and eval_hook is not None:
            eval_hook(model, gen)
    return model