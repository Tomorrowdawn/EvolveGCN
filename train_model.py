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
        for g in gen:
            optimizer.zero_grad()
            g:dgl.DGLGraph
            node_mask = g.ndata['node_mask'].to(device)
            labels = g.ndata['label'].to(device)
            prediction = model(g, g.ndata['feat'].to(device), g.edata['weight'].to(device))
            loss = criterion(prediction[node_mask], labels[node_mask])
            loss.backward()
            optimizer.step()
            if report_hook is not None:
                report_hook(epoch, loss.item())
        if epoch > 0 and epoch % eval_epoches == 0 and eval_hook is not None:
            eval_hook(model, gen)
    return model