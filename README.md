# 2024春社会计算实验
paper link: [GAT](https://arxiv.org/abs/1710.10903)  

## Dependency:
* dgl
* pandas
* numpy
* torch

## Run
```bash
python train_model.py
```

## Report

### preprocess

根据观察, 我们将votes数据进行排序, 发现其和meetings id呈时间相关顺序, 故而根据每场meeting划分子图. 总共15个图, 前12个用于训练, 最后三个用于测试. 预测目标为投票倾向(N,V,NV)

### Predictor

简单来说, 图上每个节点先经过embedding层创建embedding, 随后经过GATConv层汇聚邻居信息, 生成其表征representation. 对于每个预测对(member, bill), 将bill相关的sponsors和cosponsors取出, 与member的repr拼接后送入attention池化层, 得到池化后的隐层向量, 最后送入MLP进行分类.

### Result

0.98 on testset.
