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


## 预测算法

### 问题

给定一系列图 $G_i$, 这些图属于一个大图 $G$的子图, 有 $|G_i| = |G|, E_{G_i}\subseteq E_G$. 每个节点关联一个vote向量, 形如 $[B, ]$ ,表示对于每个bill的投票结果. $-inf$表示未投票. 每条边关联一个标量权重. 根据 $G_1, \cdots G_n$, 预测 $G_{n+1}...$的vote向量. 忽略 $-inf$项.

### backbone算法

考虑到这是一个时序图 + 节点分类问题. 有两种解决方向: 一种是使用RNN等可动态更新权重的模块来演化网络权重, 另一种是使用一种归纳式(inductive)算法. 

考虑到这些图都共享相同的节点(对应于现实中同一个人), 因此很自然的想法是对这些节点进行隐向量建模, 即, 每个节点使用一个长$\mathcal{E}$的向量进行表征. 在之前的图上训练的表征向量可以自然地迁移到下一个图上, 因为节点不变.

因此, 我们可以使用图注意力神经网络, [GAT](https://arxiv.org/abs/1710.10903), 进行学习. 

GAT使用标准的消息传递范式, 即下一层的节点表征由上一层的表征和邻居的表征变换而来:

$$
\begin{split}\begin{align}
z_i^{(l)}&=W^{(l)}h_i^{(l)},& \\
e_{ij}^{(l)}&=\text{LeakyReLU}(\vec a^{(l)^T}(z_i^{(l)}||z_j^{(l)})),&\\
\alpha_{ij}^{(l)}&=\frac{\exp(e_{ij}^{(l)})}{\sum_{k\in \mathcal{N}(i)}^{}\exp(e_{ik}^{(l)})},&\\
h_i^{(l+1)}&=\sigma\left(\sum_{j\in \mathcal{N}(i)} {\alpha^{(l)}_{ij} z^{(l)}_j }\right),&
\end{align}\end{split}
$$

可以看到, 注意力机制被应用于最后的加权上.

特别地, 在边加权的情况下, 有公式

$$
h_{i}^{l+1} = \sigma\left(\sum_{j\in\mathcal{N}(i)} w_{ij}\alpha_{ij}^lz_j^{l}\right)
$$

这里的权重是标准化之后的权重(我们使用了出度标准化).

在实现中, 我们使用了一个三层GATConv层网络, 输出一个恒等激活的隐藏层变量, 作为节点的最终表征.

### 池化 & 分类

具体到这个问题上来说,  一个bill的主要属性是其发起者和协助发起者. 因此, 一个自然的想法就是用发起者+协助发起者的表征作为bill的表征, 即

$$\mathcal{R}_B = h_{s}\oplus h_{c_1}\oplus h_{c_2}\cdots$$

为了综合节点属性,降低参数复杂度, 我们引入一个注意力池化块

$$
h_{iB}= \text{AttentionPooling}(h_i, \mathcal{R}_B) 
$$

该池化块有一组参数$W_q, W_k, W_v$, 计算方法为

$$
AP(Q, K) = Attention(W_qQ, W_kK, W_vK)
$$

因为我们认为, 衡量一个节点是否会对特定提案投票, 取决于该节点的属性和该提案的契合程度, 所以Attention计算中, 节点属于Query, 而提案属性属于Key; Value则必须和Key一一对其, 所以只能从提案变换而来.

为了降低计算量, 我们观察到4位协助者就已经可以覆盖95%的提案, 所以至多计算4位协助.

分类器只是一个简单的单层MLP, 输入$h_{iB}$, 输出一个长$3$的logits.

## 效果

![](https://img2.imgtp.com/2024/05/06/6I2GH8SN.png)

测试集报告: 准确率0.98
