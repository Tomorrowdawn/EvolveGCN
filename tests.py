from model.ReprModule import ReprModule, WeightedGAT
import torch.nn as nn
import torch
import dgl
import torch

from model.predictor import *

import unittest


class TestGetEmbeddings(unittest.TestCase):
    def test_normal(self):
        input = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        indices = torch.tensor([[1, 2]])
        output = get_embeddings(input, indices)
        self.assertEqual(len(output), 1)
        self.assertTrue(torch.tensor([[[3, 4], [5, 6]]]).equal(output[0]))

    def test_neg_indice(self):
        """检测indice=-1是否返回0"""
        input = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        indices = torch.tensor([[-1]])
        output = get_embeddings(input, indices)
        self.assertEqual(len(output), 1)
        self.assertTrue(torch.tensor([[[0, 0]]]).equal(output[0]))


class TestAttentionPooling(unittest.TestCase):
    def test_normal(self):
        torch.manual_seed(114514)
        input_dim = 2
        module = AttentionPooling(input_dim)
        query = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.float
        )
        key = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.float)
        output = module.forward(query, key)
        self.assertTrue(
            output.allclose(
                torch.tensor(
                    [
                        [1.066759, 0.5741351],
                        [1.334198, 0.614905],
                        [1.3683974, 0.62011856],
                        [1.3747227, 0.6210828],
                        [1.3759941, 0.62127656],
                    ]
                )
            )
        )

    def test_dimension(self):
        input_dim = 2
        module = AttentionPooling(input_dim)
        query = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.float
        )
        key = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.float)
        output = module.forward(query, key)
        self.assertEqual(query.shape, key.shape)
        self.assertEqual(query.shape, output.shape)
        self.assertEqual(output.shape[-1], input_dim)


class TestWeightedGAT(unittest.TestCase):
    def setUp(self) -> None:
        g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
        self.g = dgl.add_self_loop(g)

    def test_dimension(self):
        input_dim, embedding_size = 2, 3
        module = WeightedGAT(input_dim, embedding_size)
        input = nn.Embedding(6, input_dim).weight
        self.g.edata["weight"] = torch.randn(12)
        edge_weights = self.g.edata["weight"]
        output = module.forward(self.g, input, edge_weights)
        self.assertEqual(list(output.shape), [6, 3])

    def test_normal(self):
        torch.manual_seed(41)
        input_dim, embedding_size = 2, 3
        module = WeightedGAT(input_dim, embedding_size)
        input = nn.Embedding(6, input_dim)(torch.tensor([0, 1, 2, 3, 4, 5]))
        self.g.edata["weight"] = torch.randn(12)
        edge_weights = self.g.edata["weight"]
        output = module.forward(self.g, input, edge_weights)
        self.assertTrue(
            torch.tensor(
                [
                    [0.30417174, -6.352266, 0.9566772],
                    [1.6590778, -0.06782746, 0.8238987],
                    [-1.960337, 8.893644, -1.7529076],
                    [0.99067044, 3.670627, 4.4161487],
                    [-0.00902292, -0.05226272, -0.6447347],
                    [-0.4780674, -0.39514548, -2.0522437],
                ]
            ).allclose(output)
        )


class TestGATPredictor(unittest.TestCase):
    def setUp(self) -> None:
        g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
        self.g = dgl.add_self_loop(g)

    def test_dimension(self):
        pass


unittest.main(verbosity=2)
