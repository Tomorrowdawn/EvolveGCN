import torch
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import numpy as np
from networkx.algorithms.community.quality import modularity


class SubclubInspector:
    def __init__(self, dgl_graph):
        """
        Parameters:
        - dgl_graph (dgl.DGLGraph): 要进行聚类的 DGL 图对象。要求edata权重信息。
        """
        self.dgl_graph = dgl_graph
        self.nx_graph = self.dgl_to_nx_with_edge_weights(dgl_graph)

    def dgl_to_nx_with_edge_weights(self, dgl_graph):
        # DGL graph to NetworkX graph
        nx_graph = dgl_graph.to_networkx()
        weights = dgl_graph.edata["weight"].numpy()
        for i, (u, v, data) in enumerate(nx_graph.edges(data=True)):
            data["weight"] = weights[i]
        return nx_graph

    def louvain_clustering(self):
        """
        使用 Louvain 算法对图进行聚类。

        Parameters:
        - graph (networkx.Graph): 要进行聚类的图对象。

        Returns:
        - result (numpy.ndarray): 表示聚类结果的 numpy 数组，数组的每个元素对应图中的一个节点，其值表示节点所属的聚类编号。
        """
        partition = nx.algorithms.community.greedy_modularity_communities(self.nx_graph)
        result = np.zeros(self.dgl_graph.number_of_nodes())
        for idx, cluster in enumerate(partition):
            for node in cluster:
                result[node] = idx
        return result

    def girvan_newman_clustering(self):
        """
        使用 Girvan-Newman 算法进行图的层次聚类，并计算Q模块度（Q-Modularity）。

        Parameters:
        - graph (nx.Graph): 要进行聚类的 NetworkX 图对象。

        Returns:
        - result (numpy.ndarray): 聚类结果，每个节点的所属聚类编号。
        - max_q (float): 聚类对应的Q模块度。
        """
        partition = nx.algorithms.community.girvan_newman(self.nx_graph)
        result = np.zeros(self.dgl_graph.number_of_nodes())
        max_q = -1
        for communities in partition:
            q = modularity(self.nx_graph, communities)
            if q > max_q:
                max_q = q
                result = np.zeros(self.dgl_graph.number_of_nodes())
                for idx, cluster in enumerate(communities):
                    for node in cluster:
                        result[node] = idx
        return result, max_q

    def agglomerative_clustering(
        self, linkage, n_clusters=None, distance_threshold=None
    ):
        """
        使用凝聚式聚类算法进行聚类。

        Parameters:
        - graph (nx.Graph): 要进行聚类的 NetworkX 图对象。
        - linkage (str): 聚类算法使用的链接标准，可选值为 'ward', 'complete', 'average'。
        - n_clusters (int): 聚类数量，当 distance_threshold 为 None 时生效。
        - distance_threshold (float): 距离阈值，当为 None 时，使用 n_clusters 进行聚类。

        Returns:
        - labels (numpy.ndarray): 聚类结果，每个元素表示节点所属的聚类编号。
        """
        if not n_clusters and not distance_threshold:
            n_clusters = 2

        clustering = AgglomerativeClustering(
            linkage=linkage,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
        )
        clustering.fit(nx.to_numpy_array(self.nx_graph))
        return clustering.labels_

    def kmeans_clustering(self, num_clusters=2, random_state=42):
        """
        使用 KMeans 算法进行聚类。

        Parameters:
        - num_clusters (int): 聚类数量。

        Returns:
        - result (numpy.ndarray): 聚类结果，每个节点的所属聚类编号。
        """
        kmeans_result = KMeans(
            n_clusters=num_clusters, random_state=random_state
        ).fit_predict(nx.to_numpy_array(self.nx_graph))
        return kmeans_result

    def spectral_clustering(self, num_clusters=2, affinity="nearest_neighbors"):
        """
        使用谱聚类算法进行聚类。

        Parameters:
        - num_clusters (int): 聚类数量。
        - affinity (str): 用于计算图节点之间相似度的方法，可选值为 'nearest_neighbors', 'rbf'。

        Returns:
        - result (numpy.ndarray): 聚类结果，每个节点的所属聚类编号。
        """
        spectral_result = SpectralClustering(
            n_clusters=num_clusters, affinity=affinity
        ).fit_predict(nx.to_numpy_array(self.nx_graph))
        return spectral_result

    def visualize_clusters(
        self, clustering_result, figsize=(8, 6), node_size=300, path=None
    ):
        """
        绘制带有聚类信息的图。

        Parameters:
        - clustering_result (numpy.ndarray): 节点的聚类结果。
        - figsize (tuple): 图的大小。
        - node_size (int): 节点的大小。
        - path (str): 图的保存路径。为None不保存图。

        Returns:
        - None
        """
        pos = nx.spring_layout(self.nx_graph)
        plt.figure(figsize=figsize)

        # 绘制每个聚类的节点
        for cluster_id in np.unique(clustering_result):
            nodes_in_cluster = np.where(clustering_result == cluster_id)[0]
            nx.draw_networkx_nodes(
                self.nx_graph,
                pos,
                nodelist=nodes_in_cluster,
                node_color="C" + str(int(cluster_id)),
                node_size=300,
            )

        nx.draw_networkx_edges(self.nx_graph, pos)
        plt.title("Graph with Clusters")

        if path:
            plt.savefig(path)

        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建一个空的无向图
    G = nx.Graph()

    # 添加节点
    num_nodes = 15
    nodes_with_id = [(i, {"node_id": 2 * i}) for i in range(num_nodes)]
    G.add_nodes_from(nodes_with_id)

    # 添加边并赋予边权重
    edges = [
        (0, 1, {"weight": 0.5}),
        (0, 2, {"weight": 0.8}),
        (1, 3, {"weight": 0.3}),
        (2, 3, {"weight": 0.9}),
        (3, 4, {"weight": 0.7}),
        (4, 5, {"weight": 0.6}),
        (4, 6, {"weight": 0.4}),
        (5, 7, {"weight": 0.2}),
        (5, 8, {"weight": 0.5}),
        (6, 8, {"weight": 0.1}),
        (7, 9, {"weight": 0.3}),
        (8, 9, {"weight": 0.6}),
        (9, 10, {"weight": 0.4}),
        (9, 11, {"weight": 0.7}),
        (10, 12, {"weight": 0.9}),
        (11, 12, {"weight": 0.8}),
        (12, 13, {"weight": 0.5}),
        (12, 14, {"weight": 0.3}),
    ]

    G.add_edges_from(edges)

    # 将NetworkX图转换为DGLGraph
    dgl_G = dgl.from_networkx(G)

    # 添加边权重特征
    weights = torch.tensor(
        [edata["weight"] for u, v, edata in G.edges(data=True) for _ in range(2)]
    )
    dgl_G.edata["weight"] = weights
    node_ids = torch.arange(num_nodes)  # 假设节点ID从0开始递增
    dgl_G.ndata["node_id"] = node_ids

    # # 输出DGL图的一些基本信息
    # print("DGL图的节点数量：", dgl_G.number_of_nodes())
    # print("DGL图的边数量：", dgl_G.number_of_edges())
    # print("DGL图的边权重：", dgl_G.edata["weight"])
    # print("DGL图的节点ID：", dgl_G.ndata["node_id"])

    # 创建SubclubInspector对象，聚类作图分析
    subclub_inspector = SubclubInspector(dgl_G)
    louvain_result = subclub_inspector.louvain_clustering()
    print("Louvain Clustering Result:", louvain_result)
    subclub_inspector.visualize_clusters(louvain_result)

    test_others = True
    if test_others:
        # Girvan-Newman
        gn_result, max_q = subclub_inspector.girvan_newman_clustering()
        print("Girvan-Newman Clustering Result:", gn_result)
        print("Max Q:", max_q)
        subclub_inspector.visualize_clusters(gn_result, path="gn.png")

        # Agglomerative Clustering
        agg_result = subclub_inspector.agglomerative_clustering(
            linkage="ward", n_clusters=2
        )
        print("Agglomerative Clustering Result:", agg_result)
        subclub_inspector.visualize_clusters(agg_result)

        # KMeans
        kmeans_result = subclub_inspector.kmeans_clustering(num_clusters=2)
        print("KMeans Clustering Result:", kmeans_result)
        subclub_inspector.visualize_clusters(kmeans_result)

        # Spectral Clustering
        spectral_result = subclub_inspector.spectral_clustering(num_clusters=2)
        print("Spectral Clustering Result:", spectral_result)
        subclub_inspector.visualize_clusters(spectral_result)
