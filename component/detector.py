import copy

import networkx as nx
import numpy as np
import torch
import time
import random

from openpyxl import Workbook

from component.agent import Agent
from torch import optim
from grakel import Graph as gGraph
from grakel.kernels import ShortestPath
from sklearn.cluster import SpectralClustering
from component.expander import Expander
from component.graph import Graph
from utils import wr_file
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import cosine_similarity
# from component.node2vec_embedding import compute_similarity_matrix_node2vec




class Detector:

    def __init__(self, args, seed, com_index, logger=None):
        self.args = args
        self.logger = logger
        # 获取图、已知社区、种子节点
        self.graph, self.coms = self.loadDataset(args.root, args.dataset)
        self.oldKnowcoms = self.coms[-args.train_size:]   # 后100
        self.oldSeed = seed

        if args.dataset == "twitter":
            fileedge = f"datasets/{self.args.dataset}/{self.args.dataset}-1.90.ungraph.txt"
            G = self.networkx(fileedge)
            self.oldKnowcoms = self.remove_disconnected_communities(self.oldKnowcoms, G)
            if self.logger: self.logger.log(f"Connected communities count: {len(self.oldKnowcoms)}")

        # 获取子图（种子节点的k-ego以及已知社区的k层邻居），给所有节点重新编号，记录映射关系
        knowcomSeed_nodes = set([node for com in self.coms[-args.train_size:] for node in com] + [seed])   # 后100
        self.knowcomSeedGraph, self.old_to_new_node_mapping = self.graph.get_k_layer_subgraph_and_mapping(knowcomSeed_nodes, args.k_ego_subG)
        self.knowcomSeedGraph.setParentGraph(self.graph)
        # 反转映射以创建新节点ID映射到旧节点ID的字典
        self.new_to_old_node_mapping = {new_id: old_id for old_id, new_id in self.old_to_new_node_mapping.items()}
        self.args.old_to_new_node_mapping = self.old_to_new_node_mapping
        self.args.new_to_old_node_mapping = self.new_to_old_node_mapping

        # 给节点重新编号，获取新编号后的种子节点，已知社区
        self.knowcoms = [[self.old_to_new_node_mapping[node] for node in coms] for coms in self.oldKnowcoms]
        self.args.max_size = max(len(x) for x in self.knowcoms)
        self.train_comms = self.knowcoms
        self.seed = self.old_to_new_node_mapping[seed]
        self.com_index = com_index
        # self.computeSimiAndWrite()

        # 初始化expander
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.expander = self.init_expander()

    def is_connected_graph(self, graph):
        return nx.is_connected(graph)

    def remove_disconnected_communities(self, communities, G):
        connected_communities = []
        for community in communities:
            community_graph = G.subgraph(community)
            if self.is_connected_graph(community_graph):
                connected_communities.append(community)
        return connected_communities

    def networkx(self, filename):
        """--------------------------------------------------------------------------------
                     function:       把一个含有边的txt数据集表示成networkx
                     Parameters:     filename：文件名称 .txt格式
                     Returns：       G：表示成networkx的图
                    ---------------------------------------------------------------------------------"""
        fin = open(filename, 'r')
        G = nx.Graph()
        for line in fin:
            data = line.split()
            if data[0] != '#':
                G.add_edge(int(data[0]), int(data[1]))
        return G

    def loadDataset(self, root, dataset):
        '''
        加载数据集
        @param root: 根目录
        @param dataset: 数据集名称
        '''
        with open(f'{root}/{dataset}/{dataset}-1.90.ungraph.txt') as fh:
            edges = fh.read().strip().split('\n')
            edges = np.array([[int(i) for i in x.split()] for x in edges])
        with open(f'{root}/{dataset}/{dataset}-1.90.cmty.txt') as fh:
            comms = fh.read().strip().split('\n')
            comms = [[int(i) for i in x.split()] for x in comms]
        graph = Graph(edges)
        return graph, comms

    def init_expander(self):
        '''
        初始化expander
        '''
        tobelog = self.logger.log if self.logger else print
        from component.gnn import GIN, normalize_adj_torch

        args = self.args
        device = self.device
        
        # --- Deep Learning: GIN Embeddings ---
        tobelog("Initializing GIN for Deep RL Agent...")
        embedding_dim = getattr(args, 'embedding_dim', 32)
        
        # GIN Model
        # Input dim for GIN? If we use constant features, we can choose any dim.
        # Let's use embedding_dim as input dim for simplicity (using Ones).
        gin_model = GIN(input_dim=embedding_dim, hidden_dim=embedding_dim, num_layers=3).to(device)
        
        # Input dim = 1 (Diffusion Scalar) + Embedding Dim (GIN Output)
        agent_input_dim = 1 + embedding_dim
        tobelog(f"Agent Input Dimension: {agent_input_dim}")

        expander_model = Agent(args.hidden_size, input_dim=agent_input_dim).to(device)
        
        # Combine parameters for optimizer
        all_params = list(expander_model.parameters()) + list(gin_model.parameters())
        expander_optimizer = optim.Adam(all_params, lr=args.g_lr)
        
        expander = Expander(args, self.knowcomSeedGraph, expander_model, expander_optimizer, device,
                      max_size=args.max_size, gnn_model=gin_model)
        return expander

    def detect(self):
        '''
        检测社区
        '''
        tobelog = self.logger.log if self.logger else print
        res = []
        pred_com = [[self.seed]]
        tic = time.time()
        for iter_num in range(2):
            if iter_num != 0:
                self.updateTraincom(pred_com[0])
            for _ in range(self.args.epochs):
                self.train_expander()
            tobelog('=' * 50)
            tobelog(f'Iter_{iter_num}[Test]')
            if iter_num == 1:
                pred_com = [[self.seed]]
            pred_com = self.expander.generateCommunity(pred_com)
            pred_com = [x[:-1] if x[-1] == 'EOS' else x for x in pred_com]
            oldID_pred_com = [self.new_to_old_node_mapping[node] for node in pred_com[0]]
            
            # --- Evaluation ---
            try:
                # com_index is an int (index of the true community in self.coms)
                true_idx = self.com_index
                if 0 <= true_idx < len(self.coms):
                    true_community = self.coms[true_idx]
                    p, r, f1, j = self.expander.eval_scores(oldID_pred_com, true_community)
                    tobelog(f"  [Iter {iter_num} Eval] Prec: {p:.4f} | Rec: {r:.4f} | F1: {f1:.4f} | Jacc: {j:.4f} | Size: {len(oldID_pred_com)} | TrueSize: {len(true_community)}")
            except Exception as e:
                tobelog(f"  [Eval Error] Could not evaluate: {e}")

            if iter_num == 0:
                if self.args.ablation == 1:
                    # 消融实验
                    wr_file(self.oldSeed, self.com_index, oldID_pred_com, self.args)
                continue
            res = [self.oldSeed, self.com_index, oldID_pred_com]
        toc = time.time()
        tobelog(f'Elapsed Time: {(toc - tic) // 60} min {(toc - tic) % 60}s')
        return res

    def select_lists(self, matrix, n):
        '''
        从训练集中随机选择n个社区
        @param matrix: 训练集
        @param n: batch
        @return:
        '''
        num_lists = list(range(len(matrix)))
        while len(num_lists) < n:
            num_lists = num_lists + num_lists
        random_indices = np.random.choice(num_lists, size=n, replace=True)
        selected_lists = [matrix[i] for i in random_indices]
        return selected_lists

    def train_expander(self):
        '''
        训练expander
        '''
        seeds = []
        true_coms = self.select_lists(self.train_comms, self.args.g_batch_size)
        for com in true_coms:
            seeds.append(random.choice(com))

        # Reinforcement Learning
        self.expander.trainReward(seeds, true_coms)

        # Teacher Forcing
        true_comms = random.choices(self.train_comms, k=self.args.g_batch_size)
        true_comms = [self.knowcomSeedGraph.sample_expansion_from_community(x) for x in true_comms]
        self.expander.train_from_sets(true_comms)


    def computeSimiAndWrite(self):
        # NOT USED currently, but updated to avoid Node2Vec dependency if enabled
        pass 
        # Logic removed to avoid confusion and dependency since it was commented out in init.

    def compute_similarity_using_gin(self, communities):
        """
        Compute similarity matrix using GIN embeddings from self.expander
        """
        if self.expander.current_embeddings is None:
             # If called before any training/forward, we need to force a GIN pass.
             # Or return identity/zeros. But it's usually called after iter 0.
             return np.zeros((len(communities), len(communities)))
        
        # Get embeddings: Tensor [N, Dim] on Device
        node_embs = self.expander.current_embeddings
        
        com_vecs = []
        for com in communities:
            # Filter valid nodes
            valid_nodes = [n for n in com if n < len(node_embs)]
            if not valid_nodes:
                com_vecs.append(torch.zeros(node_embs.shape[1]).to(self.device))
                continue
            
            indices = torch.tensor(valid_nodes, dtype=torch.long).to(self.device)
            # Gather
            vecs = node_embs[indices] # [m, dim]
            mean_vec = torch.mean(vecs, dim=0) # [dim]
            com_vecs.append(mean_vec)
            
        if not com_vecs:
             return np.zeros((len(communities), len(communities)))
            
        # Stack
        com_matrix = torch.stack(com_vecs) # [Num_Coms, Dim]
        
        # Cosine Sim
        # Move to cpu for sklearn or do in torch
        com_matrix_np = com_matrix.detach().cpu().numpy()
        
        sim_matrix = cosine_similarity(com_matrix_np)
        return (sim_matrix + 1) / 2 # Normalize to [0, 1]

        return traincom

    def UsingEnsembleSelectCom(self, simi, communities, K=2):
        '''
        Ensemble Clustering: Spectral + Agglomerative + KMedoids
        '''
        # 1. Prepare Distance Matrix
        dist_matrix = 1 - simi
        np.fill_diagonal(dist_matrix, 0)
        dist_sq = squareform(dist_matrix, checks=False) # Ensure it's condensed for linkage if needed, or square for others

        n_samples = simi.shape[0]
        co_association = np.zeros((n_samples, n_samples))
        
        # --- Algo 1: Spectral Clustering ---
        try:
            sc = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=0)
            labels1 = sc.fit_predict(simi)
        except:
            labels1 = np.zeros(n_samples) # Fallback

        # --- Algo 2: Agglomerative (Hierarchical) ---
        try:
            # Complete linkage tends to find compact clusters
            linked = linkage(dist_sq, 'complete')
            labels2 = fcluster(linked, K, criterion='maxclust')
            # Adjust to 0-indexed if fcluster returns 1-based (it usually does)
            labels2 = labels2 - 1
        except:
            labels2 = np.zeros(n_samples)

        # --- Algo 3: K-Medoids ---
        try:
            kmedoids = KMedoids(n_clusters=K, metric='precomputed', random_state=0)
            kmedoids.fit(dist_matrix)
            labels3 = kmedoids.labels_
        except:
            labels3 = np.zeros(n_samples)

        # --- Consensus ---
        # Matrix Construct
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                score = 0
                if labels1[i] == labels1[j]: score += 1
                if labels2[i] == labels2[j]: score += 1
                if labels3[i] == labels3[j]: score += 1
                
                co_association[i][j] = score
                co_association[j][i] = score
        
        # Normalize
        co_association = co_association / 3.0
        np.fill_diagonal(co_association, 1.0)
        
        # Final Clustering on Co-association Matrix
        # Using Spectral again as it handles affinity well
        final_sc = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=42)
        try:
            final_labels = final_sc.fit_predict(co_association)
        except:
            # Fallback if singular
            final_labels = labels1

        # Select community with seed (index 0)
        traincom = []
        target_label = final_labels[0]
        for i in range(1, len(final_labels)):
            if final_labels[i] == target_label:
                traincom.append(communities[i])
        
        print(f"Ensemble Selected {len(traincom)} communities.")
        return traincom


    def updateTraincom(self, com):
        '''
        更新训练集
        @param com: 包含给定节点的局部结构
        '''
        communities_copy = copy.deepcopy(self.knowcoms)
        communities_copy.insert(0, com)
        
        # New GIN Implementation
        simi = self.compute_similarity_using_gin(communities_copy)
        simi = np.nan_to_num(simi)
        
        traincom = []
        k = self.args.k
        
        # Logic to choose method
        method = self.args.resfileName
        
        if method == "sp_cluster":
            traincom = self.UsingScSelectCom(simi, communities_copy, k)
        elif method == "KMedoids":
            traincom = self.UsingKMedoidsSelectCom(simi, communities_copy, k)
        elif method == "Gmm":
            traincom = self.UsingGmmSelectCom(simi, communities_copy, k)
        elif method == "CengCi":
            traincom = self.UsingCengCiSelectCom(simi, communities_copy, k)
        elif method == "Ensemble":  # Add this option
            traincom = self.UsingEnsembleSelectCom(simi, communities_copy, k)
        else:
            # Default fallback or if "sp_cluster" is default
            traincom = self.UsingScSelectCom(simi, communities_copy, k)
            
        if len(traincom) != 0:
            self.train_comms = traincom
        else:
            print("0000000")

    def com_trans_graph(self, knowcom):
        """--------------------------------------------------------------------------------
                     function:       将一组已知社区转换为最短路径形式表示的图
                     Parameters:     knowcom：给定的已知社区
                                     file_edge:网络图
                     Returns：       shortest_graph: 已知社区的最短路径图
                     ---------------------------------------------------------------------------------"""

        G = nx.from_numpy_array(self.knowcomSeedGraph.adj_mat)
        shortest_graph = []
        for com in knowcom:
            edges = G.subgraph(com).edges()
            G1 = nx.Graph()
            G1.add_nodes_from(com)
            G1.add_edges_from(edges)
            adj = np.array(nx.adjacency_matrix(G1).todense())
            shortest_graph.append(gGraph(adj))
        return shortest_graph

    def UsingScSelectCom(self, simi, communities, K=2):
        '''
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        '''
        # --- High-Order Enhancement (Novel Paradigm) ---
        # Improve affinity matrix by capturing 2nd-order proximity
        # M_new = M + M^2
        
        simi_2 = np.dot(simi, simi)
        if simi_2.max() > 0:
            simi_2 = simi_2 / simi_2.max()
        
        simi_enhanced = simi + simi_2
        
        # 选在其中的一类
        spectral_clustering = SpectralClustering(n_clusters=K, affinity='precomputed')
        labels = spectral_clustering.fit_predict(simi_enhanced)
        # 输出聚类结果
        simis, traincom = [], []
        # print(labels)
        # a = []
        # b = []
        #
        # for i in range(0, len(labels)):
        #     if labels[i] == 0:
        #         a.append(communities[i])
        #     else:
        #         b.append(communities[i])
        # print(f"a:{a}")
        # print(f"b:{b}")
        # a = [len(line) for line in a]
        # b = [len(line) for line in b]
        # print("a:", sum(a)/len(a), len(a))
        # print("b:", sum(b) / len(b), len(b))
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                simis.append(simi[0][i])
                traincom.append(communities[i])
        return traincom


    def UsingKMedoidsSelectCom(self, simi, communities, K=2):
        '''
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        '''
        # 将相似性矩阵转换为距离矩阵
        distance_matrix = 1 - simi
        # 初始化K-Medoids模型
        kmedoids = KMedoids(n_clusters=K, metric='precomputed', random_state=0)

        # 训练模型
        kmedoids.fit(distance_matrix)

        # 获取聚类标签
        labels = kmedoids.labels_
        print("Cluster labels:", labels)

        # 输出聚类结果
        traincom = []
        # print(labels)
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])
        return traincom


    def UsingGmmSelectCom(self, simi, communities, K=2):
        '''
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        '''
        # 将相似性矩阵转换为距离矩阵
        distance_matrix = squareform(pdist(simi, 'euclidean'))

        # 进行高斯混合模型聚类
        gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=0)
        gmm.fit(distance_matrix)
        labels = gmm.predict(distance_matrix)
        print("Cluster labels:", labels)

        # 输出聚类结果
        traincom = []
        # print(labels)
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])
        return traincom


    def UsingCengCiSelectCom(self, simi, communities, K=2):
        '''
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        '''
        # 将相似性矩阵转换为距离矩阵
        # dist_matrix = squareform(pdist(simi, 'euclidean'))
        dist_matrix = 1 - simi
        # print(dist_matrix)
        np.fill_diagonal(dist_matrix, 0)

        linked = linkage(squareform(dist_matrix), 'complete')

        # 使用K指定聚类数量
        labels = fcluster(linked, K, criterion='maxclust')
        print("Cluster labels:", labels)

        # 输出聚类结果
        traincom = []
        # print(labels)
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])

        return traincom


