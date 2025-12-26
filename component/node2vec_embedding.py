import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import random

class Node2VecEmbedding:
    def __init__(self, dimension=64, walk_length=10, num_walks=10, window_size=5):
        self.dimension = dimension
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size

    def fit_transform(self, graph_nx):
        """
        Generates node embeddings for a given NetworkX graph using a DeepWalk-like approach
        (approximated via PMI Matrix Factorization for speed and no-gensim dependency).
        Returns a dictionary mapping node_id -> embedding_vector.
        """
        nodes = list(graph_nx.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)
        
        if n_nodes == 0:
            return {}

        # 1. Generate Random Walks
        walks = []
        for _ in range(self.num_walks):
            nodes_shuffled = list(nodes)
            random.shuffle(nodes_shuffled)
            for node in nodes_shuffled:
                walk = [node]
                curr = node
                for _ in range(self.walk_length - 1):
                    neighbors = list(graph_nx.neighbors(curr))
                    if neighbors:
                        curr = random.choice(neighbors)
                        walk.append(curr)
                    else:
                        break
                walks.append(walk)

        # 2. Build Co-occurrence Matrix (Skip-gram equivalent count)
        # We can use a simple approximate PMI approach without full word2vec training
        # But for community similarity, we just need stable structural features.
        # Let's use a dense SVD on the transition probabilities or adjacency k-step if graph is small.
        # Since we want "Node2Vec" specifically requested, we stick to the walk concept.
        
        # A faster way without gensim is using matrix factorization of the PPMI matrix derived from walks
        # Construct vocab
        
        co_occur = dict()
        
        for walk in walks:
            for i in range(len(walk)):
                for j in range(max(0, i - self.window_size), min(len(walk), i + self.window_size + 1)):
                    if i == j: continue
                    u = node_to_idx[walk[i]]
                    v = node_to_idx[walk[j]]
                    pair = (u, v)
                    co_occur[pair] = co_occur.get(pair, 0) + 1
        
        # Build Sparse Matrix
        rows = [p[0] for p in co_occur]
        cols = [p[1] for p in co_occur]
        data = [co_occur[p] for p in co_occur]
        
        C = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        # Compute PMI
        sum_total = sum(data)
        sum_rows = np.array(C.sum(axis=1)).flatten()
        sum_cols = np.array(C.sum(axis=0)).flatten()
        
        # Avoid division by zero
        sum_rows[sum_rows == 0] = 1
        sum_cols[sum_cols == 0] = 1
        
        # We need (D_row * D_col) / sum_total
        # P_ij = C_ij / sum_total
        # P_i = sum_row_i / sum_total
        # P_j = sum_col_j / sum_total
        # PMI = log(P_ij / (P_i * P_j)) = log( (C_ij * sum_total) / (sum_row_i * sum_col_j) )
        
        # We can do this efficiently with sparse operations or just densify if small.
        # Given these are local subgraphs or communities (usually small < 100-500 nodes), densify is fine.
        
        if n_nodes < 2000:
            C_dense = C.toarray()
            # Add eps
            C_dense = C_dense * sum_total
            outer = np.outer(sum_rows, sum_cols)
            # Clip
            
            with np.errstate(divide='ignore'):
                 PPMI = np.log(C_dense / outer)
            
            PPMI[np.isinf(PPMI)] = 0
            PPMI[PPMI < 0] = 0
            PPMI = np.nan_to_num(PPMI)
        else:
            # Fallback for huge graphs (unlikely here for local community detection settings)
            return {node: np.zeros(self.dimension) for node in nodes}

        # 3. SVD
        # k = min(self.dimension, n_nodes - 1)
        k = min(self.dimension, n_nodes)
        if k < 2:
            U = np.zeros((n_nodes, self.dimension))
        else:
            svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
            U = svd.fit_transform(PPMI)
            
        # Pad if n_nodes < dimension
        if U.shape[1] < self.dimension:
            pad = np.zeros((n_nodes, self.dimension - U.shape[1]))
            U = np.hstack([U, pad])

        return {node: U[i] for i, node in enumerate(nodes)}

    def get_community_embedding(self, graph_nx, community_nodes):
        """
        Computes the embedding of a community by averaging the embeddings of its nodes.
        Note: The graph_nx should ideally be the context graph (e.g. ego graph).
        If graph_nx is just the subgraph induced by community, the boundary info is lost.
        """
        embeddings = self.fit_transform(graph_nx)
        
        com_vecs = []
        for node in community_nodes:
            if node in embeddings:
                com_vecs.append(embeddings[node])
            else:
                com_vecs.append(np.zeros(self.dimension))
        
        if not com_vecs:
            return np.zeros(self.dimension)
            
        return np.mean(com_vecs, axis=0)

def compute_similarity_matrix_node2vec(communities, parent_graph_nx, dim=32):
    """
    Computes sim matrix between list of communities using Node2Vec graph embeddings.
    communities: list of list of nodes
    parent_graph_nx: the context graph
    """
    embedder = Node2VecEmbedding(dimension=dim, walk_length=8, num_walks=8, window_size=3)
    
    # Optimization: We can embed the whole parent graph ONCE, then aggregate for communities.
    # This is much faster.
    
    # 1. Embed parent graph
    # Check graph size. If parent graph is the large full graph, this is slow.
    # In Detector context, `knowcomSeedGraph` is the k-ego graph, which is relatively small.
    # So we can embed `detector.knowcomSeedGraph`.
    
    node_embeddings = embedder.fit_transform(parent_graph_nx)
    
    com_embeddings = []
    valid_indices = []
    
    for idx, com in enumerate(communities):
        vecs = []
        for node in com:
            if node in node_embeddings:
                vecs.append(node_embeddings[node])
        
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
            com_embeddings.append(mean_vec)
            valid_indices.append(idx)
        else:
            # Empty or invalid community handling
            com_embeddings.append(np.zeros(dim))
    
    if not com_embeddings:
        return np.zeros((len(communities), len(communities)))

    com_embeddings = np.array(com_embeddings)
    
    # Cosine Similarity
    # Reshape if 1D
    if com_embeddings.ndim == 1:
        com_embeddings = com_embeddings.reshape(1, -1)
        
    sim_matrix = cosine_similarity(com_embeddings)
    
    # Normalize to [0, 1] if needed, but cosine is [-1, 1].
    # Spectral clustering often expects non-negative affinity.
    # We can shift scale: (sim + 1) / 2
    sim_matrix = (sim_matrix + 1) / 2
    
    return sim_matrix
