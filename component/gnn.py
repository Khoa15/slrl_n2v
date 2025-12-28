"""
This module contains Graph Neural Network layers.

1. GIN (Graph Isomorphism Network): Used for learning deep structural node embeddings (Feature Extraction).
   Replacing the legacy Node2Vec approach.

2. GraphConv: Used for the diffusion process in the RL environment (State Representation).
   This is NOT a trainable GNN layer but a fixed diffusion operator (similar to PageRank/PPR) 
   used to represent the current community state 'z_nodes'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from .layers import make_linear_block, Swish

class GINLayer(nn.Module):
    def __init__(self, input_dim, output_dim, eps=0, train_eps=True):
        super(GINLayer, self).__init__()
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            
        self.mlp = nn.Sequential(
            make_linear_block(input_dim, output_dim, act_cls=Swish, norm_type='batch_norm'),
            make_linear_block(output_dim, output_dim, act_cls=Swish, norm_type='batch_norm')
        )

    def forward(self, x, adj):
        # x: [N, input_dim]
        # adj: [N, N] sparse tensor
        
        # Aggregation: sum neighbors
        # out = adj @ x
        out = torch.spmm(adj, x)
        
        # GIN update: MLP((1 + eps) * x + aggregate(x))
        out = out + (1 + self.eps) * x
        out = self.mlp(out)
        return out

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.5):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        
        # Input projection if needed, or first layer handles it
        self.layers.append(GINLayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(hidden_dim, hidden_dim))
            
    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# Legacy GraphConv for compatibility if needed, or we can just remove it if we fully replace.
# Keeping a simple version of GraphConv or helper methods might be useful.
from .graph import Graph

class GraphConv:

    def __init__(self, graph: Graph, k: int = 3, alpha: float = 0.85):
        self.graph = graph
        self.k = k
        self.alpha = alpha
        self.normlized_adj_mat = self._normalize_adj(graph.adj_mat).astype(np.float32)

    def __repr__(self):
        return f'Conv_{self.k}_{self.alpha}'

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: sp.spmatrix):
        init_val = x
        for _ in range(self.k):
            x = self.alpha * (self.normlized_adj_mat @ x) + (1 - self.alpha) * init_val
        return x

    def updateGraph(self, graph: Graph):
        self.graph = graph
        self.normlized_adj_mat = self._normalize_adj(graph.adj_mat).astype(np.float32)

    @staticmethod
    def _normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def normalize_adj_torch(adj_matrix):
    """
    Normalize adjacency matrix for efficient sparse matrix multiplication in PyTorch.
    adj_matrix: scipy sparse matrix
    Returns: torch sparse tensor
    """
    if not isinstance(adj_matrix, sp.coo_matrix):
        adj_matrix = adj_matrix.tocoo()
    
    # Add self loops? GIN adds (1+eps)*x separately, so usually typical A is fine.
    # But usually GNNs use A + I or normalized A. 
    # SEAL/GIN usually works on raw A + sum pool. GIN equation handles self-connection via epsilon.
    # So we just convert A to sparse tensor.
    
    row = torch.from_numpy(adj_matrix.row.astype(np.int64))
    col = torch.from_numpy(adj_matrix.col.astype(np.int64))
    data = torch.from_numpy(adj_matrix.data.astype(np.float32))
    size = torch.Size(adj_matrix.shape)
    
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), data, size)
    return adj
