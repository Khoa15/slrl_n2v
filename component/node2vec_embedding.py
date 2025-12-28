'''
LEGACY CODE: Node2VecEmbedding
This module is no longer used in the main pipeline. 
It has been replaced by GIN (Graph Isomorphism Network) in component/gnn.py and component/detector.py.
We keep this file for reference or fallback if needed.

import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import random

class Node2VecEmbedding:
    ...
def compute_similarity_matrix_node2vec(communities, parent_graph_nx, dim=32):
    ...
'''
