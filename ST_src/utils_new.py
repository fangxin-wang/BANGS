import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl

# def normalize_adjacency_matrix(adj_matrix):
#     """
#     Normalize an adjacency matrix to make it a transition matrix.
#     Each row sums to 1.
#     """
#     row_sums = adj_matrix.sum(axis=1, keepdims=True) # Row-wise
#     # Avoid division by zero for isolated nodes
#     row_sums[row_sums == 0] = 1
#     return adj_matrix / row_sums
#
# def compute_influence_score(adj_matrix, k):
#     """
#     Compute the influence score matrix for k steps.
#     """
#     # Normalize the adjacency matrix to get the transition matrix
#     transition_matrix = normalize_adjacency_matrix(adj_matrix)
#     # Raise the transition matrix to the power of k to compute k-step transitions
#     influence_matrix = np.linalg.matrix_power(transition_matrix, k)
#     return influence_matrix
#
# def get_influence_matrix(adj_matrix,k):
#     """
#     Normalize the influence score matrix by each arriving node.
#     """
#     influence_matrix_list =[]
#
#     for i in range(1,k+1):
#         influence_matrix_hopk = compute_influence_score(adj_matrix, k)
#         influence_matrix_list.append( influence_matrix_hopk )
#
#     return influence_matrix_list

# def normalize_adjacency_matrix(adj_matrix):
#     """
#     Normalize an adjacency matrix to make it a transition matrix.
#     Each row sums to 1.
#     """
#     row_sums = adj_matrix.sum(axis=1)  # Row-wise sums
#     # Avoid division by zero for isolated nodes
#     row_sums[row_sums == 0] = 1
#     # Normalize in-place
#     for i in range(adj_matrix.shape[0]):
#         adj_matrix[i, :] /= row_sums[i]
#     return adj_matrix

def get_influence_matrix(adj_matrix, k):
    """
    Compute the influence score matrix for k steps using a sparse adjacency matrix.
    """
    #transition_matrix = normalize_adjacency_matrix(adj_matrix).to_sparse()
    transition_matrix = adj_matrix
    influence_matrix = transition_matrix.clone()

    #print('Norm')
    influence_matrix_list = [transition_matrix]

    for _ in range(1, k):

        influence_matrix = torch.sparse.mm(influence_matrix, transition_matrix)
        influence_matrix_list.append( influence_matrix )
        print('k')

    return influence_matrix_list

def get_ppr_influence_matrix(adj, alpha=0.85, tol=1e-4, max_iter=100):
    device = adj.device
    N = adj.size(0)

    # Initialize the PageRank vector
    I = torch.eye(N, device=device)
    pagerank = I.clone()

    adj = adj.float()
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    D_inv = torch.diag(1.0 / degree)
    # Transition matrix M = D^(-1) * adj
    M = torch.sparse.mm(D_inv, adj)

    for i in range(max_iter):
        pagerank_new = alpha * torch.sparse.mm(M, pagerank) + (1 - alpha) * I
        # Check for convergence based on the L1 norm
        ERR = torch.norm(pagerank_new - pagerank, p=1)
        if ERR < tol:
            pagerank = pagerank_new
            break
        print( i, '-th round, ERR:', ERR )
        pagerank = pagerank_new
    print(pagerank.size() )
    return pagerank.to_sparse()


def get_k_hop_neighbors(adj_matrix, node_indices, k):
    device = adj_matrix.device
    #print(device)
    N = adj_matrix.size(0)
    neighbors = node_indices.clone()

    # Create a tensor to keep track of visited nodes
    visited = torch.zeros(N, dtype=torch.bool).to(device)
    visited[node_indices] = True

    # BFS to get k-hop neighbors
    current_level_nodes = node_indices

    for _ in range(k):
        # Find the neighbors of the current level nodes
        # Convert current level nodes to a dense one-hot vector
        current_level_one_hot = torch.zeros(N)
        current_level_one_hot[current_level_nodes] = 1
        current_level_one_hot = current_level_one_hot.unsqueeze(1).to(device)
        #print(adj_matrix.device, current_level_one_hot.device)

        # Multiply adjacency matrix with one-hot vector to get neighbors
        next_level_one_hot = torch.sparse.mm(adj_matrix, current_level_one_hot).squeeze(1)

        # Find the indices of the next level neighbors
        next_level_nodes = torch.nonzero(next_level_one_hot, as_tuple=False).flatten().to(device)

        # Remove already visited nodes
        next_level_nodes = next_level_nodes[~visited[next_level_nodes]]

        # Update the visited nodes
        visited[next_level_nodes] = True

        # Add new level neighbors to the list of all neighbors
        neighbors = torch.cat((neighbors, next_level_nodes))

        # Move to the next level
        current_level_nodes = next_level_nodes

    return neighbors

def add_sparse_matrices(mat1, mat2):
    """
    Adds two sparse matrices.
    """
    mat1 = mat1.coalesce()
    mat2 = mat2.coalesce()

    indices = torch.cat([mat1.indices(), mat2.indices()], dim=1)
    values = torch.cat([mat1.values(), mat2.values()])

    result = torch.sparse.FloatTensor(indices, values, mat1.size())
    return result.coalesce()


def scale_sparse_matrix(sparse_mat, scale):
    """
    Scales a sparse matrix by a scalar.
    """
    scaled_values = sparse_mat._values() * scale
    return torch.sparse.FloatTensor(sparse_mat._indices(), scaled_values, sparse_mat.size())


def calculate_entropy(distribution):
    """
    Calculate the Shannon entropy of a given probability distribution.
    """
    # Filter out zero probabilities to avoid log_before_12(0)
    distribution = distribution[distribution > 0]
    return -torch.sum(distribution * torch.log2(distribution))
#
# def get_IGP(node_j, node_i, labeled_node_idx, influence_matrix, y_probs, NORM = True):
#     """
#     Calculate IGP from node i to node j, considering entropy of all labeled or pseudo-labeled nodes.
#     """
#     influence_matrix_labeled = torch.from_numpy( influence_matrix[labeled_node_idx,node_j] ). float()
#     if NORM:
#         normalized_influence_matrix_labeled = F.normalize(influence_matrix_labeled, p = 1, dim = 0)
#         total_prob_origin = F.normalize( np.matmul(normalized_influence_matrix_labeled , y_probs[labeled_node_idx]) , dim = 0)
#     else:
#         total_prob_origin = F.normalize(np.matmul(influence_matrix_labeled, y_probs[labeled_node_idx]),dim=0)
#
#     entropy_origin = calculate_entropy(total_prob_origin)
#
#     #print("entropy_origin",entropy_origin)
#
#     num_class = y_probs.shape[1]
#     node_i_prob = y_probs[node_i]
#     #most_likely_label =  torch.argmax(node_i_prob)
#
#     expected_IGP = entropy_origin
#     for label in range(num_class):
#         # if most_likely_label == label:
#             # print('No gain')
#             # continue
#
#         class_prob = node_i_prob[label]
#         if class_prob > 0:
#             node_i_prob_one_hot = np.zeros(num_class)
#             node_i_prob_one_hot[label] = 1
#             y_probs[node_i] =  torch.from_numpy( node_i_prob_one_hot).type_as(y_probs)
#             labeled_node_idx_add = torch.concat((labeled_node_idx, torch.unsqueeze(node_i, 0)))
#
#             # add another row in total_prob_origin
#             if NORM:
#                 influence_matrix_labeled_add = F.normalize( torch.from_numpy(influence_matrix[labeled_node_idx_add,node_j] ). float() , p = 1, dim = 0)
#                 total_prob_add = F.normalize( np.matmul(influence_matrix_labeled_add, y_probs[labeled_node_idx_add]), dim = 0 )
#
#             else:
#                 influence_matrix_labeled_add =  torch.from_numpy(influence_matrix[labeled_node_idx_add, node_j]).float()
#                 total_prob_add = F.normalize(np.matmul(influence_matrix_labeled_add, y_probs[labeled_node_idx_add]),
#                                              dim=0)
#
#             entropy_add = calculate_entropy(total_prob_add)
#             expected_IGP -= class_prob * entropy_add
#             #print("entropy_add", entropy_add)
#     #print("expected_IGP",expected_IGP)
#     return expected_IGP


def plot_tensor_distribution(tensor):
    """
    Plots the histogram of the tensor's values to show its distribution.

    Args:
    - tensor (torch.Tensor): A 1D tensor whose distribution is to be plotted.
    """
    # Convert tensor to numpy array if it's not already detached
    if tensor.requires_grad:
        tensor = tensor.detach()

    tensor_np = tensor.numpy()  # Convert tensor to a NumPy array for plotting

    # Create the histogram
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.hist(tensor_np, bins=30, alpha=0.75, color='blue')  # Histogram with 30 bins
    plt.title('Distribution of Tensor Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def find_intersection(tensor2_sorted, tensor1_sorted):
    idx = torch.searchsorted(tensor2_sorted, tensor1_sorted)
    mask = idx < tensor2_sorted.size(0)
    matched = tensor2_sorted[idx[mask]] == tensor1_sorted[mask]
    return tensor1_sorted[mask][matched]

