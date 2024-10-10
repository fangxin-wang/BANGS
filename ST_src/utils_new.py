from collections import Counter

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx



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
    # transition_matrix = normalize_adjacency_matrix(adj_matrix).to_sparse()
    transition_matrix = adj_matrix
    influence_matrix = transition_matrix.clone()

    # print('Norm')
    influence_matrix_list = [transition_matrix]

    for _ in range(1, k):
        influence_matrix = torch.sparse.mm(influence_matrix, transition_matrix)
        influence_matrix_list.append(influence_matrix)
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
    M = torch.sparse.mm(D_inv, adj).to(device)

    for i in range(max_iter):
        pagerank_new = alpha * torch.sparse.mm(M, pagerank) + (1 - alpha) * I
        # Check for convergence based on the L1 norm
        ERR = torch.norm(pagerank_new - pagerank, p=1)
        if ERR < tol:
            pagerank = pagerank_new
            break
        print(i, '-th round, ERR:', ERR)
        pagerank = pagerank_new
    print(pagerank.shape)
    return pagerank #.to_sparse()

import torch
import dgl
import dgl.function as fn
import dgl.sparse as dglsp


def coo_matrix_2_sparse_tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def get_deg_matrix_sparse_tensor(g):
    N = g.number_of_nodes()
    # Precompute out-degree to normalize messages
    degs = g.out_degrees().float()
    #degs[degs == 0] = 1  # To avoid division by zero
    degs_inv = dglsp.diag((1. / degs))

    deg_coo = torch.sparse_coo_tensor(
        degs_inv.indices(),
        degs_inv.val,
        (N,N)
    )
    return deg_coo


def caluclate_W(g, device):

    deg_coo = get_deg_matrix_sparse_tensor(g).to(device)
    adj_mtx = g.adj_external(scipy_fmt='coo')
    adj_coo = torch.transpose( coo_matrix_2_sparse_tensor(adj_mtx) , 0, 1).to(device)
    # Nomarlized adj: W = AD^{-1} for random walk, normalized by end node (column)
    # Note: W = D^{-1/2}AD^{1/2} in model training
    W = torch.sparse.mm( adj_coo, deg_coo)
    print(torch.sum(W, dim=1), torch.sum(W, dim=0))
    return W


def compute_ppr_matrix_parallel(g, W, node_set = None, alpha=0.15, max_iter=100, tol=5e-6, T=None):
    N = g.number_of_nodes()
    n = node_set.size(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    W = W.to(device)

    if T is None:
        T = torch.ones(N, n).to(device) / n  # Uniform teleport matrix
    else:
        T = T.to(device)
        T = T / T.sum(dim=0, keepdim=True)  # Ensure columns sum to 1

    # Initialize PPR matrix
    pagerank = T.clone()

    for _ in range(max_iter):
        #print(_)
        prev_pagerank = pagerank.clone()
        pagerank = alpha * T + (1 - alpha) * torch.sparse.mm( W , prev_pagerank )
        # Check for convergence
        diff = torch.norm(pagerank - prev_pagerank, p='fro')
        print(_, "   ", diff)
        if diff < tol:
            break

    final = torch.transpose(pagerank,0,1)
    print(final.shape)
    # Check: columns should all sum to the same prob (random walk)
    # print(torch.sum(pagerank, dim=1))
    return final



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


def get_sparse_column(sparse_tensor, col_idx):
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    col_indices = (indices[1, :] == col_idx).nonzero(as_tuple=True)[0]

    row_indices = indices[0, col_indices]
    col_values = values[col_indices]

    sparse_col_tensor = torch.sparse_coo_tensor(
        torch.stack([row_indices, torch.zeros_like(row_indices)]),
        col_values,
        (sparse_tensor.size(0), 1),
        device=sparse_tensor.device
    )

    return sparse_col_tensor

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

def print_node_attributes(G, labels, pl_idx, idx_train, idx_train_ag, confidence, logger):
    confidence[idx_train_ag] = 1
    nx_G = G.to('cpu').to_networkx().to_undirected()
    centrality = nx.degree_centrality(nx_G)

    l_degree, l_centrality, l_confidence_2_hop, l_entropy = [], [], [], []
    total_neighbor_set = set()

    for node_tensor in pl_idx:

        node = node_tensor.item()
        # print(f"Node {node}:")
        # Degree
        degree = G.in_degrees(node)
        # print(f"  Degree: {degree}")
        l_degree.append(degree)

        # Centrality
        # print(f"  Centrality: {centrality[node]}")
        l_centrality.append(centrality[node])

        # Neighbors (1-hop)
        neighbors_1_hop = list(nx_G.neighbors(node))
        # Neighbors (2-hop)
        neighbors_2_hop_set = set()
        for n in neighbors_1_hop:
            neighbors_2_hop_set.update(nx_G.neighbors(n))
        neighbors_2_hop_set.difference_update(set(neighbors_1_hop + [node]))
        neighbors_2_hop = list(neighbors_2_hop_set)

        total_neighbor_set = total_neighbor_set.union(neighbors_2_hop_set)

        confidences_1_hop, confidences_2_hop = confidence[neighbors_1_hop], confidence[neighbors_2_hop]
        # print(confidences_1_hop)
        # print(confidences_2_hop)
        mean_confidence_1_hop = torch.nanmean(confidences_1_hop)
        mean_confidence_2_hop = torch.nanmean(confidences_2_hop)
        # print(f"  Mean Confidence 1-hop: {mean_confidence_1_hop}")
        # print(f"  Mean Confidence 2-hop: {mean_confidence_2_hop}")

        l_confidence_2_hop.append(mean_confidence_2_hop.item())

        label_frequency_1_hop = Counter(labels[neighbors_1_hop].tolist())
        label_frequency_2_hop = Counter(labels[neighbors_2_hop].tolist())
        total_labels = len(labels[neighbors_2_hop])
        # Calculate the entropy
        entropy = -sum(
            (count / total_labels) * math.log2(count / total_labels) for count in label_frequency_2_hop.values())

        # print(f" Neighborhood Label 1-hop: {label_frequency_1_hop}")
        # print(f" Neighborhood Label 2-hop: {label_frequency_2_hop}, Entropy: {entropy}")
        l_entropy.append(entropy)

    logger.info(f"l_degree: {np.nanmean(l_degree)}, l_centrality: {np.nanmean(l_centrality)}, "
                f"l_confidence_2_hop: {np.nanmean(l_confidence_2_hop)}, l_entropy: {np.nanmean(l_entropy)}, "
                f"total neighbor num: {len(total_neighbor_set)}")


def get_adaptive_threshold(output, idx_train, global_thres, local_thres, decay=0.9):
    # output = torch.softmax(output, dim=1)

    max_prob, argmax_pos = torch.max(output, dim=1)

    global_thres_updated = decay * global_thres + (1 - decay) * torch.mean(max_prob[~idx_train])
    local_thres_updated = decay * local_thres + (1 - decay) * torch.mean(output[~idx_train], dim=0)

    max_local_thres = torch.max(local_thres_updated)
    local_thres_final = local_thres_updated / max_local_thres * global_thres_updated

    mask = max_prob > local_thres_final[argmax_pos]

    return mask, global_thres_updated, local_thres_updated

def compute_ece(predictions, labels, n_bins=20):
    # Ensure predictions are probabilities
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    confidences, predictions = torch.max(predictions, dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=predictions.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=predictions.device)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def Entropy_utility(y_probs, idx_sampled, idx_train, idx_train_ag, idx_unlabeled, influence_matrix):
    device = influence_matrix.device
    # print(f"Device of influence_matrix: {device}")
    """
    y_probs: idx_train -> one hot
    """
    if idx_sampled.dim() == 0:
        idx_sampled = idx_sampled.unsqueeze(0)

    idx_train_ag, idx_train = torch.where(idx_train_ag == True)[0], torch.where(idx_train == True)[0]
    idx_pseudo_ag = torch.cat((idx_train_ag, idx_sampled), dim=0)

    # Creating a boolean mask for influenced nodes
    influenced_mask = torch.zeros(influence_matrix.size(0), dtype=torch.bool).to(device)

    # Iterate over idx_pseudo_ag to update the influenced_mask
    for idx in idx_pseudo_ag:
        column_tensor = get_sparse_column(influence_matrix, idx)
        # print(column_tensor.device, influenced_mask.device)
        influenced_mask |= (column_tensor.to_dense().squeeze() > 0)

    influenced_nodes = torch.where(influenced_mask & idx_unlabeled)[0]
    # print("influenced_nodes")

    # ####################
    # # Randomly select positions to set to False
    # bool_tensor = influenced_mask & idx_unlabeled
    # num_to_set_false = int( len(bool_tensor)/2 )
    # indices_to_set_false = torch.randperm(len(bool_tensor))[:num_to_set_false]
    #
    # # Set the selected positions to False
    # bool_tensor[indices_to_set_false] = False
    # influenced_nodes = torch.where(bool_tensor) [0]
    # ####################

    influence_matrix_influenced = influence_matrix.index_select(0, influenced_nodes)
    influence_submatrix = influence_matrix_influenced.index_select(1, idx_pseudo_ag)
    # print(influence_matrix_influenced.size(), influence_submatrix.size())

    total_prob_origin = F.normalize(
        torch.sparse.mm(influence_submatrix, y_probs[idx_pseudo_ag]), dim=1)

    entropies = -torch.sum(total_prob_origin * torch.log(total_prob_origin + 1e-9), dim=1)
    # print(total_prob_origin.size(), entropies.size())
    total_Ent = entropies.sum()
    # print('total_Ent', total_Ent.item() )

    # total_Ent = 0
    # for node_j in influenced_nodes:
    #     total_prob_origin = F.normalize(
    #         torch.matmul(influence_matrix[node_j].to_dense()[idx_pseudo_ag], y_probs[idx_pseudo_ag]), dim=0)
    #     entropy_origin = calculate_entropy(total_prob_origin)
    #     total_Ent += entropy_origin
    #
    # print('total_Ent', total_Ent)

    influence_matrix_train = influence_matrix.index_select(0, idx_train)
    influence_submatrix_train = influence_matrix_train.index_select(1, idx_pseudo_ag)

    Prob_train_prop = F.normalize(
        torch.sparse.mm(influence_submatrix_train, y_probs[idx_pseudo_ag]), dim=1
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    Cross_Ent = criterion(Prob_train_prop, y_probs[idx_train])

    # print('Cross_Ent', Cross_Ent.item() )

    return - total_Ent - Cross_Ent
