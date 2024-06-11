import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def normalize_adjacency_matrix(adj_matrix):
    """
    Normalize an adjacency matrix to make it a transition matrix.
    Each row sums to 1.
    """
    row_sums = adj_matrix.sum(axis=1, keepdims=True) # Row-wise
    # Avoid division by zero for isolated nodes
    row_sums[row_sums == 0] = 1
    return adj_matrix / row_sums

def compute_influence_score(adj_matrix, k):
    """
    Compute the influence score matrix for k steps.
    """
    # Normalize the adjacency matrix to get the transition matrix
    transition_matrix = normalize_adjacency_matrix(adj_matrix)
    # Raise the transition matrix to the power of k to compute k-step transitions
    influence_matrix = np.linalg.matrix_power(transition_matrix, k)
    return influence_matrix

def get_influence_matrix(adj_matrix,k):
    """
    Normalize the influence score matrix by each arriving node.
    """
    influence_matrix_list =[]

    for i in range(1,k+1):
        influence_matrix_hopk = compute_influence_score(adj_matrix, k)
        influence_matrix_list.append( influence_matrix_hopk )

    return influence_matrix_list


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

