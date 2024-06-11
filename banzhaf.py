import numpy as np
from random import randint
import torch
import torch.nn.functional as F
from utils import *
from utils_new import *


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

def uniform_sample(indices, K, N):
    if K > len(indices):
        raise ValueError("K cannot be greater than the number of indices available")

    result_matrix = [indices[torch.where(torch.rand(len(indices)) < 0.5)[0]] for _ in range(N)]
    return result_matrix


def uniform_sample_N(indices, K, N):
    if K > len(indices):
        raise ValueError("K cannot be greater than the number of indices available")

    # Sample indices N times

    result_matrix = torch.stack([indices[torch.randperm(len(indices))[:K]] for _ in range(N)])
    return result_matrix


def confidence_importance_sample(confidence, K, N, temperature=1):
    # Normalize the confidences to ensure they sum to 1 (if not already probabilities)
    # Lower temperature -> sharper distribution
    confidences_normalized = torch.softmax(confidence / temperature, dim=0)

    # Sample K indices N times based on the probabilities
    sampled_indices = torch.multinomial(confidences_normalized, num_samples=K * N, replacement=True)

    # Reshape the flat list of indices into an N x K matrix
    sampled_indices_matrix = sampled_indices.reshape(N, K)

    return sampled_indices_matrix


def convert_to_one_hot(labels, num_class):
    """
    Convert numerical labels to one-hot encoded labels.
    """
    # labels = labels.to(torch.int64)
    one_hot_labels = F.one_hot(labels, num_classes=num_class)

    return one_hot_labels


def calculate_entropy(distribution):
    """
    Calculate the Shannon entropy of a given probability distribution.
    """
    # Filter out zero probabilities to avoid log_before_12(0)
    distribution = distribution[distribution > 0]
    return -torch.sum(distribution * torch.log2(distribution))


# def Entropy_utility(y_probs, idx_sampled, idx_train_ag, influence_matrix):
#
#     influenced_columns = torch.where(torch.any(influence_matrix[idx_sampled] > 0, axis=0))[0]
#     idx_pseudo_ag = torch.cat((idx_train_ag, idx_sampled), dim=0)
#
#     total_Ent = 0
#     for node_j in influenced_columns:
#         normalized_influence_matrix_labeled = F.normalize(influence_matrix[idx_pseudo_ag, node_j], p=1, dim=0)
#         total_prob_origin = F.normalize(torch.matmul(normalized_influence_matrix_labeled, y_probs[idx_pseudo_ag]),
#                                         dim=0)
#         entropy_origin = calculate_entropy(total_prob_origin)
#         total_Ent += entropy_origin
#
#     return total_Ent

def probs_convert_to_one_hot_tensor(prob_tensor):

    _, max_indices = torch.max(prob_tensor, dim=1)

    # Convert these indices to a one-hot encoded tensor
    one_hot_tensor = F.one_hot(max_indices, num_classes=prob_tensor.shape[1])

    return one_hot_tensor.float()

def Entropy_utility(y_probs, idx_sampled, idx_train, idx_train_ag, idx_unlabeled, influence_matrix):
    """
    y_probs: idx_train -> one hot
    """
    if idx_sampled.dim() == 0:
        idx_sampled = idx_sampled.unsqueeze(0)

    idx_train_ag,idx_train =  torch.where( idx_train_ag == True) [0], torch.where( idx_train == True) [0]
    idx_pseudo_ag = torch.cat((idx_train_ag, idx_sampled), dim=0)
    influenced_nodes = torch.where(torch.any(influence_matrix[:, idx_pseudo_ag] > 0, axis=1) & idx_unlabeled )[0]


    # assume the pseudo labels are fixed
    # y_probs[idx_pseudo_ag] = probs_convert_to_one_hot_tensor(y_probs[idx_pseudo_ag])

    # Information propogated to currently unlabeled (including pl) nodes from labeled and pl nodes
    total_Ent = 0
    for node_j in influenced_nodes:
        # node j influenced by all pl & labeled nodes
        total_prob_origin = F.normalize(torch.matmul(influence_matrix[node_j, idx_pseudo_ag], y_probs[idx_pseudo_ag]),
                                        dim=0)
        entropy_origin = calculate_entropy(total_prob_origin)
        total_Ent += entropy_origin


    #return total_Ent


    #Cross entropy loss between GT label and distribution propogated by pls

    influence_matrix_train = influence_matrix[idx_train,:]

    Prob_train_prop = F.normalize(torch.matmul(influence_matrix_train[:, idx_pseudo_ag], y_probs[idx_pseudo_ag]),
                                        dim=0)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
    Cross_Ent = criterion( Prob_train_prop, y_probs[idx_train] )
    #print( "Total_Ent ", total_Ent, " Cross_Ent", Cross_Ent)

    return total_Ent -  Cross_Ent


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


def IGP_sample_banzhaf(output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args, avg_ece_validation,
                       N=100, FIX_NUM_SAMPLE=False):
    """
    Maximal Sample Reuse:
    - Sampling Budget : N
    """
    t = 50

    influence_matrix = torch.from_numpy(influence_matrix).float()
    # Step 0: Prepare probs of all nodes
    y_probs = output.clone()
    y_probs = torch.softmax(y_probs, dim=1)
    # confidence, pred_label = torch.max(y_probs, dim=1)
    num_class = y_probs.shape[1]
    y_probs[idx_train] = convert_to_one_hot(labels[idx_train], num_class).float()

    # Step 1: Sample from unlabeled nodes, with a fixed number of K in this round

    ####### Test
    # full_mask = torch.zeros(confidence.shape[0], dtype=torch.int)
    # full_mask[idx_unlabeled] = 1
    # idx_to_label = torch.where ( (confidence > args.threshold ) & full_mask )[0]
    # print(idx_to_label.shape)

    K = args.top


    # Select nodes with similar confidence
    # max_confidence = torch.max(confidence)
    # thres_confidence = max_confidence - avg_ece_validation
    #
    # idx_to_label = torch.where(confidence >= thres_confidence)[0]
    #
    # # Select more nodes
    #
    # SAMPLES_RANGE = np.min( [ 2* K, K + 100 ] )
    # if idx_to_label.shape[0] <= SAMPLES_RANGE:
    #     _, idx_to_label = torch.topk(confidence, SAMPLES_RANGE)
    # else:
    #     SAMPLES_RANGE = idx_to_label.shape[0]

    if args.candidate_num == 0:
        if K < 100:
            SAMPLES_RANGE = 2 * K
        else:
            SAMPLES_RANGE = np.max((K + 100, K))
    else:
        SAMPLES_RANGE = args.candidate_num


    conf_score, idx_to_label = torch.topk(confidence, SAMPLES_RANGE)

    ####### Test

    # Each node is expected to be sampled t times
    N = int(SAMPLES_RANGE * t / K)
    print("Compute top ", K, "nodes from ",SAMPLES_RANGE, " nodes for Banzhaf, and sampling", N, " times...")

    # if FIX_NUM_SAMPLE:
    #     sampled_matrix = uniform_sample_N(idx_to_label, K, N)
    # else:
    #     sampled_matrix = uniform_sample(idx_to_label, K, N)

    sampled_matrix = uniform_sample(idx_to_label, K, N)

    # sampled_matrix = confidence_importance_sample(confidence, K, N)

    Utilities = torch.zeros(N).to(args.device)
    for n in range(N):
        idx_sampled = sampled_matrix[n]
        influence_matrix = influence_matrix.to(args.device)
        # sampled_labels = torch.multinomial(y_probs, num_samples=1).squeeze()
        # sampled_labels[idx_train] = labels[idx_train].copy()
        Utilities[n] = Entropy_utility(y_probs, idx_sampled, idx_train, idx_train_ag, idx_unlabeled, influence_matrix)

    # plot_tensor_distribution(Utilities)

    Values = torch.Tensor(output.shape[0])
    Values.fill_(float('-inf'))
    for node_i in idx_to_label:
        # if confidence[node_i] < args.threshold:
        #     print("Not confident")
        #     continue

        contain_mask = torch.tensor( [ (sampled_matrix[n] == node_i).any() for n in range(N) ] )
        U_mean_with = torch.mean(Utilities[contain_mask])
        U_mean_without = torch.mean(Utilities[~contain_mask])
        Values[node_i] = U_mean_with - U_mean_without

    # plot_tensor_distribution(Values)

    return Values


def get_IGP_idx_game(output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args, avg_ece_validation):
    """
    Select Nodes with Game-theoretical value calculation.
    - Utility function: IGP
    - Value calculation: 1) Banzhaf; 2) Shapely
    - Sampling: 1) Confidence Top K; 2) Random; 3) Importance Sampling
    """

    # TODO: Shapely
    # confidence, pred_label = get_confidence(output)
    Values = IGP_sample_banzhaf(output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence,
                                args, avg_ece_validation)

    # 1. TOP k
    Ent, pl_idx = torch.topk(Values, args.top)

    # 2. Positive
    #pl_idx = torch.where(Values > 0) [0]

    # print( torch.where(Values > 0) [0].shape )
    pl_confidence = confidence[pl_idx]
    if torch.min(pl_confidence) <= 0.0001:
        pl_idx = torch.where( confidence>0) [0]
        if pl_idx.shape[0] == 0:
            print("All labeled")
            return torch.tensor([])

    print("Adding", pl_idx.shape, " nodes to set ...")
    return pl_idx.clone().detach()


def top_k_indices(input_dict, k):
    # Extract items from the dictionary and sort them by value in descending order
    sorted_items = sorted(input_dict.items(), key=lambda item: item[1], reverse=True)

    # Extract the top k items and their indices (keys)
    top_k_indices = [item[0] for item in sorted_items[:k]]

    return top_k_indices

# def get_IGP_idx(output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args, avg_ece_validation):
#     influence_matrix = torch.from_numpy(influence_matrix).float()
#     # Step 0: Prepare probs of all nodes
#     y_probs = output.clone()
#     y_probs = torch.softmax(y_probs, dim=1)
#     # confidence, pred_label = torch.max(y_probs, dim=1)
#     num_class = y_probs.shape[1]
#     y_probs[idx_train] = convert_to_one_hot(labels[idx_train], num_class).float()
#
#     K = args.top
#
#     if args.candidate_num == 0:
#         if K < 100:
#             SAMPLES_RANGE = 2 * K
#         else:
#             SAMPLES_RANGE = np.max ( ( K + 100 , K ) )
#     else:
#         SAMPLES_RANGE = args.candidate_num
#
#
#     conf_score, idx_to_label = torch.topk(confidence, SAMPLES_RANGE)
#
#     Values = {}
#     for idx in idx_to_label:
#         Values[idx] = Entropy_utility(y_probs, idx, idx_train, idx_train_ag, idx_unlabeled, influence_matrix)
#
#     return torch.tensor(top_k_indices(Values, K), dtype=torch.int32)