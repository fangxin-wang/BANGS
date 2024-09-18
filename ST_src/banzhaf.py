import torch
import torch.nn.functional as F
import numpy as np
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


# def confidence_importance_sample(confidence, K, N, temperature=1):
#     # Normalize the confidences to ensure they sum to 1 (if not already probabilities)
#     # Lower temperature -> sharper distribution
#     confidences_normalized = torch.softmax(confidence / temperature, dim=0)
#
#     # Sample K indices N times based on the probabilities
#     sampled_indices = torch.multinomial(confidences_normalized, num_samples=K * N, replacement=True)
#
#     # Reshape the flat list of indices into an N x K matrix
#     sampled_indices_matrix = sampled_indices.reshape(N, K)
#
#     return sampled_indices_matrix


def convert_to_one_hot(labels, num_class):
    """
    Convert numerical labels to one-hot encoded labels.
    """
    # labels = labels.to(torch.int64)
    one_hot_labels = F.one_hot(labels, num_classes=num_class)

    return one_hot_labels


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


# def Entropy_utility(y_probs, idx_sampled, idx_train, idx_train_ag, idx_unlabeled, influence_matrix):
#     """
#     y_probs: idx_train -> one hot
#     """
#     if idx_sampled.dim() == 0:
#         idx_sampled = idx_sampled.unsqueeze(0)
#
#     idx_train_ag,idx_train =  torch.where( idx_train_ag == True) [0], torch.where( idx_train == True) [0]
#     idx_pseudo_ag = torch.cat((idx_train_ag, idx_sampled), dim=0)
#     influenced_nodes = torch.where(torch.any(influence_matrix[:, idx_pseudo_ag] > 0, axis=1) & idx_unlabeled )[0]
#
#     # assume the pseudo labels are fixed
#     # y_probs[idx_pseudo_ag] = probs_convert_to_one_hot_tensor(y_probs[idx_pseudo_ag])
#
#     # Information propogated to currently unlabeled (including pl) nodes from labeled and pl nodes
#     total_Ent = 0
#     for node_j in influenced_nodes:
#         # node j influenced by all pl & labeled nodes
#         total_prob_origin = F.normalize(torch.matmul(influence_matrix[node_j, idx_pseudo_ag], y_probs[idx_pseudo_ag]),
#                                         dim=0)
#         entropy_origin = calculate_entropy(total_prob_origin)
#         total_Ent += entropy_origin
#
#
#     #return total_Ent
#
#     #Cross entropy loss between GT label and distribution propogated by pls
#
#     influence_matrix_train = influence_matrix[idx_train,:]
#
#     Prob_train_prop = F.normalize(torch.matmul(influence_matrix_train[:, idx_pseudo_ag], y_probs[idx_pseudo_ag]),
#                                         dim=0)
#
#     criterion = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
#     Cross_Ent = criterion( Prob_train_prop, y_probs[idx_train] )
#     #print( "Total_Ent ", total_Ent, " Cross_Ent", Cross_Ent)
#
#     return total_Ent -  Cross_Ent

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


def Entropy_utility_efficient(y_probs, idx_sampled, idx_train, idx_train_ag, idx_poss_ag,
                              influence_submatrix, total_prob_origin, influence_submatrix_train):
    device = influence_submatrix.device
    # total_prob_origin
    # probs influenced by training nodes

    if idx_sampled.dim() == 0:
        idx_sampled = idx_sampled.unsqueeze(0)

    ### STEP 1
    # sample_prob
    sample_mask = torch.zeros(y_probs.shape[0], dtype=torch.bool)
    sample_mask[idx_sampled] = True
    y_probs_copy = y_probs.clone()
    y_probs_copy[~sample_mask] = 0

    # influence_submatrix: influenced_nodes * idx_poss_ag
    sample_prob = torch.sparse.mm(influence_submatrix, y_probs_copy[idx_poss_ag])

    final_prob = F.normalize(total_prob_origin + sample_prob, dim=1)
    ######## 8.20
    final_prob = torch.softmax(final_prob, dim=1)
    ########
    class_prob = torch.mean(final_prob, dim = 0)
    class_Ent = -torch.sum(  class_prob * torch.log(class_prob + 1e-9)  )

    individual_entropies = -torch.sum(final_prob * torch.log(final_prob + 1e-9), dim=1)
    individual_Ent  = torch.mean(individual_entropies)

    # individual_Ent = individual_entropies.sum()
    #print('individual_Ent', individual_Ent.item() )

    ### STEP 2

    # influence_submatrix_train = idx_train_value * idx_poss_ag

    # idx_pseudo_ag = torch.cat((idx_train_ag, idx_sampled), dim=0)
    Prob_train_prop = F.normalize(
        torch.sparse.mm(influence_submatrix_train, y_probs_copy[idx_poss_ag]), dim=1)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    Cross_Ent = criterion(Prob_train_prop, y_probs[idx_train])

    #print('class_Ent', class_Ent.item() , 'individual_Ent', individual_Ent.item(), 'Cross_Ent', Cross_Ent.item())
    return class_Ent - individual_Ent - Cross_Ent


    #return class_Ent - total_Ent - Cross_Ent

    # return - total_Ent


def IGP_sample_banzhaf(adj, output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args,
                       avg_ece_validation,
                       N=100, FIX_NUM_SAMPLE=False):
    """
    Maximal Sample Reuse:
    - Sampling Budget : N
    """
    t = 100

    # Step 0: Prepare probs of all nodes
    y_probs = output.clone()

    # print('y_probs', torch.max(y_probs), torch.min(y_probs))
    # y_probs = torch.softmax(y_probs, dim=1)
    # confidence, pred_label = torch.max(y_probs, dim=1)

    num_class = y_probs.shape[1]
    # 9.12 test
    # y_probs[idx_train] = convert_to_one_hot(labels[idx_train], num_class).float()
    # print( 'y_probs', torch.max(y_probs), torch.min(y_probs))

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
    print("Compute top ", K, "nodes from ", SAMPLES_RANGE, " nodes for Banzhaf, and sampling", N, " times...")

    # if FIX_NUM_SAMPLE:
    #     sampled_matrix = uniform_sample_N(idx_to_label, K, N)
    # else:
    #     sampled_matrix = uniform_sample(idx_to_label, K, N)

    # For efficiency, we only care about nodes to label and its neighborhood nodes
    # candidate_neighbor_idx = get_k_hop_neighbors( adj, idx_to_label, 3)
    # print(candidate_neighbor_idx.shape, ' 3-hop neighbors of', idx_to_label.shape, ' nodes.')
    # mask = torch.zeros_like(idx_train_ag, dtype=torch.bool)
    # mask[candidate_neighbor_idx] = True
    #
    # # did not revise idx_train: not used in this version
    # print("Before: ", torch.sum(idx_train_ag).item() , ' out of ', idx_train_ag.shape, ' nodes')
    # idx_train_ag = idx_train_ag & mask
    # print("After: ", torch.sum(idx_train_ag).item() )

    sampled_matrix = uniform_sample(idx_to_label, K, N)
    # sampled_matrix = confidence_importance_sample(confidence, K, N)
    Utilities = torch.zeros(N).to(args.device)

    EFFICIENT = True
    # 08/01: For efficiency, calculate the probs of ALL POSSIBLY INFLUENCED nodes influenced by ONLY original training nodes
    if EFFICIENT:

        if idx_to_label.dim() == 0:
            idx_poss = idx_to_label.unsqueeze(0)
        else:
            idx_poss = idx_to_label

        idx_train_ag = torch.where(idx_train_ag == True)[0]
        idx_poss_ag = torch.cat((idx_train_ag, idx_poss), dim=0)
        influenced_mask = torch.zeros(influence_matrix.size(0), dtype=torch.bool).to(args.device)
        for idx in idx_poss_ag:
            column_tensor = get_sparse_column(influence_matrix, idx)
            influenced_mask |= (column_tensor.to_dense().squeeze() > 0)
        influenced_nodes = torch.where(influenced_mask & idx_unlabeled)[0]

        # ONLY take probs of original training nodes
        y_probs_mask = ~idx_train
        y_probs_copy = y_probs.clone()
        y_probs_copy[y_probs_mask] = 0

        influence_matrix_influenced = influence_matrix.index_select(0, influenced_nodes)
        influence_submatrix = influence_matrix_influenced.index_select(1, idx_poss_ag)
        # print(influence_matrix_influenced.size(), influence_submatrix.size())

        idx_train_value = torch.where(idx_train == True)[0]
        influence_matrix_train = influence_matrix.index_select(0, idx_train_value)
        influence_submatrix_train = influence_matrix_train.index_select(1, idx_poss_ag)

        total_prob_origin = torch.sparse.mm(influence_submatrix, y_probs_copy[idx_poss_ag])
        for n in range(N):
            idx_sampled = sampled_matrix[n]
            Utilities[n] = Entropy_utility_efficient(y_probs, idx_sampled, idx_train, idx_train_ag, idx_poss_ag,
                                                     influence_submatrix, total_prob_origin, influence_submatrix_train)
    else:

        for n in range(N):
            idx_sampled = sampled_matrix[n]
            Utilities[n] = Entropy_utility(y_probs, idx_sampled, idx_train, idx_train_ag, idx_unlabeled,
                                           influence_matrix)

    # plot_tensor_distribution(Utilities)

    Values = torch.Tensor(output.shape[0])
    Values.fill_(float('-inf'))
    for node_i in idx_to_label:
        # if confidence[node_i] < args.threshold:
        #     print("Not confident")
        #     continue

        contain_mask = torch.tensor([(sampled_matrix[n] == node_i).any() for n in range(N)])
        U_mean_with = torch.mean(Utilities[contain_mask])
        U_mean_without = torch.mean(Utilities[~contain_mask])
        Values[node_i] = U_mean_with - U_mean_without

    # plot_tensor_distribution(Values)

    return Values


def get_IGP_idx_game(adj, output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args,
                     avg_ece_validation):
    """
    Select Nodes with Game-theoretical value calculation.
    - Utility function: IGP
    - Value calculation: 1) Banzhaf; 2) Shapely
    - Sampling: 1) Confidence Top K; 2) Random
    """

    # TODO: Shapely
    # confidence, pred_label = get_confidence(output)
    Values = IGP_sample_banzhaf(adj, output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix,
                                confidence,
                                args, avg_ece_validation)

    # 1. TOP k
    Ent, pl_idx = torch.topk(Values, args.top)

    # print( torch.where(Values > 0) [0].shape )
    pl_confidence = confidence[pl_idx]
    if torch.min(pl_confidence) <= 0.0001:
        pl_idx = torch.where(confidence > 0)[0]
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
