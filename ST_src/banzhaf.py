import torch
import torch.nn.functional as F
import numpy as np
from  ST_src.utils_new import get_sparse_column, compute_ppr_matrix_parallel
from dgl.nn import APPNPConv
from ST_src.utils import get_confidence

def uniform_sample(indices, K, N):
    if K > len(indices):
        raise ValueError("K cannot be greater than the number of indices available")

    result_matrix = [indices[torch.where(torch.rand(len(indices)) < 0.5)[0]] for _ in range(N)]
    return result_matrix


def uniform_sample_less_K(indices, K, N):
    if K > len(indices):
        raise ValueError("K cannot be greater than the number of indices available")

    result_matrix = []
    for _ in range(N):
        while True:
            # Sample indices with a probability of 0.5
            selected_indices = indices[torch.where(torch.rand(len(indices)) < 0.5)[0]]

            # Check if the sampled indices are at most K in size
            if len(selected_indices) <= K:
                result_matrix.append(selected_indices)
                break

    return result_matrix

def convert_to_one_hot(labels, num_class):
    """
    Convert numerical labels to one-hot encoded labels.
    """
    # labels = labels.to(torch.int64)
    one_hot_labels = F.one_hot(labels, num_classes=num_class)

    return one_hot_labels

def probs_convert_to_one_hot_tensor(prob_tensor):
    _, max_indices = torch.max(prob_tensor, dim=1)

    # Convert these indices to a one-hot encoded tensor
    one_hot_tensor = F.one_hot(max_indices, num_classes=prob_tensor.shape[1])

    return one_hot_tensor.float()


def Entropy_utility_efficient(y_probs, idx_sampled, idx_train, idx_train_ag, idx_poss_ag,
                              influence_matrix_candidate, total_prob_origin, influence_submatrix_train):
    device = influence_submatrix_train.device

    if idx_sampled.dim() == 0:
        idx_sampled = idx_sampled.unsqueeze(0)

    sample_prob = torch.sparse.mm(influence_matrix_candidate, y_probs[idx_sampled])

    final_prob = total_prob_origin + sample_prob
    final_prob = torch.softmax(final_prob, dim=1)
    # print("final_prob", final_prob)

    class_prob = torch.mean(final_prob, dim = 0)
    class_Ent = -torch.sum(  class_prob * torch.log(class_prob + 1e-9)  )

    individual_entropies = -torch.sum(final_prob * torch.log(final_prob + 1e-9), dim=1)
    individual_Ent  = torch.mean(individual_entropies)

    ### STEP 2
    idx_pseudo_ag = torch.cat((idx_train_ag, idx_sampled), dim=0)
    train_prob_prop = torch.sparse.mm(influence_submatrix_train, y_probs[idx_pseudo_ag])
    Prob_train_prop = torch.softmax( train_prob_prop, dim=1)
    y_train_pred = torch.softmax(y_probs[idx_train], dim=1)
    # print("train_prob_prop", train_prob_prop)
    # print("y_train_pred", y_train_pred)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    Cross_Ent = criterion(Prob_train_prop, y_train_pred)

    # print('class_Ent', class_Ent.item() , 'individual_Ent', individual_Ent.item(), 'Cross_Ent', Cross_Ent.item())

    return class_Ent - individual_Ent - Cross_Ent
    # return - individual_Ent
    #return - Cross_Ent - individual_Ent


def IGP_sample_banzhaf(g, output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args,
                       avg_ece_validation,
                       N=100):
    """
    Maximal Sample Reuse:
    - Sampling Budget : N
    """
    t = args.sample_num

    # Step 0: Prepare probs of all nodes
    y_probs = output.clone()

    # print('y_probs', torch.max(y_probs), torch.min(y_probs))
    # y_probs = torch.softmax(y_probs, dim=1)
    # confidence, pred_label = torch.max(y_probs, dim=1)
    # # 9.12 test
    num_class = y_probs.shape[1]
    y_probs[idx_train] = convert_to_one_hot(labels[idx_train], num_class).float()
    # print( 'y_probs', torch.max(y_probs), torch.min(y_probs))

    K = args.top

    if args.candidate_num == 0:
        if K < 100:
            SAMPLES_RANGE = 2 * K
        else:
            SAMPLES_RANGE = np.max((K + 100, K))
    else:
        SAMPLES_RANGE = args.candidate_num

    conf_score, idx_to_label = torch.topk(confidence, SAMPLES_RANGE)

    # Each node is expected to be sampled t times
    N = int(SAMPLES_RANGE * t / K)
    print("Compute top ", K, "nodes from ", SAMPLES_RANGE, " nodes for Banzhaf, and sampling", N, " times...")

    if not args.k_union:
        sampled_matrix = uniform_sample(idx_to_label, K, N)
    else:
        print("Only sample unions with less than k members.")
        sampled_matrix = uniform_sample_less_K(idx_to_label, K, N)
    Utilities = torch.zeros(N).to(args.device)

    # 1. For efficiency, calculate the probs of ALL POSSIBLY INFLUENCED nodes influenced by ONLY original training nodes
    if idx_to_label.dim() == 0:
        idx_poss = idx_to_label.unsqueeze(0)
    else:
        idx_poss = idx_to_label
    idx_train_ag = torch.where(idx_train_ag == True)[0]
    idx_poss_ag = torch.cat((idx_train_ag, idx_poss), dim=0)
    # influenced_mask = torch.zeros(influence_matrix.size(0), dtype=torch.bool).to(args.device)
    # for idx in idx_poss_ag:
    #     column_tensor = get_sparse_column(influence_matrix, idx)
    #     influenced_mask |= (column_tensor.to_dense().squeeze() > 0)
    # influenced_nodes = torch.where(influenced_mask & idx_unlabeled)[0]

    # Let rows of influence_matrix sum up to 1
    # print( 'row',torch.sum(influence_matrix,dim=0), ' col',torch.sum(influence_matrix,dim=1))
    influence_matrix = torch.transpose(influence_matrix, 0, 1)
    idx_train_value = torch.where(idx_train == True)[0]

    # 2. how train nodes affect the probs of influenced nodes
    # y_probs_train = y_probs.clone()
    # y_probs_train[~idx_train] = 0

    influence_submatrix = influence_matrix.index_select(1, idx_train_value)
    total_prob_origin = torch.sparse.mm(influence_submatrix, y_probs[idx_train])

    # 3. how all PL/labeled nodes affect GT nodes
    influence_matrix_train = influence_matrix.index_select(0, idx_train_value)
    # influence_submatrix_train: [idx_train * idx_poss_ag]

    #influence_matrix_candidate = influence_matrix.index_select(1, idx_poss)
    # Calculate Utility
    for n in range(N):
        idx_sampled = sampled_matrix[n]
        influence_matrix_candidate = influence_matrix.index_select(1, idx_sampled)
        idx_sampled_ag = torch.cat((idx_train_ag, idx_sampled), dim=0)
        influence_submatrix_train = influence_matrix_train.index_select(1, idx_sampled_ag)
        Utilities[n] = Entropy_utility_efficient(y_probs, idx_sampled, idx_train, idx_train_ag, idx_poss_ag,
                                                 influence_matrix_candidate, total_prob_origin, influence_submatrix_train)

    Values = get_Banzhaf(output,idx_to_label,sampled_matrix,Utilities,N)
    # plot_tensor_distribution(Values)
    return Values

def get_Banzhaf(output, idx_to_label, sampled_matrix, Utilities, N):
    Values = torch.Tensor(output.shape[0])
    Values.fill_(float('-inf'))
    for node_i in idx_to_label:
        # if confidence[node_i] < args.threshold:
        #     print("Not confident")
        #     continue

        contain_mask = torch.tensor([(sampled_matrix[n] == node_i).any() for n in range(N)])
        #print(contain_mask)
        U_mean_with = torch.mean(Utilities[contain_mask])
        U_mean_without = torch.mean(Utilities[~contain_mask])
        Values[node_i] = U_mean_with - U_mean_without

    return Values


def IGP_sample_banzhaf_conv(g, output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix,
                                confidence,
                                args, avg_ece_validation):
    t = args.sample_num

    K = args.top

    if args.candidate_num == 0:
        if K < 100:
            SAMPLES_RANGE = 2 * K
        else:
            SAMPLES_RANGE = np.max((K + 100, K))
    else:
        SAMPLES_RANGE = args.candidate_num

    conf_score, idx_to_label = torch.topk(confidence, SAMPLES_RANGE)

    # Each node is expected to be sampled t times
    N = int(SAMPLES_RANGE * t / K)
    print("Compute top ", K, "nodes from ", SAMPLES_RANGE, " nodes for Banzhaf, and sampling", N, " times...")

    if not args.k_union:
        sampled_matrix = uniform_sample(idx_to_label, K, N)
    else:
        print("Only sample unions with less than k members.")
        sampled_matrix = uniform_sample_less_K(idx_to_label, K, N)

    Utilities = torch.zeros(N).to(args.device)

    # For efficiency, calculate the probs of ALL POSSIBLY INFLUENCED nodes influenced by ONLY original training nodes
    if idx_to_label.dim() == 0:
        idx_poss = idx_to_label.unsqueeze(0)
    else:
        idx_poss = idx_to_label

    idx_train_ag = torch.where(idx_train_ag == True)[0]
    idx_train_value = torch.where(idx_train == True)[0]

    conv = APPNPConv( k = args.batchPPR, alpha=0.15)

    logits_train = output.clone()
    # features = g.ndata['feat']

    # ########
    confidence, pred_label = get_confidence(logits_train,True)
    num_class = output.shape[1]
    logits_train[idx_train_ag] = convert_to_one_hot(pred_label[idx_train_ag], num_class).float()
    # ########

    logits_train[~idx_train_value] = 0
    total_prob_origin = conv(g, logits_train) # only propagate from train nodes

    for n in range(N):
        idx_sampled = sampled_matrix[n]
        logits_sampled = output.clone()
        logits_sampled[~idx_sampled] = 0

        prob_sampled = conv(g, logits_sampled)
        final_prob = torch.softmax(total_prob_origin + prob_sampled, dim=1)
        final_prob_unlabeled = final_prob[~idx_train_ag]

        # #############
        # logits_train[idx_sampled] = convert_to_one_hot(pred_label[idx_sampled], num_class).float()
        # #print(logits_train)
        # final_prob = torch.softmax( conv(g, logits_train)  , dim=1)
        # final_prob_unlabeled = final_prob[~idx_train_ag]
        if n==0:
            _, final_pred_label = torch.max(final_prob_unlabeled, dim =1)
            # print('pred in teacher model',pred_label[~idx_train_ag])
            # print('estimated pred for student model', final_pred_label)
            print('portion of same label', torch.sum(pred_label[~idx_train_ag] == final_pred_label) / final_pred_label.size(0) )
        #############

        class_prob = torch.sum(final_prob_unlabeled, dim=0)
        class_Ent = -torch.sum(class_prob * torch.log(class_prob + 1e-9))

        individual_entropies = -torch.sum(final_prob_unlabeled * torch.log(final_prob_unlabeled + 1e-9), dim=1)
        individual_Ent = torch.sum(individual_entropies)


        final_prob_train = final_prob[idx_train_value]
        y_train_pred = torch.softmax(output[idx_train], dim=1)

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        Cross_Ent = criterion(final_prob_train, y_train_pred)

        Utilities[n] = class_Ent - individual_Ent - Cross_Ent
        #Utilities[n] = - individual_Ent - Cross_Ent
        # print('class_Ent', class_Ent , 'individual_Ent', individual_Ent)

    Values = get_Banzhaf(output, idx_to_label, sampled_matrix, Utilities, N)

    return Values


def get_IGP_idx_game(g, output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args,
                     avg_ece_validation):
    """
    Select Nodes with Game-theoretical value calculation.
    """

    # confidence, pred_label = get_confidence(output)

    if args.PageRank:
        Values = IGP_sample_banzhaf_conv(g, output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix,
                                    confidence,
                                    args, avg_ece_validation)
    else:
        Values = IGP_sample_banzhaf(g, output, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix,
                                    confidence,
                                    args, avg_ece_validation)

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

