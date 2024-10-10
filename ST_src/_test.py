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