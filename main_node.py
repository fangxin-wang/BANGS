import argparse

import numpy as np
import torch

from data import *
import sys
import logging
from banzhaf import *

from src.calibrator.calibrator import \
    TS, VS, ETS, CaGCN, GATS, IRM, SplineCalib, Dirichlet, OrderInvariantCalib

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="Cora", help='dataset for training')
parser.add_argument('--hidden', type=int, default=128,help='Number of hidden units.')
parser.add_argument("--hid_dim_1", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--view", type=int, default=5, help="Number of extra view of augmentation")

parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--epochs_ft', type=int, default=2000, help='Number of epochs to finetuning.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers.')
parser.add_argument('--nb_heads', type=int, default=8)
parser.add_argument('--nb_out_heads', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in training")
parser.add_argument("--aug_drop", type=float, default=0.1, help="Attribute augmentation dropout rate")
parser.add_argument('--lr', type=float, default=0.001,help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--beta', type=float, default=1,help='coefficient for weighted CE loss')

parser.add_argument("--threshold", type=float, default=0, help="Threshold for pseudo labeling")
parser.add_argument("--adaptive_threshold", action='store_true', default=False)
parser.add_argument("--iter", type=int, default=20, help="Number of pseudo labeling iteration")
parser.add_argument("--top", type=int, default=100, help="Number of pseudo label in each iteration")
parser.add_argument("--candidate_num", type=int, default=0, help="Number of candidate nodes")
parser.add_argument('--patience', type=int, default=500)
parser.add_argument('--multiview', action='store_true', default=False)
parser.add_argument("--random_pick", action="store_true", default=False, help="Indicator of random pseudo labeling")
parser.add_argument("--conf_pick", action="store_true", default=False, help="Indicator of CPL labeling")
parser.add_argument("--IGP_pick", action="store_true", default=False, help="Indicator of IGP labeling")

parser.add_argument("--seed", type=int, default=1024, help="Random seed")
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
parser.add_argument("--device", type=str, default='cpu', help="device of the model")
parser.add_argument("--noisy", type=float, default=0, help="Flip labels")


parser.add_argument("--train_portion", type = float, default = 0.05, help="Training Label Portion")
parser.add_argument("--valid_portion", type = float, default = 0.15, help="Validation Label Portion")

parser.add_argument('--split_by_num', action='store_true', default=False)
parser.add_argument("--train_num", type = int, default = 4, help="Training Label Num")
parser.add_argument("--valid_num", type = int, default = 1, help="Validation Label Num")

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

criterion = torch.nn.CrossEntropyLoss().cuda()
args.device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
device = args.device
print(device)




def train(args, model_path, idx_train, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, FT=False):

    nclass = labels.max().int().item() + 1
    # Model and optimizer
    model = get_models(args, features.shape[1], nclass, g=g)

    # Fine Tuning
    if not FT:
        epochs = args.epochs
    else:
        state_dict = torch.load('./save_model/tmp.pth')
        model.load_state_dict(state_dict)
        epochs = args.epochs_ft

    optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)

    best, bad_counter = 0, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        output = torch.softmax(output, dim=1)
        output = torch.mm(output, T)
        sign = False
        loss_train = weighted_cross_entropy(output[idx_train], pseudo_labels[idx_train], bald[idx_train], args.beta, nclass, sign)
        # loss_train = criterion(output[idx_train], pseudo_labels[idx_train])
        acc_train = accuracy(output[idx_train], pseudo_labels[idx_train])
        loss_train.backward()
        optimizer.step()


        with torch.no_grad():
            model.eval()
            output = model(features, adj)
            loss_val = criterion(output[idx_val], labels[idx_val])
            loss_test = criterion(output[idx_test], labels[idx_test])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])

        if not FT and epoch == 100:
            torch.save(model.state_dict(), './save_model/tmp.pth', _use_new_zipfile_serialization=False)
        if acc_val > best:
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
            best = acc_val
            bad_counter = 0
            best_output = output
            best_print = [f'epoch: {epoch}',
                          f'loss_train: {loss_train.item():.4f}',
                          f'acc_train: {acc_train:.4f}',
                          f'loss_val: {loss_val.item():.4f}',
                          f'acc_val: {acc_val:.4f}',
                          f'loss_test: {loss_test.item():4f}',
                          f'acc_test: {acc_test:.4f}']
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            # print('early stop')
            break

    print("best result: ", best_print)
    logger.info("best validation result: {:.4f}".format( best ) )
    return best, best_output


@torch.no_grad()
def test(adj, features, labels, idx_test, nclass, model_path, g, logger):
    nfeat = features.shape[1]
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g=g)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    logger.info("Test set results: loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(),acc_test))

    return acc_test, loss_test


from collections import Counter

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

        confidences_1_hop, confidences_2_hop = confidence [neighbors_1_hop], confidence [neighbors_2_hop]
        #print(confidences_1_hop)
        #print(confidences_2_hop)
        mean_confidence_1_hop = torch.nanmean(confidences_1_hop)
        mean_confidence_2_hop = torch.nanmean(confidences_2_hop)
        # print(f"  Mean Confidence 1-hop: {mean_confidence_1_hop}")
        # print(f"  Mean Confidence 2-hop: {mean_confidence_2_hop}")

        l_confidence_2_hop.append(mean_confidence_2_hop.item())

        label_frequency_1_hop = Counter(labels[neighbors_1_hop].tolist())
        label_frequency_2_hop = Counter(labels[neighbors_2_hop].tolist())
        total_labels = len(labels[neighbors_2_hop])
        # Calculate the entropy
        entropy = -sum((count / total_labels) * math.log2(count / total_labels) for count in label_frequency_2_hop.values())

        # print(f" Neighborhood Label 1-hop: {label_frequency_1_hop}")
        # print(f" Neighborhood Label 2-hop: {label_frequency_2_hop}, Entropy: {entropy}")
        l_entropy.append(entropy)

    logger.info(f"l_degree: {np.nanmean(l_degree) }, l_centrality: {np.nanmean(l_centrality)}, "
          f"l_confidence_2_hop: {np.nanmean(l_confidence_2_hop)}, l_entropy: {np.nanmean(l_entropy)}, "
          f"total neighbor num: {len(total_neighbor_set)}")


def get_adaptive_threshold(output, idx_train, global_thres, local_thres, decay = 0.9):
    # output = torch.softmax(output, dim=1)

    max_prob, argmax_pos = torch.max(output, dim = 1)

    global_thres_updated = decay * global_thres + (1-decay) * torch.mean(max_prob[~idx_train])
    local_thres_updated = decay * local_thres + (1-decay) * torch.mean(output[~idx_train], dim = 0)

    max_local_thres = torch.max(local_thres_updated)
    local_thres_final = local_thres_updated / max_local_thres * global_thres_updated

    mask = max_prob > local_thres_final[argmax_pos]

    return mask, global_thres_updated, local_thres_updated





if __name__ == '__main__':

    if not args.split_by_num:
        g, adj, features, labels, idx_train, idx_val, idx_test, oadj, pyg_graph = load_data(args.dataset, args.noisy,
                                                                                 args.train_portion, args.valid_portion, args.device, True)
    else:
        g, adj, features, labels, idx_train, idx_val, idx_test, oadj, pyg_graph = load_data(args.dataset, args.noisy,
                                                                                 args.train_num, args.valid_num, args.device, False)

    g = g.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    train_index = torch.where(idx_train)[0]
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    idx_pseudo = torch.zeros_like(idx_train)
    n_node = labels.size()[0]
    nclass = labels.max().int().item() + 1

    # args.top = int(n_node * 0.8 // 2000 * 100)
    if args.IGP_pick:
        model_path = './save_model/%s-%s-itr%s-top%s-seed%s-m%.0f-IGP.pth' % (
            args.model, args.dataset, args.iter, args.top, args.seed, args.multiview)
        log_path = './log/cautious-%s-%s-itr%s-top%s-seed%s-m%.0f-IGP.txt' % (
            args.model, args.dataset, args.iter, args.top, args.seed, args.multiview)
    if args.conf_pick:
        model_path = './save_model/%s-%s-itr%s-top%s-seed%s-m%.0f-Conf.pth' % (
            args.model, args.dataset, args.iter, args.top, args.seed, args.multiview)
        log_path = './log/cautious-%s-%s-itr%s-top%s-seed%s-m%.0f-Conf.txt' % (
            args.model, args.dataset, args.iter, args.top, args.seed, args.multiview)
    if args.random_pick:
        model_path = './save_model/%s-%s-itr%s-top%s-seed%s-m%.0f-Rand.pth' % (
            args.model, args.dataset, args.iter, args.top, args.seed, args.multiview)
        log_path = './log/cautious-%s-%s-itr%s-top%s-seed%s-m%.0f-Rand.txt' % (
            args.model, args.dataset, args.iter, args.top, args.seed, args.multiview)
    log_time_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(levelname)s %(asctime)s - %(message)s'
    log_time_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(
        format=log_format,
        datefmt=log_time_format,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    if args.random_pick:
        method = "random_pick"
    elif args.conf_pick:
        method = "conf_pick"
    elif args.IGP_pick:
        method = "IGP_pick"

    if args.conf_pick == True:
        logger.info(f"Cautious. model: {args.model}, dataset: {args.dataset}, Noisy_portion: {args.noisy}, N_node: {n_node}, "
                    f"N_class: {nclass}, Method: {method}, threshold: {args.threshold}, seed: {args.seed}")
    if args.IGP_pick == True:
        logger.info(f"IGP. model: {args.model}, dataset: {args.dataset}, Noisy_portion: {args.noisy}, N_node: {n_node}, "
                    f"N_class: {nclass}, Method: {method}, threshold: {args.threshold}, seed: {args.seed}")

    train_idx_num = int( torch.sum(idx_train).cpu().numpy() )
    valid_idx_num = int( torch.sum(idx_val).cpu().numpy() )
    test_idx_num = int( torch.sum(idx_test).cpu().numpy() )

    logger.info(f"idx_train: {train_idx_num}, idx_val: { valid_idx_num} , idx_test: { test_idx_num}")

    idx_train_ag = idx_train.clone().to(device)
    pseudo_labels = labels.clone().to(device)

    T = nn.Parameter(torch.eye(nclass, nclass).to(device)) # transition matrix
    T.requires_grad = False


    bald = torch.ones(n_node).to(device)
    best_valid, best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g)


    # global_thres = torch.Tensor([1/nclass])
    # local_thres = torch.full( (nclass,), 1/nclass)
    # print(f"Original global threshold: {global_thres}, class conditional threshold: {local_thres}")
    # global_thres, local_thres = global_thres.to(device), local_thres.to(device)

    acc_test0, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)


    # generate pseudo labels
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g=g)
    model.load_state_dict(state_dict)
    model.to(device)

    print("######")

    consistency = []
    PL_node = []
    tests = []
    pl_acc = []

    # calculate influence matrix list
    influence_matrix_list = get_influence_matrix(adj.cpu().to_dense(), k=2)

    val_acc_l, test_acc_l = [], []
    for itr in range(args.iter):

        if not args.multiview:

            print('Calibrating...')
            #temp_model = TS(model)
            temp_model = ETS(model, nclass)
            # cal_dropout_rate = 0.2
            # temp_model = CaGCN(model, n_node, nclass, 0.2)
            #print(temp_model.device)

            cal_wdecay = 5e-3
            temp_model.fit(pyg_graph, idx_val, idx_train, cal_wdecay)
            with torch.no_grad():
                temp_model.eval()
                logits = temp_model(pyg_graph.x, pyg_graph.adj)
                output_ave = F.softmax(logits, dim=1).detach()
                confidence, predict_labels = torch.max(output_ave, dim=1)


            consist = 1
            consistency.append(round(consist, 5))
        else:
            model.eval()
            output_ave, confidence, predict_labels, consist = multiview_pred(model, features, adj, g, args)
            consistency.append(round(consist,5))

        ece_validation = compute_ece(output_ave[idx_val], labels[idx_val])
        ece_test = compute_ece(output_ave[idx_test], labels[idx_test])
        logger.info(f"ECE of validation data : {ece_validation}, ECE of test data: {ece_test}")

        # Cannot choose nodes already augmented or labeled
        confidence[idx_train_ag] = 0


        if args.adaptive_threshold:
            mask, global_thres_updated, local_thres_updated = get_adaptive_threshold(output_ave, idx_train, global_thres, local_thres)
            confidence *= mask
            global_thres, local_thres = global_thres_updated, local_thres_updated
            print(f"Current global threshold: {global_thres.cpu().numpy()}, class conditional threshold: {local_thres.cpu().numpy()}")


        if args.random_pick:
            print("===random_pick===")
            idxes = torch.where(confidence > 0.)[0]
            pl_idx = random.sample(idxes.tolist(), args.top)
            pl_idx = torch.Tensor(pl_idx).long().to(device)
            conf_score = confidence[pl_idx]

        if args.conf_pick:
            print("===conf_pick===")
            pl_idx = torch.where(confidence > args.threshold)[0]
            conf_score = confidence[pl_idx]
            if len(conf_score) >= args.top:
                conf_score, pl_idx = torch.topk(confidence, args.top)

        if args.IGP_pick:
            print("===IGP_pick===")
            # TODO: Check any label leakage
            # we can pseudo label any nodes except with fixed labels
            idx_unlabeled = ~idx_train_ag

            #idx_unlabeled = ~(idx_train_ag | idx_test | idx_val)
            #idx_unlabeled = torch.where( idx_unlabeled == True) [0]

            # pl_idx = get_IGP_idx(output_ave, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args)
            avg_ece_validation = ece_validation/ 20

            # label rate
            beta = ( n_node - len(PL_node) - args.top) / n_node

            influence_matrix = beta * influence_matrix_list[0] + beta * beta * influence_matrix_list[1]

            pl_idx = get_IGP_idx_game(output_ave, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args, avg_ece_validation)
            #pl_idx = get_IGP_idx(output_ave, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args, avg_ece_validation)

        #print_node_attributes(g, labels, pl_idx, idx_train, idx_train_ag, confidence,logger)

        if len(pl_idx) > 0:
            conf_score = confidence[pl_idx]
            print(conf_score)

            pl_labels = predict_labels[pl_idx]
            idx_train_ag[pl_idx] = True
            pred_diff =  pseudo_labels[pl_idx] == pl_labels
            pred_diff = len(pred_diff[pred_diff==True])/len(pred_diff)

            pseudo_labels[pl_idx] = pl_labels
            PL_node += list(pl_idx.cpu().numpy())         
            pred_diff_sum =  pseudo_labels[idx_train_ag^idx_train] == labels[idx_train_ag^idx_train]
            pred_diff_sum = len(pred_diff_sum[pred_diff_sum==True])/len(pred_diff_sum)
            pl_acc.append(pred_diff_sum)

            #best_valid, best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, FT=True)

            best_valid, best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj,
                                                pseudo_labels, labels, bald, T, g)


            # Testing
            acc_test, _ = test(adj, features, labels, idx_test, nclass, model_path, g, logger)
            val_acc_l.append(best_valid)
            test_acc_l.append(acc_test)

            ### MODIFY
            conf_test, _ = torch.max(output_ave[idx_test], dim=1)

            # plt.hist(conf_test, bins=30, alpha=0.5)
            # plt.title('Histogram of all confidence values')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.show()

            conf_avg_test = torch.nanmean( conf_test )

            ### MODIFY
            tests.append(acc_test)

            logger.info('itr {} summary: added {} pl labels with confidence {:.5f}, pl_acc: {}, consistency {:.5f}, {} pl labels in total, test_acc: {:.4f}, test_conf: {:.4f}'.format(
                itr, len(pl_idx), conf_score.min().item(), pred_diff*100, consist, len(PL_node), acc_test, conf_avg_test) )
        else:
            break
    logger.info('original acc: {:.5f}, best test accuracy: {:.5f}, final test accuracy: {:.5f}, consistency: {}, pl_acc: {}'.format(
        acc_test0, max(tests), acc_test, consistency[np.argmax(tests)], pl_acc[np.argmax(tests)]))
    logger.info('Best Acc Early Stopped by Valid Acc: {:.5f}'.format(  test_acc_l[np.argmax(val_acc_l)]) )
    print("ENDS")