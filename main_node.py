import argparse
import logging
import random
import numpy as np
import sys
import time

import torch
import torch.optim as optim
import torch.nn as nn
from dgl import PPR

from ST_src.banzhaf import *
from ST_src.data import *
from ST_src.models import *
from ST_src.utils import *
from ST_src.utils_new import *
from ST_src.train_test import *
from calib_src.calibrator.calibrator import ETS, CaGCN

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="Cora", help='dataset for training')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument("--hid_dim_1", type=int, default=32, help="Hidden layer dimension")
parser.add_argument("--view", type=int, default=5, help="Number of extra view of augmentation")

######### test run #########
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
#########

parser.add_argument('--epochs_ft', type=int, default=1000, help='Number of epochs to finetuning.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers.')
parser.add_argument('--nb_heads', type=int, default=8)
parser.add_argument('--nb_out_heads', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in training")
parser.add_argument("--aug_drop", type=float, default=0.1, help="Attribute augmentation dropout rate")
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--beta', type=float, default=1, help='coefficient for weighted CE loss')
parser.add_argument("--FT", action='store_true', default=False, help="Fine tuning during self-training")


parser.add_argument("--threshold", type=float, default=0, help="Threshold for pseudo labeling")
parser.add_argument("--adaptive_threshold", action='store_true', default=False)
parser.add_argument("--iter", type=int, default=20, help="Number of pseudo labeling iteration")
parser.add_argument("--top", type=int, default=100, help="Number of pseudo label in each iteration")
parser.add_argument("--candidate_num", type=int, default=0, help="Number of candidate nodes. TOP + 100 if not specified.")

parser.add_argument('--patience', type=int, default=500)
parser.add_argument('--multiview', action='store_true', default=False)
parser.add_argument('--calib', action='store_true', default=False)
parser.add_argument("--random_pick", action="store_true", default=False, help="Indicator of random pseudo labeling")
parser.add_argument("--conf_pick", action="store_true", default=False, help="Indicator of CPL labeling")
parser.add_argument("--IGP_pick", action="store_true", default=False, help="Indicator of IGP labeling")
parser.add_argument('--sample_num', type=int, default=100)
parser.add_argument("--k_union", action="store_true", default=False, help="Only sample unions with less than k members.")


parser.add_argument("--seed", type=int, default=1024, help="Random seed")
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
parser.add_argument("--device", type=str, default='cpu', help="device of the model")
parser.add_argument("--noisy", type=float, default=0, help="Flip labels")
parser.add_argument("--PageRank", action="store_true", default=False, help="Calculate Influence Matrix by PageRank")
parser.add_argument("--batchPPR", type=int, default=0, help="Batch PPR iteration number.")


parser.add_argument("--train_portion", type=float, default=0.05, help="Training Label Portion")
parser.add_argument("--valid_portion", type=float, default=0.15, help="Validation Label Portion")

parser.add_argument('--split_by_num', action='store_true', default=False)
parser.add_argument("--train_num", type=int, default=15, help="Training Label Num")
parser.add_argument("--valid_num", type=int, default=5, help="Validation Label Num")

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)


args.device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
device = args.device
print(device, args.seed)

start_time = time.time()


if __name__ == '__main__':

    IF_PORTION = not args.split_by_num
    IF_CALIB = args.calib

    if not IF_PORTION:
        train_split, valid_split = args.train_num, args.valid_num
    else:
        train_split, valid_split = args.train_portion, args.valid_portion

    g, adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset,
                                                                       args.device, args.seed,
                                                                       args.noisy,
                                                                       train_split, valid_split,
                                                                       IF_PORTION, IF_CALIB)
    print('Load!')

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

    # Save log and model

    if args.random_pick:
        method = "random"
    elif args.conf_pick:
        method = "conf"
    elif args.IGP_pick:
        method = "IGP"

    model_path = './save_model/%s-%s-itr%s-top%s-seed%s-m%.0f-%s-ft%s-ppr%s.pth' % (
        args.model, args.dataset, args.iter, args.top, args.seed, args.multiview, method, args.FT, args.PageRank)
    log_path = './final_log/cautious-%s-%s-itr%s-top%s-seed%s-m%.0f-%s-ft%s-ppr%s.txt' % (
        args.model, args.dataset, args.iter, args.top, args.seed, args.multiview, method, args.FT, args.PageRank)

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

    logger.info(f"model: {args.model}, dataset: {args.dataset}, Noisy_portion: {args.noisy}, N_node: {n_node}, "
                f"N_class: {nclass}, Method: {method}, threshold: {args.threshold}, seed: {args.seed}")

    train_idx_num = int(torch.sum(idx_train).cpu().numpy())
    valid_idx_num = int(torch.sum(idx_val).cpu().numpy())
    test_idx_num = int(torch.sum(idx_test).cpu().numpy())

    logger.info(f"idx_train: {train_idx_num}, idx_val: {valid_idx_num} , idx_test: {test_idx_num}")
    logger.info(f"Sample Num: {args.sample_num}")

    idx_train_ag = idx_train.clone().to(device)
    pseudo_labels = labels.clone().to(device)

    T = nn.Parameter(torch.eye(nclass, nclass).to(device))  # transition matrix
    T.requires_grad = False

    # Compute Influence Matrix
    if args.IGP_pick:
        if args.PageRank:
            logger.info("PPR")
            # When dataset is small
            if not args.batchPPR:
                influence_matrix = get_ppr_influence_matrix(adj)
                # transform = PPR(avg_degree=2)
                # N = g.number_of_nodes()
                # new_g = transform(g)
                # values = new_g.edata['w']
                # row, col = new_g.adj_tensors('coo')
                # influence_matrix = torch.sparse_coo_tensor(
                #     torch.stack([row, col]),
                #     values,
                #     (N, N),
                #     device=device
                # )
            else:
                print("####test####: PPR using APPNPConv")
                influence_matrix = None
                # g.ndata['w'] = caluclate_W(g, device)
                # influence_matrix = compute_ppr_matrix_parallel(g, g.ndata['W'])

        else:
            logger.info("IM 2 hop")
            influence_matrix_list = get_influence_matrix(adj, k=2)

    # Train initial model
    bald = torch.ones(n_node).to(device)
    best_valid, best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels,
                                    labels, bald, T, g, logger)

    ACC_LIST, ENT_LIST = [], []
    acc_test0, _, output_prev = test(args, adj, features, labels, idx_test, nclass, model_path, g, logger)
    ACC_LIST.append(acc_test0)

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

    log_time_of_step(start_time,logging)
    # calculate influence matrix list
    # influence_matrix_list = get_influence_matrix(adj.cpu().to_dense(), k=2)

    val_acc_l, test_acc_l = [], []

    T_conf, T_select, T_retrain = [], [], []

    for itr in range(args.iter):
        t0 = time.time()
        ##### 1. CONFIDENCE ########

        if args.calib:

            if args.dataset == 'Pubmed': #Pubmed
                cal_dropout_rate = 0.2
                temp_model = CaGCN(model, n_node, nclass, cal_dropout_rate)
                print('Calibrating with CaGCN...')
            else:
                temp_model = ETS(model, nclass)
                print('Calibrating with ETS...')

            cal_wdecay = 5e-3
            temp_model.fit(features, adj, labels, idx_val, idx_train, cal_wdecay)
            with torch.no_grad():
                temp_model.eval()
                output_ave = temp_model(features, adj)
                # print( output_ave.shape )

                confidence, predict_labels = get_confidence(output_ave)
                # print(torch.max(confidence), confidence[:5] )

                # # Normalize logits of psuedo label nodes
                logits_norm = get_norm_logit(output_ave)
                output_ave = logits_norm


            # output_ave = torch.softmax(logits, dim=1).detach()
            # ece_validation = compute_ece(output_ave[idx_val], labels[idx_val])
            # ece_test = compute_ece(output_ave[idx_test], labels[idx_test])
            # logger.info(f"ECE of validation data : {ece_validation}, ECE of test data: {ece_test}")

            consist = 1
            consistency.append(round(consist, 5))

        elif args.multiview:
            print('multi view')
            model.eval()
            # best_output, scores, pl_labels
            output_ave, confidence, predict_labels, consist = multiview_pred(model, features, adj, g, args)
            consistency.append(round(consist, 5))
        else:
            print('one view')
            output_ave = model(features, adj)
            confidence, predict_labels = get_confidence(output_ave)
            consist = 1
            # avg_ece_validation = 20 ## not computed
            consistency.append(round(consist, 5))

        # Cannot choose nodes already augmented or labeled
        confidence[idx_train_ag] = 0

        t1 = time.time()
        T_conf.append(t1 - t0)

        ##### 2. SELECTION ########

        # if args.adaptive_threshold:
        #     mask, global_thres_updated, local_thres_updated = get_adaptive_threshold(output_ave, idx_train, global_thres, local_thres)
        #     confidence *= mask
        #     global_thres, local_thres = global_thres_updated, local_thres_updated
        #     print(f"Current global threshold: {global_thres.cpu().numpy()}, class conditional threshold: {local_thres.cpu().numpy()}")

        if args.random_pick:
            print("===random_pick===")
            idxes = torch.where(confidence > 0.)[0].tolist()
            if len(idxes) >= args.top:
                pl_idx = random.sample(idxes, args.top)
                pl_idx = torch.Tensor(pl_idx).long().to(device)
                conf_score = confidence[pl_idx]
            else:
                pl_idx = []

        if args.conf_pick:
            print("===conf_pick===")
            print(torch.max(confidence), args.threshold)
            pl_idx = torch.where(confidence > args.threshold)[0]
            conf_score = confidence[pl_idx]
            if len(conf_score) >= args.top:
                conf_score, pl_idx = torch.topk(confidence, args.top)

        if args.IGP_pick:
            print("===IGP_pick===")
            # TODO: Check any label leakage
            # we can pseudo label any nodes except with fixed labels
            idx_unlabeled = ~idx_train_ag

            # idx_unlabeled = ~(idx_train_ag | idx_test | idx_val)
            # idx_unlabeled = torch.where( idx_unlabeled == True) [0]

            avg_ece_validation = 0  # ece_validation/ 20

            if not args.PageRank:
                    # label rate
                    beta = (n_node - len(PL_node) - args.top) / n_node

                    # Perform the calculation: beta * influence_matrix_list[0] + beta * beta * influence_matrix_list[1]
                    scaled_matrix_0 = scale_sparse_matrix(influence_matrix_list[0], beta)
                    scaled_matrix_1 = scale_sparse_matrix(influence_matrix_list[1], beta * beta)

                    # Add the scaled sparse matrices
                    influence_matrix = add_sparse_matrices(scaled_matrix_0, scaled_matrix_1).to(
                        device)  # .to_dense().numpy()
                    influence_matrix = beta * influence_matrix_list[0] + beta * beta * influence_matrix_list[1]

                # influence_matrix = influence_matrix_list[1]
            pl_idx = get_IGP_idx_game(g, output_ave, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix,confidence, args, avg_ece_validation)
            # pl_idx = get_IGP_idx(output_ave, labels, idx_train, idx_train_ag, idx_unlabeled, influence_matrix, confidence, args, avg_ece_validation)

        # print_node_attributes(g, labels, pl_idx, idx_train, idx_train_ag, confidence,logger)

        t2 = time.time()
        T_select.append(t2 - t1)

        ##### 3. RETRAINING ########

        if len(pl_idx) > 0:
            conf_score = confidence[pl_idx]
            #print(conf_score)

            pl_labels = predict_labels[pl_idx]
            idx_train_ag[pl_idx] = True
            pred_diff = pseudo_labels[pl_idx] == pl_labels
            pred_diff = len(pred_diff[pred_diff == True]) / len(pred_diff)

            pseudo_labels[pl_idx] = pl_labels
            PL_node += list(pl_idx.cpu().numpy())
            pred_diff_sum = pseudo_labels[idx_train_ag ^ idx_train] == labels[idx_train_ag ^ idx_train]
            pred_diff_sum = len(pred_diff_sum[pred_diff_sum == True]) / len(pred_diff_sum)
            pl_acc.append(pred_diff_sum)

            best_valid, best_output = train(args, model_path, idx_train_ag, idx_val, idx_test, features, adj,
                                            pseudo_labels, labels, bald, T, g, logger, args.FT)

            # Testing
            acc_test, _, output_curr = test(args, adj, features, labels, idx_test, nclass, model_path, g, logger)
            ACC_LIST.append(acc_test)

            # ########
            prob_prev = torch.softmax(output_prev, dim=1)
            prob_curr = torch.softmax(output_curr, dim=1)

            #print('conf train', torch.max(prob_prev[idx_train], dim=1))
            num_class = output_curr.shape[1]
            # prob_prev[idx_train] = convert_to_one_hot(labels[idx_train], num_class).float()

            # output_diff = torch.abs(prob_prev - prob_curr)
            # ent_train, ent_pseudo, ent_unlabeled = (output_diff[idx_train]).numpy(), (
            # output_diff[idx_pseudo]).numpy(), (output_diff[idx_unlabeled]).numpy()
            # print("idx_train diff prob: ", np.mean(ent_train, axis=0))
            # print("idx_pseudo diff prob: ", np.mean(ent_pseudo, axis=0))
            # print("unlabeled diff prob: ", np.mean(ent_unlabeled, axis=0))


            entropy_prev = -torch.sum(prob_prev * torch.log(prob_prev + 1e-9), dim=1)
            entropy_curr = -torch.sum(prob_curr * torch.log(prob_curr + 1e-9), dim=1)

            idx_unlabeled = ~idx_train_ag.cpu().numpy()

            # output_diff = torch.abs(entropy_prev - entropy_curr).numpy()
            # print("idx_train diff ent: ", np.mean(output_diff[idx_train]))
            # print("idx_pseudo diff ent: ", np.mean(output_diff[idx_pseudo]))
            # print("unlabeled diff ent: ", np.mean(output_diff[idx_unlabeled]))

            print("total entropy before:", torch.mean(entropy_prev[idx_unlabeled]), "after:",
                  torch.mean(entropy_curr[idx_unlabeled]))
            diff_unlabeled_ent = torch.mean(entropy_curr[idx_unlabeled]).item() - torch.mean(
                entropy_prev[idx_unlabeled]).item()
            print("ACC +", ACC_LIST[-1] - ACC_LIST[-2], "Entro +", diff_unlabeled_ent)
            ENT_LIST.append(diff_unlabeled_ent)

            output_prev = output_curr

            ########

            val_acc_l.append(best_valid)
            test_acc_l.append(acc_test)

            ### MODIFY
            # conf_test, _ = torch.max(output_ave[idx_test], dim=1)
            #
            # # plt.hist(conf_test, bins=30, alpha=0.5)
            # # plt.title('Histogram of all confidence values')
            # # plt.xlabel('Value')
            # # plt.ylabel('Frequency')
            # # plt.show()
            #
            # conf_avg_test = torch.nanmean( conf_test )

            ### MODIFY
            tests.append(acc_test)

            logger.info(
                'itr {} summary: added {} pl labels with confidence {:.5f}, pl_acc: {}, consistency {:.5f}, {} pl labels in total, test_acc: {:.4f}'.format(
                    itr, len(pl_idx), conf_score.min().item(), pred_diff * 100, consist, len(PL_node), acc_test))
        else:
            print('No labels to add.')
            break

        log_time_of_step(start_time,logging)

        t3 = time.time()
        T_retrain.append(t3 - t2)
        print(T_conf[-1], T_select[-1], T_retrain[-1])

    print(ACC_LIST, ENT_LIST)

    ENT_DIFF = ENT_LIST[: -1]
    ACC_DIFF = np.diff(ACC_LIST)[: len(ENT_DIFF)]

    # plot_ent_acc_fig(ACC_DIFF, ENT_DIFF)

    symbol_correlation = np.sign(ENT_DIFF) == np.sign(ACC_DIFF)
    # Calculate the percentage of correlation in symbol
    symbol_correlation_percentage = np.mean(symbol_correlation) * 100
    logger.info("symbol_correlation_percentage: {:.5f}".format(symbol_correlation_percentage) )

    correlation = np.corrcoef(ENT_DIFF, ACC_DIFF)[0, 1]
    logger.info("Pearson correlation: {:.5f}".format(correlation) )

    logger.info(
        'original acc: {:.5f}, best test accuracy: {:.5f}, final test accuracy: {:.5f}, consistency: {}, pl_acc: {}'.format(
            acc_test0, max(tests), acc_test, consistency[np.argmax(tests)], pl_acc[np.argmax(tests)]))
    logger.info('Best Acc Early Stopped by Valid Acc: {:.5f}'.format(test_acc_l[np.argmax(val_acc_l)]))

    logger.info(
        'Confidence avg time: {:.5f}, Selection avg time: {:.5f}, Retraining avg time: {:.5f}'.format(np.mean(T_conf),
                                                                                                      np.mean(T_select),
                                                                                                      np.mean(
                                                                                                          T_retrain)))
    print("ENDS")
