import dgl
from copy import deepcopy

import scipy.sparse as sp
import torch.optim as optim

from ST_src.data import *
from ST_src.models import *
from ST_src.utils_new import *

def get_models(args, nfeat, nclass, g, FT=False):
    model_name = args.model
    if not FT:
        droprate = args.dropout
    else:
        droprate = args.aug_drop
    if model_name == 'GCN':
        model = GCN(nfeat=nfeat,
                    nhid=args.hidden,
                    nclass=nclass,
                    dropout=droprate)
    elif model_name == 'GAT':
        model = GAT(g=g,
                    num_layers=2,
                    in_dim=nfeat,
                    num_hidden=args.hidden,
                    num_classes=nclass,
                    heads=([args.nb_heads] * 1) + [args.nb_out_heads],
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=args.alpha,
                    residual=True)
    #
    # elif model_name == 'GraphTransformer':
    #     model = GraphTransformer(g = g,
    #                              nfeat = nfeat ,
    #                              nhid = args.hidden,
    #                              nclass = nclass,
    #                              dropout = droprate)
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(g=g,
                          in_feats=nfeat,
                          n_hidden=args.hidden,
                          n_classes=nclass,
                          activation=F.relu,
                          dropout=droprate,
                          aggregator_type='gcn') # 'gcn'
    elif model_name == 'APPNP':
        model = APPNP(g=g,
                      in_feats=nfeat,
                      hiddens=args.hidden,
                      n_classes=nclass,
                      activation=F.relu,
                      dropout=droprate,
                      alpha=0.1,
                      k=10)
    elif model_name == 'GIN':
        model = GIN(g=g,
                    in_feats=nfeat,
                    hidden=args.hidden,
                    n_classes=nclass,
                    activation=F.tanh,
                    feat_drop=droprate,
                    eps=0.2)
    elif model_name == 'SGC':
        model = SGC(g=g,
                    in_feats=nfeat,
                    n_classes=nclass,
                    num_k=2)
    elif model_name == 'MixHop':
        model = MixHop(g=g,
                       in_dim=nfeat,
                       hid_dim=args.hidden,
                       out_dim=nclass,
                       num_layers=args.num_layers,
                       input_dropout=droprate,
                       layer_dropout=0.9,
                       activation=torch.tanh,
                       batchnorm=True)

    return model


def accuracy(pred, targ):
    pred = torch.softmax(pred, dim=1)
    pred_max_index = torch.max(pred, 1)[1]
    ac = ((pred_max_index == targ).float()).sum().item() / targ.size()[0]
    return ac


def weighted_cross_entropy(output, labels, bald, beta, nclass, sign=True):
    bald += 1e-6
    if sign:
        output = torch.softmax(output, dim=1)


    # output[output==0] += 1e-6
    # output[output==1] -= 1e-6
    bald = bald / (torch.mean(bald) * beta)
    labels = F.one_hot(labels, nclass)
    loss = -torch.log(torch.sum(output * labels, dim=1))
    loss = torch.sum(loss * bald)
    loss /= labels.size()[0]

    return loss


def get_norm_logit(output, with_softmax=False):
    if with_softmax:
        output = torch.softmax(output, dim=1)
    # Substitute inf & -inf value
    else:
        EPS = 0.0001
        logits_no_inf = output[torch.isfinite(output)]
        max_finite_value, min_finite_value =  torch.max(logits_no_inf), torch.min(logits_no_inf)
        output[output == float('inf')] = max_finite_value
        output[output == -float('inf')] = min_finite_value
        output = torch.abs(output)
        print(torch.max(output), torch.min(output))

        max_value = torch.max(output)
        min_value = torch.min(output) + EPS
        output = (output - min_value) / (max_value-min_value)

    print(torch.max(output), torch.min(output))
    return output

def get_confidence(output, with_softmax=True):
    if with_softmax:
        output = torch.softmax(output, dim=1)
    confidence, pred_label = torch.max(output, dim=1)
    return confidence, pred_label

@torch.no_grad()
def multiview_pred(model, features, adj, g, args):
    device = args.device
    best_output = model(features, adj)
    _, pred = get_confidence(best_output)
    print("get_confidence")
    consist = torch.ones(pred.shape).bool().to(device)
    
    if not args.multiview:
        for _ in range(3):
            output_d = model(features, adj)
            best_output += output_d
            _, pred_d = get_confidence(output_d)
            consist *= pred == pred_d
    else:
        # drop feature
        features_aug = F.dropout(features, p=args.aug_drop)
        print('features_aug')
        aug_g = deepcopy(g).to(args.device)
        aug_g.ndata['feat'] = features_aug
        if args.model != 'GCN':
            model.g = aug_g
        output_aug = model(features_aug, adj)
        best_output += output_aug
        _, pred_aug = get_confidence(output_aug)
        consist *= pred == pred_aug

        # drop edge
        num_edges = g.number_of_edges()
        num_dropped_edges = int(num_edges * args.aug_drop)
        edges_to_drop = np.random.choice(num_edges, num_dropped_edges, replace=False)
        g_aug = dgl.remove_edges(g, edges_to_drop).to(device)

        ############
        # adj_aug = graph2adj(g_aug).to(device)
        adj_aug = dgl_only_get_adj(g_aug).to(device)
        ############


        if args.model != 'GCN':
            model.g = aug_g
            output_aug = model(features, adj)
        else:
            output_aug = model(features, adj_aug)
        best_output += output_aug
        _, pred_aug = get_confidence(output_aug)
        consist *= pred == pred_aug

        # feature noise
        aug_g = deepcopy(g).to(args.device)
        noise = torch.rand(g.ndata['feat'].shape).to(device) * torch.std(g.ndata['feat']) * args.aug_drop / 2
        features_aug = features + noise
        aug_g.ndata['feat'] += noise 
        if args.model != 'GCN':
            model.g = aug_g
        output_aug = model(features_aug, adj)
        best_output += output_aug
        _, pred_aug = get_confidence(output_aug)
        consist *= pred == pred_aug

    best_output /= 4
    scores, pl_labels = get_confidence(best_output)
    return best_output, scores, pl_labels, consist.cpu().float().mean().item()


## drgst_utils
def get_mc_adj(oadj, device, droprate=0.1):
    f_pass = 100
    edge_index = oadj.coalesce().indices()
    mc_adj = []
    for i in range(f_pass):
        adj_tmp = oadj.clone().to_dense()
        drop = np.random.random(edge_index.size()[1])
        drop = np.where(drop < droprate)[0]
        edge_index_tmp = edge_index[:, drop]
        adj_tmp[edge_index_tmp[0], edge_index_tmp[1]] = 0

        adj_tmp = preprocess_adj(sp.coo_matrix(adj_tmp))
        adj_tmp = sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device)

        mc_adj.append(adj_tmp)
    return mc_adj

def update_T(output, idx_train, labels, T, device):
    output = torch.softmax(output, dim=1)
    T.requires_grad = True
    optimizer = optim.Adam([T], lr=0.01, weight_decay=5e-4)
    mse_criterion = torch.nn.MSELoss().cuda()
    index = torch.where(idx_train)[0]
    nclass = labels.max().item() + 1
    for epoch in range(200):
        optimizer.zero_grad()
        loss = mse_criterion(output[index], T[labels[index]]) + mse_criterion(T, torch.eye(nclass).to(device))
        loss.backward()
        optimizer.step()
    T.requires_grad = False
    return T

def uncertainty_dropedge(mc_adj, adj, features, g, nclass, model_path, args, device):
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g=g)
    model.load_state_dict(state_dict)
    model.to(device)
    out_list = []
    with torch.no_grad():
        model.eval()
        for madj in mc_adj:
            output = model.gc1(features, adj)
            output = torch.relu(output)
            output = model.gc2(output, madj)
            output = torch.softmax(output, dim=1)
            output = output + 1e-15
            out_list.append(output)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        entropy = torch.sum(torch.mean(out_list * torch.log(out_list), dim=0), dim=1)
        Eentropy = torch.sum(out_mean * torch.log(out_mean), dim=1)
        bald = entropy - Eentropy
    return bald

def uncertainty_dropout(adj, features, g, nclass, model_path, args, device):
    f_pass = 100
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g)
    model.load_state_dict(state_dict)
    model.to(device)
    out_list = []
    with torch.no_grad():
        for _ in range(f_pass):
            output = model(features, adj)
            output = torch.softmax(output, dim=1)
            out_list.append(output)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        out_list[out_list == 0] += 1e-6
        out_list[out_list == 1] -= 1e-6
        out_mean[out_mean == 0] += 1e-6
        out_mean[out_mean == 1] -= 1e-6
        entropy = torch.sum(torch.mean(out_list * torch.log(out_list), dim=0), dim=1)
        Eentropy = torch.sum(out_mean * torch.log(out_mean), dim=1)
        bald = entropy - Eentropy
    return bald

def regenerate_pseudo_label(output, labels, idx_train, unlabeled_index, threshold, device, sign=False):
    're-generate pseudo labels every stage'
    unlabeled_index = torch.where(unlabeled_index == True)[0]
    confidence, pred_label = get_confidence(output, sign)
    index = torch.where(confidence > threshold)[0]
    pseudo_index = []
    pseudo_labels, idx_train_ag = labels.clone().to(device), idx_train.clone().to(device)
    for i in index:
        if i not in idx_train:
            pseudo_labels[i] = pred_label[i]
            # pseudo_labels[i] = labels[i]
            if i in unlabeled_index:
                idx_train_ag[i] = True
                pseudo_index.append(i)
    idx_pseudo = torch.zeros_like(idx_train)
    pseudo_index = torch.tensor(pseudo_index)
    if pseudo_index.size()[0] != 0:
        idx_pseudo[pseudo_index] = 1
    return idx_train_ag, pseudo_labels, idx_pseudo
#
# def generate_IGP_pseudo_label(adj, output, labels, idx_train, idx_unlabeled, device = 'cpu'):
#
#     idx_unlabeled = torch.where(idx_unlabeled == True)[0]
#
#     output = torch.softmax(output, dim=1)
#     confidence, pred_label = torch.max(output, dim=1)
#
#     influence_matrix = get_influence_matrix(adj, k = 2)
#     selected_index = get_IGP_idx(output, labels, idx_train, idx_unlabeled, influence_matrix, N_selected = 20)
#
#     pseudo_index = []
#     pseudo_labels, idx_train_ag = labels.clone().to(device), idx_train.clone().to(device)
#     for i in selected_index:
#         if i not in idx_train:
#             pseudo_labels[i] = pred_label[i]
#             if i in idx_unlabeled:
#                 idx_train_ag[i] = True
#                 pseudo_index.append(i)
#     idx_pseudo = torch.zeros_like(idx_train)
#     pseudo_index = torch.tensor(pseudo_index)
#     if pseudo_index.size()[0] != 0:
#         idx_pseudo[pseudo_index] = 1
#     return idx_train_ag, pseudo_labels, idx_pseudo


def plot_ent_acc_fig(ACC_DIFF, ENT_DIFF):
    import matplotlib.pyplot as plt
    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    # Accuracy line
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(ACC_DIFF, color='tab:blue', label='Accuracy Diff')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Entropy line (second y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Entropy', color='tab:red')
    ax2.plot(ENT_DIFF, color='tab:red', label='Entropy Diff')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Show plot
    fig.tight_layout()
    plt.title("Accuracy and Entropy Over Time")
    plt.show()



