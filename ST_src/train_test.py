import logging
import random
import numpy as np
import sys
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from ST_src.GRAND import consis_loss

from ST_src.banzhaf import *
from ST_src.data import *
from ST_src.models import *
from ST_src.utils import *
from ST_src.utils_new import *

#### sum/mean
criterion = torch.nn.CrossEntropyLoss(reduction='sum').cuda()


def train(args, model_path, idx_train, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, logger, FT = False):
    device = args.device
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

        if args.model == 'GRAND':
            output = model(features, adj, True)
            loss_train = 0
            for k in range(args.sample):
                loss_train += F.nll_loss(output[k][idx_train], pseudo_labels[idx_train])
            loss_train = loss_train / args.sample + consis_loss(output, args.tem, args.lam)
            acc_train = accuracy(output[0][idx_train], pseudo_labels[idx_train])
        else:
            output = model(features, adj)
            output = torch.softmax(output, dim=1)
            output = torch.mm(output, T)
            sign = False
            loss_train = weighted_cross_entropy(output[idx_train], pseudo_labels[idx_train], bald[idx_train], args.beta,
                                                nclass, sign)
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
        # if epoch % 100 == 0:
        #     print("Training success at", epoch ,"epoch with test acc", acc_test)

    print("best result: ", best_print)
    logger.info("best validation result: {:.4f}".format(best))
    return best, best_output


@torch.no_grad()
def test(args, adj, features, labels, idx_test, nclass, model_path, g, logger):
    nfeat = features.shape[1]
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g=g)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    logger.info("Test set results: loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(), acc_test))

    return acc_test, loss_test, output
