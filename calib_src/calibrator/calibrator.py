import copy
from typing import Sequence

import numpy as np
import scipy
import torch
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from torch import nn, optim
from torch.nn import functional as F

from calib_src.calibrator.attention_ts import CalibAttentionLayer
# from calib_src.model.model import GCN
from ST_src.models import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def intra_distance_loss(output, labels):
    """
    Marginal regularization from CaGCN (https://github.com/BUPT-GAMMA/CaGCN)
    """
    output = F.softmax(output, dim=1)
    pred_max_index = torch.max(output, 1)[1]
    correct_i = pred_max_index == labels
    incorrect_i = pred_max_index != labels
    output = torch.sort(output, dim=1, descending=True)
    pred, sub_pred = output[0][:, 0], output[0][:, 1]
    incorrect_loss = torch.sum(pred[incorrect_i] - sub_pred[incorrect_i]) / labels.size(0)
    correct_loss = torch.sum(1 - pred[correct_i] + sub_pred[correct_i]) / labels.size(0)
    return incorrect_loss + correct_loss


def fit_calibration(temp_model, eval, features, adj, labels, train_mask, test_mask, patience=100):
    """
    Train calibrator
    """
    vlss_mn = float('Inf')
    with torch.no_grad():
        if temp_model.model.__class__.__name__== 'GRAND':
            logits = temp_model.model(features, adj, False)
        else:
            logits = temp_model.model(features, adj)
        dev = labels.device
        logits = logits.to(dev)

        model_dict = temp_model.state_dict()
        parameters = {k: v for k, v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(2000):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        # Post-hoc calibration set the classifier to the evaluation mode
        temp_model.model.eval()
        assert not temp_model.model.training
        calibrated = eval(logits)

        loss = F.cross_entropy(calibrated[train_mask], labels[train_mask])
        # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
        # margin_reg = 0.
        # loss = loss + margin_reg * dist_reg
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            calibrated = eval(logits)
            val_loss = F.cross_entropy(calibrated[test_mask], labels[test_mask])
            # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
            # val_loss = val_loss + margin_reg * dist_reg
            if val_loss <= vlss_mn:
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)


class TS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def temperature_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, features, adj, labels, train_mask, test_mask, wdecay):
        self.to(device)

        def eval(logits):
            temperature = self.temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, features, adj, labels, train_mask, test_mask)
        return self


class ETS(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.zeros(1))
        self.weight3 = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.temp_model = TS(model)

    def forward(self, x, adj):
        logits = self.model(x, adj)
        temp = self.temp_model.temperature_scale(logits)
        p = self.w1 * F.softmax(logits / temp, dim=1) + self.w2 * F.softmax(logits,
                                                                            dim=1) + self.w3 * 1 / self.num_classes
        return torch.log(p)

    def fit(self, features, adj, labels, train_mask, test_mask, wdecay):
        self.to(device)
        self.temp_model.fit(features, adj, labels, train_mask, test_mask, wdecay)
        torch.cuda.empty_cache()
        logits = self.model(features, adj)[train_mask]
        label = labels[train_mask]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.unsqueeze(-1), 1)
        temp = self.temp_model.temperature.cpu().detach().numpy()
        w = self.ensemble_scaling(logits.cpu().detach().numpy(), one_hot.cpu().detach().numpy(), temp)
        self.w1, self.w2, self.w3 = w[0], w[1], w[2]
        return self

    def ensemble_scaling(self, logit, label, t):
        """
        Official ETS implementation from Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning
        Code taken from (https://github.com/zhang64-llnl/Mix-n-Match-Calibration)
        Use the scipy optimization because PyTorch does not have constrained optimization.
        """
        EPSILON = 1e-10
        p1 = np.exp(logit) / (np.sum(np.exp(logit), 1) + EPSILON)[:, None]
        logit = logit / t
        p0 = np.exp(logit) / (np.sum(np.exp(logit), 1) + EPSILON)[:, None]
        p2 = np.ones_like(p0) / self.num_classes

        bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0),)

        def my_constraint_fun(x): return np.sum(x) - 1

        constraints = {"type": "eq", "fun": my_constraint_fun, }
        w = scipy.optimize.minimize(ETS.ll_w, (1.0, 0.0, 0.0), args=(p0, p1, p2, label), method='SLSQP',
                                    constraints=constraints, bounds=bnds_w, tol=1e-12, options={'disp': False})
        w = w.x
        return w

    @staticmethod
    def ll_w(w, *args):
        ## find optimal weight coefficients with Cros-Entropy loss function
        p0, p1, p2, label = args
        p = (w[0] * p0 + w[1] * p1 + w[2] * p2)
        N = p.shape[0]
        EPSILON = 1e-10
        ce = -np.sum(label * np.log(p)) / (N + EPSILON)
        return ce


class CaGCN(nn.Module):
    def __init__(self, model, num_nodes, num_class, dropout_rate):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagcn = GCN(nfeat=num_class,
                         nhid=16,
                         nclass=1,
                         dropout=dropout_rate)
        #  def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers)
        #self.cagcn = GCN(num_class, 1, 16, drop_rate=dropout_rate, num_layers=2)

    def forward(self, x, adj):
        logits = self.model(x, adj)
        temperature = self.graph_temperature_scale(logits, adj)
        return logits * F.softplus(temperature)


    def graph_temperature_scale(self, logits, adj):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagcn(logits, adj)
        return temperature

    def fit(self, features, adj, labels, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits, adj)
            calibrated = logits * F.softplus(temperature)
            return calibrated

        self.train_param = self.cagcn.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, features, adj, labels, train_mask, test_mask)
        return self





class GATS(nn.Module):
    def __init__(self, model, edge_index, num_nodes, train_mask, num_class, dist_to_train, heads, bias):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagat = CalibAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         edge_index=edge_index,
                                         num_nodes=num_nodes,
                                         train_mask=train_mask,
                                         dist_to_train=dist_to_train,
                                         heads=heads,
                                         bias=bias)


    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature


    def graph_temperature_scale(self, logits):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))

    def fit(self, features, adj, labels, train_mask, test_mask, wdecay):
        self.to(device)

        def eval(logits):
            temperature = self.graph_temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = self.cagat.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, features, adj, labels, train_mask, test_mask)
        return self
