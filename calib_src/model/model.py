import torch
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv


def create_model(dataset, args):
    """
    Create model with hyperparameters
    """

    num_layers = 2
    if args.model == 'GAT':
        num_hidden = 8
        attention_head = [8, 1]
    else:
        num_hidden = 64

    if args.model == 'GCN':
        return GCN(in_channels=dataset.num_features, num_classes=dataset.num_classes, num_hidden=num_hidden,
                   drop_rate=args.dropout_rate, num_layers=num_layers)
    elif args.model == 'GAT':
        return GAT(in_channels=dataset.num_features, num_classes=dataset.num_classes, num_hidden=num_hidden,
                   attention_head=attention_head, drop_rate=args.dropout_rate, num_layers=num_layers)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers - 2):
            self.feature_list.insert(-1, num_hidden)
        layer_list = []

        for i in range(len(self.feature_list) - 1):
            layer_list.append(["conv" + str(i + 1),
                               GraphConv(self.feature_list[i], self.feature_list[i + 1])])

        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, x, g): # g: edge_idx
        for i in range(len(self.feature_list) - 1):
            x = self.layer_list["conv" + str(i + 1)](g, x)
            if i < len(self.feature_list) - 2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, attention_head, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers - 2):
            self.feature_list.insert(-1, num_hidden)
        attention_head = [1] + attention_head
        layer_list = []
        for i in range(len(self.feature_list) - 1):
            # concat = False if i == num_layers-1 else True
            layer_list.append(
                ["conv" + str(i + 1), GATConv(self.feature_list[i] * attention_head[i], self.feature_list[i + 1],
                                              num_heads=attention_head[i + 1], feat_drop=drop_rate,
                                              attn_drop=drop_rate)])
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, x, g):
        for i in range(len(self.feature_list) - 1):
            x = F.dropout(x, self.drop_rate, self.training)
            x = self.layer_list["conv" + str(i + 1)](g, x)
            if i < len(self.feature_list) - 2:
                x = F.elu(x)
        return x
