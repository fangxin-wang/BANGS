import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, WikiCSDataset
from dgl.data import CiteseerGraphDataset
from dgl.data import CoauthorCSDataset
from dgl.data import CoauthorPhysicsDataset
from dgl.data import CoraFullDataset
from dgl.data import CoraGraphDataset
from dgl.data import PubmedGraphDataset, FlickrDataset, RedditDataset, YelpDataset
from dgl.data import AsNodePredDataset
from torch_geometric.datasets import Actor, Twitch, LastFMAsia

import torch
import dgl
from torch_geometric.data import Data
import torch_geometric

from ogb.nodeproppred import DglNodePropPredDataset

def get_noisy_labels(train_mask, val_mask, labels, flip_prob):
    combined_mask = train_mask | val_mask

    label_values = torch.unique(labels)
    rand_vals = torch.rand(labels.shape)
    flip_mask = rand_vals <= flip_prob

    new_labels = torch.empty_like(labels)

    data_num = labels.numel()
    for i in range(data_num):
        if flip_mask[i] & combined_mask[i]:

            exclude_label = labels[i].item()
            mask = label_values != exclude_label
            possible_labels = label_values[mask]

            torch.manual_seed(512)
            random_index = torch.randint(0, possible_labels.size(0), (1,)).item()
            new_labels[i] = possible_labels[random_index].clone()

        else:
            new_labels[i] = labels[i]

    portion_flipped = (new_labels != labels).sum().item() / (combined_mask == True).sum().item()
    print("{:.3f}% of training & validation labels have been randomly flipped. ".format(portion_flipped * 100))

    return new_labels

def get_edge_index(data):
    nxg = torch_geometric.utils.to_networkx(data)
    adj = nx.to_scipy_sparse_array(nxg, dtype=float)
    sparse_mx = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def objn_to_pyg(dataset):
    from torch_geometric.data import Data
    from scipy.sparse import coo_matrix

    # Extract the graph and labels
    dgl_graph, labels = dataset[0]

    # Move the graph and labels to the desired device (e.g., CPU or GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dgl_graph = dgl_graph.to(device)
    labels = labels.squeeze(1).to(device)

    # Extract node features
    node_features = dgl_graph.ndata['feat'].to(device) if 'feat' in dgl_graph.ndata else None

    # Extract edge indices
    edge_index = torch.stack(dgl_graph.edges()).to(device)

    # Extract edge attributes (if any)
    edge_attr = dgl_graph.edata['feat'].to(device) if 'feat' in dgl_graph.edata else None

    # Move edge indices to CPU for adjacency matrix creation
    edge_index_cpu = edge_index.cpu()

    # Create adjacency matrix
    num_nodes = dgl_graph.num_nodes()
    adj_matrix = coo_matrix((torch.ones(edge_index_cpu.shape[1]), (edge_index_cpu[0], edge_index_cpu[1])),
                            shape=(num_nodes, num_nodes))
    adj = torch.sparse_coo_tensor(torch.tensor([adj_matrix.row, adj_matrix.col]), torch.tensor(adj_matrix.data),
                                  torch.Size(adj_matrix.shape)).to(device)

    # Create the PyG Data object
    pyg_data = Data(
        x=node_features,
        y=labels,
        edge_index=edge_index,
        edge_attr=edge_attr,
        adj=adj
    )

    # Print the PyG Data object
    return pyg_data


def dgl_to_pyg(dgl_dataset, device):
    # Check if the dataset contains any graphs and extract the first graph
    if hasattr(dgl_dataset, 'graphs') and len(dgl_dataset.graphs) > 0:
        dgl_graph = dgl_dataset.graphs[0]  # Assuming the first graph is what we want
    elif len(dgl_dataset) > 0:
        dgl_graph = dgl_dataset[0]  # Alternatively, the dataset might be directly iterable
    else:
        raise ValueError("The dataset does not contain any graphs.")

    # Extract node features
    node_features = dgl_graph.ndata['feat'].to(device) if 'feat' in dgl_graph.ndata else None

    # Convert DGL edges to PyG edge_index format
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0).to(device)

    # Extract edge features (if available)
    edge_features = dgl_graph.edata['weight'] if 'weight' in dgl_graph.edata else None

    # Create PyG Data object
    pyg_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features,
                     y=dgl_graph.ndata['label'].to(device))
    pyg_graph.adj = get_edge_index(pyg_graph).to(device)

    return pyg_graph


def pyg_to_dgl(dataset):
    edge_index = dataset.edge_index
    num_nodes = dataset.num_nodes
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    node_features = dataset.x
    if node_features is not None:
        g.ndata['feat'] = node_features
    g.ndata['label'] = dataset.y
    return [g]


def load_data(dataset, device, seed, noisy_portion=0, train_portion=0.05, valid_portion=0.15, portion=True,
              calib=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    citation_data = ['Cora', 'Citeseer', 'Pubmed']
    if dataset == 'Cora':
        data = CoraGraphDataset()
    elif dataset == 'Citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'Pubmed':
        data = PubmedGraphDataset()
    elif dataset == 'CoraFull':
        data = CoraFullDataset()
    elif dataset == 'CaCS':
        data = CoauthorCSDataset()
    elif dataset == 'CaPH':
        data = CoauthorPhysicsDataset()
    elif dataset == 'APh':
        data = AmazonCoBuyPhotoDataset()
    elif dataset == 'Flickr':
        data = FlickrDataset()
    elif dataset == 'Reddit':
        data = RedditDataset()
    elif dataset == 'ACom':
        data = AmazonCoBuyComputerDataset()
    elif dataset == 'Yelp':
        data = YelpDataset()
    elif dataset == 'WikiCS':
        data = WikiCSDataset()
    elif dataset in ['obgnarxiv','ogbnmag','obgnproducts']:
        if dataset == 'obgnarxiv':
            data_obj = DglNodePropPredDataset(name='ogbn-arxiv')
        # hetero -- not tested
        elif dataset == 'ogbnmag':
            exit()
        elif dataset == 'obgnproducts':
            data_obj = DglNodePropPredDataset(name='ogbn-products')

        # pyg_graph = objn_to_pyg(data_obj)
        dataset_ogb = AsNodePredDataset(data_obj)
        dgl_graph = dataset_ogb[0].to(device)


    elif dataset in ['Actor', 'LastFM', 'Twitch']:
        root = os.path.join('dataset', dataset)
        if dataset == 'Actor':
            data = pyg_to_dgl(Actor(root)[0])
        elif dataset == 'Twitch':
            data = pyg_to_dgl(Twitch(root, "PT")[0])
        elif dataset == 'LastFM':
            data = pyg_to_dgl(LastFMAsia(root)[0])
    else:
        raise ValueError('wrong dataset name.')

    if dataset in ['obgnarxiv', 'obgnproducts']:

        g = dgl_graph
        g = dgl.add_self_loop(g)
        features = g.ndata['feat']

    else:
        g = data[0]
        g = dgl.add_self_loop(g)
        features = g.ndata['feat']
        labels = g.ndata['label']

        # pyg_graph = dgl_to_pyg(data, device)

    # Check if one-hot label
    # if labels.max().int().item() == 1:
    #     labels = torch.argmax(labels, dim=1)

    if dataset in citation_data and train_portion == 0.05 and valid_portion == 0.15:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    else:
        # Split with Portion
        if portion:
            portion_list = [train_portion, valid_portion, 1 - (train_portion + valid_portion)]
            train_mask, val_mask, test_mask = generate_mask(g, portion_list, seed)
        # Split with Number in each class
        else:
            train_mask, val_mask, test_mask = split_dataset_class(g, labels, train_portion, valid_portion, seed)
    print('Split')

    # Noisy Setting: Labels in train and val have
    if noisy_portion > 0:
        noisy_labels = get_noisy_labels(train_mask, val_mask, labels, noisy_portion)
        labels = noisy_labels
        print('noisy:', noisy_portion)

    ##########################
    g = g.to(device)
    adj = dgl_only_get_adj(g)
    # adj = graph2adj(g)
    ##########################

    return g, adj, features, labels, train_mask, val_mask, test_mask


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_identity(adj):
    # Get the size of the adjacency matrix
    size = adj.shape[0]

    # Create an identity matrix in sparse format
    identity_indices = torch.arange(size)
    identity_values = torch.ones(size)
    identity_matrix = torch.sparse_coo_tensor(
        torch.stack([identity_indices, identity_indices]),
        identity_values,
        (size, size)
    )

    # Add the identity matrix to the adjacency matrix
    device = adj.device
    adj_with_identity = adj + identity_matrix.to(device)

    return adj_with_identity


def dgl_only_get_adj(g):
    # transform = GCNNorm()
    # g = transform(g)
    adj = g.adj()
    adj = sparse_mx_to_torch_sparse_tensor_dgl(adj)
    adj = add_identity(adj)

    device = adj.device

    rowsum = torch.sparse.sum(adj, dim=1).to_dense()
    print('rowsum')

    # Compute the inverse square root of the rowsum
    d_inv_sqrt = torch.pow(rowsum, -0.5).to(device)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # Handle infinity
    print('d_inv_sqrt')

    # Create a sparse diagonal matrix
    d_inv_sqrt_diag = torch.sparse_coo_tensor(
        torch.stack([torch.arange(adj.size(0)).to(device), torch.arange(adj.size(0)).to(device)]),
        d_inv_sqrt,
        (adj.size(0), adj.size(0))
    )
    print('d_inv_sqrt_diag')

    # Perform the normalization operation in sparse format
    d_inv_sqrt_diag_transpose = d_inv_sqrt_diag.transpose(0, 1)  # torch.sparse.transpose(d_inv_sqrt_diag, 0, 1)
    normalized_adj = torch.sparse.mm(d_inv_sqrt_diag, adj)
    normalized_adj = torch.sparse.mm(normalized_adj, d_inv_sqrt_diag_transpose)

    return normalized_adj


def graph2adj(g):
    nxg = g.cpu().to_networkx()
    adj = nx.to_scipy_sparse_array(nxg, dtype=float)

    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def split_dataset_class(g, labels, n, m, seed ):
    np.random.seed(seed)
    num_nodes = g.number_of_nodes()
    # Extract the labels and find the unique classes
    y = labels
    classes = np.unique(y)
    print(classes)

    train_idx = []
    val_idx = []
    test_idx = []

    # Iterate through each class and assign indices to train, val, and test
    for c in classes:
        # Find all indices of class c
        class_idx = np.where(y == c)[0]
        np.random.shuffle(class_idx)  # Shuffle to ensure random selection

        # Check if there are enough samples
        if len(class_idx) < (n + m):
            raise ValueError(f"Not enough samples in class {c} for the required split.")

        # Append indices to respective lists
        train_idx.extend(class_idx[:n])
        val_idx.extend(class_idx[n:n + m])
        test_idx.extend(class_idx[n + m:])

    # Convert lists to tensors
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def generate_mask(g, labelrate, seed):
    # Generate the train/validation/test masks
    np.random.seed(seed)

    num_nodes = g.number_of_nodes()
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_ratio, val_ratio, _ = labelrate
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    # Generate random indices for each split
    train_indices = np.random.choice(num_nodes, num_train, replace=False)
    val_indices = np.random.choice(np.setdiff1d(np.arange(num_nodes), train_indices), num_val, replace=False)
    test_indices = np.setdiff1d(np.arange(num_nodes), np.concatenate((train_indices, val_indices)))

    # Set the corresponding mask values to True
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)


def preprocess_adj(adj, with_ego=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    if with_ego:
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        adj_normalized = normalize_adj(adj)
    return adj_normalized


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()  # D
    print('rowsum')
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)  # D^-0.5
    print('d_inv_sqrt')
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    print('d_mat_inv_sqrt')
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def sparse_mx_to_torch_sparse_tensor_dgl(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.float()  # coo().astype(np.float32)
    # indices = torch.from_numpy(
    #         np.vstack((sparse_mx.row(), sparse_mx.col())).astype(np.int64))
    indices = torch.vstack((sparse_mx.row, sparse_mx.col))
    values = sparse_mx.val
    shape = sparse_mx.shape
    return torch.sparse.FloatTensor(indices, values, shape)
