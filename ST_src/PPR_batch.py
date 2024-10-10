import torch
import dgl
import dgl.function as fn


def prepare_parallel_g(g,args):
    # Precompute out-degree to normalize messages
    degs = g.out_degrees().float().to(args.device)
    degs[degs == 0] = 1  # To avoid division by zero
    degs = degs.view(-1, 1)  # Reshape for broadcasting
    g.ndata['deg'] = degs
    return g

def compute_ppr_matrix_parallel(g, node_set, alpha=0.15, max_iter=100, tol=1e-6):
    N = g.number_of_nodes()
    n = node_set.size(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Uniform teleport matrix
    T = torch.ones(n, N).to(device) / n

    # Initialize PPR matrix
    pagerank = T.clone()

    for _ in range(max_iter):
        prev_pagerank = pagerank.clone()

        # Compute messages
        g.ndata['h'] = pagerank / g.ndata['deg']  # Shape: [N, N_roots]
        # Update all nodes
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_new'))

        # Apply the PageRank formula
        pagerank = alpha * T + (1 - alpha) * g.ndata['h_new']

        # Check for convergence
        diff = torch.norm(pagerank - prev_pagerank, p='fro')
        if diff < tol:
            break

    return pagerank