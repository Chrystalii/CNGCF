import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score,precision_recall_curve, ndcg_score

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        # with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
        #     u = pkl._Unpickler(rf)
        #     u.encoding = 'latin1'
        #     cur_data = u.load()
        #     objects.append(cur_data)

        with open("data/{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
        
    x, tx, allx, graph = tuple(objects)

    test_idx_reorder = parse_index_file(
        "data/{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil() # allx:shape (1708, 1433); tx:  shape (1000, 1433)
    # print(features.shape) #(2708, 1433)
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense(), dtype=np.float32))
    # features = features / features.sum(-1, keepdim=True)
    # adding a dimension to features for future expansion
    if len(features.shape) == 2:
        features = features.view([1,features.shape[0], features.shape[1]])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # print(features.shape)  # torch.Size([1, 2708, 1433])
    # print(adj.shape) #(2708, 2708)
    # print(features)
    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename, encoding='latin1'):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def generate_false_edges(edges_all, val_edges, test_edges, adj):
    all_idx = np.array(list(zip(*edges_all)))
    all_idx = np.unique(all_idx)

    num_val_false = len(val_edges)
    num_test_false = len(test_edges)

    val_false_edges = np.array(list(zip(np.random.randint(0, adj.shape[0], num_val_false),
                                    np.random.randint(0, adj.shape[0], num_val_false))))

    test_false_edges = np.array(list(zip(np.random.randint(0, adj.shape[0], num_test_false),
                                        np.random.randint(0, adj.shape[0], num_test_false))))

    return val_false_edges, test_false_edges

def mask_test_edges(adj):
    # Function to build test set with 10% positive links

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    adj_triu = sp.triu(adj)
    edges = sparse_to_tuple(adj_triu)[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    val_edges_false, test_edges_false = generate_false_edges(edges_all, val_edges, test_edges, adj)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

# def mask_test_edges(adj):
#     # Function to build test set with 10% positive links
#     # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
#     # TODO: Clean up.
#
#     # Remove diagonal elements
#     adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
#     adj.eliminate_zeros()
#
#     # Check that diag is zero:
#     # assert np.diag(adj.todense()).sum() == 0
#
#     adj_triu = sp.triu(adj)
#     edges = sparse_to_tuple(adj_triu)[0]
#     edges_all = sparse_to_tuple(adj)[0]
#     num_test = int(np.floor(edges.shape[0] / 10.))
#     num_val = int(np.floor(edges.shape[0] / 20.))
#
#     all_edge_idx = list(range(edges.shape[0]))
#     np.random.shuffle(all_edge_idx)
#     val_edge_idx = all_edge_idx[:num_val]
#     test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
#     test_edges = edges[test_edge_idx]
#     val_edges = edges[val_edge_idx]
#     train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
#
#     def ismember(a, b, tol=5):
#         rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
#         return np.any(rows_close)
#
#     test_edges_false = []
#     while len(test_edges_false) < len(test_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], edges_all):
#             continue
#         if test_edges_false:
#             if ismember([idx_j, idx_i], np.array(test_edges_false)):
#                 continue
#             if ismember([idx_i, idx_j], np.array(test_edges_false)):
#                 continue
#         test_edges_false.append([idx_i, idx_j])
#
#     val_edges_false = []
#     while len(val_edges_false) < len(val_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], train_edges):
#             continue
#         if ismember([idx_j, idx_i], train_edges):
#             continue
#         if ismember([idx_i, idx_j], val_edges):
#             continue
#         if ismember([idx_j, idx_i], val_edges):
#             continue
#         if val_edges_false:
#             if ismember([idx_j, idx_i], np.array(val_edges_false)):
#                 continue
#             if ismember([idx_i, idx_j], np.array(val_edges_false)):
#                 continue
#         val_edges_false.append([idx_i, idx_j])
#
#     assert ~ismember(test_edges_false, edges_all)
#     assert ~ismember(val_edges_false, edges_all)
#     assert ~ismember(val_edges, train_edges)
#     assert ~ismember(test_edges, train_edges)
#     assert ~ismember(val_edges, test_edges)
#
#     data = np.ones(train_edges.shape[0])
#
#     # Re-build adj matrix
#     adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
#     adj_train = adj_train + adj_train.T
#
#     # NOTE: these edge lists only contain single direction of edge!
#     return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, edges_pos, edges_neg, gdc,k):
    def GraphDC(x):
        if gdc == 'ip':
            return 1 / (1 + np.exp(-x))
        elif gdc == 'bp':
            return 1 - np.exp(- np.exp(x))

    J = emb.shape[0]

    # Predict on test set of edges
    edges_pos = np.array(edges_pos).transpose((1, 0))
    emb_pos_sp = emb[:, edges_pos[0], :]
    emb_pos_ep = emb[:, edges_pos[1], :]

    # preds_pos is torch.Tensor with shape [J, #pos_edges]
    preds_pos = GraphDC(
        np.einsum('ijk,ijk->ij', emb_pos_sp, emb_pos_ep)
    )

    edges_neg = np.array(edges_neg).transpose((1, 0))
    emb_neg_sp = emb[:, edges_neg[0], :]
    emb_neg_ep = emb[:, edges_neg[1], :]

    preds_neg = GraphDC(
        np.einsum('ijk,ijk->ij', emb_neg_sp, emb_neg_ep)
    )

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(preds_pos.shape[-1]), np.zeros(preds_neg.shape[-1])])

    roc_score = np.array(
        [roc_auc_score(labels_all, pred_all.flatten()) \
         for pred_all in np.vsplit(preds_all, J)]
    ).mean()

    ap_score = np.array(
        [average_precision_score(labels_all, pred_all.flatten()) \
         for pred_all in np.vsplit(preds_all, J)]
    ).mean()

    precision = []
    recall = []
    for pred_all in np.vsplit(preds_all, J):
        pre, rec, _ = precision_recall_curve(labels_all, pred_all.flatten())
        precision.append(pre)
        recall.append(rec)

    precision = np.concatenate(precision).mean()
    recall = np.concatenate(recall).mean()

    ndcgs =[]
    for i in range(J):
        ndcg = ndcg_score([labels_all.tolist()], [preds_all[i].tolist()],k= k)
        ndcgs.append(ndcg)

    ndcg = sum(ndcgs) / len(ndcgs)


    return roc_score, ap_score, precision, recall, ndcg
