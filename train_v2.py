from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import scipy.sparse as sp
import torch

from torch import optim

from model import GCNModelSIGVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, sparse_mx_to_torch_sparse_tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING = 1

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--model', type=str, default='gcn_vae', help="models used.")

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Number of epochs to train.')

parser.add_argument('--Ks', type=int, default=10, help="top k.")
parser.add_argument('--monit', type=int, default=3, help='Number of epochs to train before a test')
parser.add_argument('--edim', type=int, default=32, help='Number of units in noise epsilon.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')  # cora, all_beauty, appliances
parser.add_argument('--encsto', type=str, default='semi', help='encoder stochasticity.')
parser.add_argument('--gdc', type=str, default='ip', help='type of graph decoder')
parser.add_argument('--noise-dist', type=str, default='Bernoulli',
                    help='Distriubtion of random noise in generating psi.')
parser.add_argument('--K', type=int, default=15,
                    help='number of samples to draw for MC estimation of h(psi).')
parser.add_argument('--J', type=int, default=20,
                    help='Number of samples to draw for MC estimation of log-likelihood.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'
print('Using', args.device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def batch_data_generator(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]


class LOADData(torch.utils.data.Dataset):
    def __init__(self):
        self.adj, self.features = load_data(args.dataset_str)
        # self.features = self.features[:, 0:self.adj.shape[0], :].squeeze(0)
        # self.n_nodes, self.feat_dim = self.features.shape[0], self.features.shape[1]
        _, self.n_nodes, self.feat_dim = self.features.shape
        self.adj_train, self.train_edges, self.val_edges, self.val_edges_false, self.test_edges, self.test_edges_false = mask_test_edges(
            self.adj)
        self.adj_norm = preprocess_graph(self.adj_train)
        self.adj_label = self.adj_train + sp.eye(self.adj_train.shape[0])
        # self.adj_label = sparse_mx_to_torch_sparse_tensor(self.adj_label)
        self.adj_label = torch.FloatTensor(self.adj_label.toarray())
        self.pos_weight = torch.tensor(
            [float(self.adj_train.shape[0] * self.adj_train.shape[0] - self.adj_train.sum()) / self.adj_train.sum()])
        self.norm = self.adj_train.shape[0] * self.adj_train.shape[0] / float(
            (self.adj_train.shape[0] * self.adj_train.shape[0] - self.adj_train.sum()) * 2)

    def __getitem__(self, index):
        # adj_norm = self.adj_norm[index].to_dense()
        # adj_label = self.adj_label[index].to_dense()
        adj_norm = self.adj_norm[index]
        adj_label = self.adj_label[index]

        return adj_norm, adj_label, self.features[:, index, :]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.adj.shape[0]

    def ValSampler(self):
        return self.val_edges, self.val_edges_false

    def TestSampler(self):
        return self.test_edges, self.test_edges_false

class ValTestDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data[:min(len(data), len(labels))]
        self.labels = labels[:min(len(data), len(labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx, :]

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))

    # adj, features = load_data(args.dataset_str)
    # _, n_nodes, feat_dim = features.shape

    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # adj = adj_train

    # Some preprocessing
    # adj_norm = preprocess_graph(adj_train)

    # adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    # adj_label = torch.FloatTensor(adj_label.toarray())

    # pos_weight = torch.tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    Batch_Size = args.batch_size
    num_workers = 4
    train_data = LOADData()
    train_loader = DataLoader(dataset=train_data, batch_size=Batch_Size, shuffle=True, num_workers=num_workers,
                              drop_last=True)

    val_edges, val_edges_false = train_data.ValSampler()
    test_edges, test_edges_false = train_data.TestSampler()

    val_dataset = ValTestDataset(val_edges, val_edges_false)
    val_loader = DataLoader(val_dataset, batch_size=Batch_Size,shuffle=True, num_workers=num_workers, drop_last=True)

    test_dataset = ValTestDataset(test_edges, test_edges_false)
    test_loader = DataLoader(test_dataset, batch_size=Batch_Size,shuffle=True, num_workers=num_workers, drop_last=True)

    feat_dim = train_data.feat_dim
    n_nodes = train_data.n_nodes
    pos_weight = train_data.pos_weight
    norm = train_data.norm

    model = GCNModelSIGVAE(
        args.edim, feat_dim, args.hidden1, args.hidden2, args.dropout,
        encsto=args.encsto,
        gdc=args.gdc,
        ndist=args.noise_dist,
        copyK=args.K,
        copyJ=args.J,
        device=args.device
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None

    model.to(args.device)

    # features = features.to(args.device)
    # adj_norm = adj_norm.to(args.device)
    # adj_label = adj_label.to(args.device)
    # pos_weight = pos_weight.to(args.device)

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        hidden_emb = []
        for adj_norm, adj_label, features in tqdm(train_loader):
            # create 3D data for convolution
            print(features)
            if len(features.shape) == 2:
                features = features.view([1, features.shape[0], features.shape[1]])

        recovered, mu, logvar, z, z_scaled, eps, rk, snr = model(features, adj_norm)
        print('xxxx', features.shape, adj_norm.shape,
              adj_label.shape)  # torch.Size([1, 2708, 1433]) torch.Size([2708, 2708])
        loss_rec, loss_prior, loss_post = loss_function(
            preds=recovered,
            labels=adj_label,
            mu=mu,
            logvar=logvar,
            emb=z,
            eps=eps,
            n_nodes=n_nodes,
            norm=norm,
            pos_weight=pos_weight
        )

        WU = np.min([epoch / 300., 1.])
        reg = (loss_post - loss_prior) * WU / (n_nodes ** 2)

        loss_train = loss_rec + WU * reg
        # loss_train = loss_rec
        loss_train.backward()

        cur_loss = loss_train.item()
        cur_rec = loss_rec.item()
        # cur_rec_bce = loss_rec1.item()
        optimizer.step()

        hidden_emb = z_scaled.detach().cpu().numpy()

        roc_curr, ap_curr, pre_curr, rec_curr, ndcg_curr = get_roc_score(hidden_emb, val_edges, val_edges_false,
                                                                         args.gdc, args.Ks)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "rec_loss=", "{:.5f}".format(cur_rec),
              "val_ap=", "{:.5f}".format(ap_curr),
              "precision_val=", "{:.5f}".format(pre_curr),
              "recall_val=", "{:.5f}".format(rec_curr),
              "NDCG_val=", "{:.5f}".format(ndcg_curr),  # ap_curr
              "time=", "{:.5f}".format(time.time() - t)
              )
        # print(rk.detach().cpu().numpy())

        cur_snr = snr.detach().cpu().numpy()
        print("SNR: ", cur_snr)

        if ((epoch + 1) % args.monit == 0):
            model.eval()
            recovered, mu, logvar, z, z_scaled, eps, rk, _ = model(features, adj_norm)
            hidden_emb = z_scaled.detach().cpu().numpy()
            roc_score, ap_score, pre, rec, ndcg = get_roc_score(hidden_emb, val_edges, val_edges_false, args.gdc,
                                                                args.Ks)
            rslt = "Test ROC score: {:.4f}, Test AP score: {:.4f}\n Test Precision: {:.4f}, Test Recall: {:.4f}\n Test NDCG score: {:.4f} \n".format(
                roc_score, ap_score, pre, rec, ndcg)
            print("\n", rslt, "\n")
            with open("results.txt", "a+") as f:
                f.write(rslt)

    print("Optimization Finished!")


if __name__ == '__main__':
    gae_for(args)
