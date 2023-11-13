from dataset import Amazon,Amazon_reduce,save
import argparse
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import label_binarize
import os.path as osp
import sys
import pickle as pkl
import numpy as np
import scipy
import pandas as pd
from scipy import sparse
from collections import defaultdict
import pickle
import random

import torch
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='All_Beauty', help='dataset name.')
parser.add_argument('--dataset_dir', type=str, default='dataset_raw', help='dataset dictionary')

args = parser.parse_args()

if __name__ == '__main__':

    # is_DiGraph = False
    #
    # def save_file(folder, prefix, name, obj):
    #     path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))
    #     with open(path, 'wb') as f:
    #         pickle.dump(obj, f)
    #
    #
    # # allx x tx都是csr_matrix ally y ty都是nd array graph是defaultdict
    # # 0:Case_Based 1:Genetic_Algorithms 2:Neural_Networks 3:Probabilistic_Methods 4:Reinforcement_Learning 5:Rule_Learning 6:Theory
    #
    # # 导入数据：分隔符为空格
    # nameDic = {"Case_Based": 0, "Genetic_Algorithms": 1, "Neural_Networks": 2, "Probabilistic_Methods": 3,
    #            "Reinforcement_Learning": 4, "Rule_Learning": 5, "Theory": 6}
    # raw_data = pd.read_csv('cora.content', sep='\t', header=None)
    # num_vertex = raw_data.shape[0]  # 样本点数2708
    # raw_data = raw_data.sample(frac=1).reset_index(drop=True)
    # raw_data = raw_data.to_numpy()
    # raw_data = pd.DataFrame(raw_data).reset_index(drop=True)
    #
    # x = raw_data.iloc[:, 1:1434]
    # x = x.to_numpy().astype('float32')
    #
    # # 将论文的编号转[0,2707]
    # a = list(raw_data.index)
    # b = list(raw_data[0])
    # c = zip(b, a)
    # map = dict(c)
    #
    # raw_data_cites = pd.read_csv('cora.cites', sep='\t', header=None)
    # num_edge = raw_data_cites.shape[0]
    #
    # # 创建一个规模和邻接矩阵一样大小的矩阵
    # dic = defaultdict(list)
    # # 创建邻接矩阵
    # for i in range(2708):
    #     dic[i] = []
    # for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
    #     source = map[i]
    #     target = map[j]
    #     dic[source].append(target)
    #     if not is_DiGraph:
    #         dic[target].append(source)
    #
    # allx = scipy.sparse.csr_matrix(x[:1708])
    # tx = scipy.sparse.csr_matrix(x[1708:])
    # x = scipy.sparse.csr_matrix(x[:140])
    # # print(allx,allx.shape)
    #

#-------------------------------------------------------

   #  names = ['x', 'tx', 'allx', 'graph']
   #  objects = []
   #  for i in range(len(names)):
   #      with open("data/ind.cora.{}".format(names[i]), 'rb') as f:
   #          if sys.version_info > (3, 0):
   #              objects.append(pkl.load(f, encoding='latin1'))
   #          else:
   #              objects.append(pkl.load(f))
   #
   #  x, tx, allx, graph = tuple(objects)
   #
   #  ass = np.load("data/cora.x",allow_pickle=True,encoding='latin1')
   # # print(ass, ass.shape, type(ass))
# ------------------------------------------------
#     x, adj = Amazon(args.dataset_dir,args.dataset)
#     print('ok')
#     save(data)




#     test_index = random.sample(range(0, 324038), 500)
#     with open('file', 'w') as f:
#         for line in test_index:
#             f.write("%s\n" % line)



    # # Custom data sampler that returns batch indices
    # class BatchSampler(object):
    #     def __init__(self, num_samples, batch_size):
    #         self.num_samples = num_samples #2708
    #         self.batch_size = batch_size #32
    #
    #     def __iter__(self):
    #         n_batch = len(self) // self.batch_size
    #         tail = len(self) % self.batch_size
    #         index = torch.arange(0, len(self)).long()
    #         for i in range(n_batch):
    #             yield index[i * self.batch_size:(i + 1) * self.batch_size]
    #         if tail:
    #             yield index[-tail:]
    #
    #     def __len__(self):
    #         return self.num_samples
    #
    #
    # # Custom dataset that returns the feature, adj_norm and adj_label for each index
    # class CustomDataset(Dataset):
    #     def __init__(self, features, adj_norm, adj_label):
    #         self.features = features
    #         self.adj_norm = adj_norm
    #         self.adj_label = adj_label
    #
    #     def __getitem__(self, index):
    #         return self.features[:, index], self.adj_norm[index], self.adj_label[index]
    #
    #     def __len__(self):
    #         return self.features.shape[1]
    #
    #
    # # # Create the custom dataset
    # # features = torch.randn(1, 2708, 1433)
    # # adj_norm = torch.randn(2708, 2708)
    # # adj_label = torch.randn(2708, 2708)
    # # dataset = CustomDataset(features, adj_norm, adj_label)
    # #
    # # # Create the batch sampler
    # # num_samples = dataset.__len__()
    # # print(num_samples)
    # # batch_size = 32
    # # batch_sampler = BatchSampler(num_samples, batch_size)
    # #
    # # # Create the DataLoader
    # # train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
    # # for batch_idx, (features_batch, adj_norm_batch, adj_label_batch) in enumerate(train_loader):
    # #     print(features_batch.shape, adj_norm_batch.shape, adj_label_batch.shape)
    #
    # features = torch.randn(1, 2708, 1433)
    # adj_norm = torch.randn(2708, 2708)
    # adj_label = torch.randn(2708, 2708)
    # batch_size = 32
    # num_samples = features.shape[1]
    # num_batches = num_samples // batch_size
    #
    # for batch_idx in range(num_batches):
    #     start = batch_idx * batch_size
    #     end = start + batch_size
    #
    #     # Create batchwise feature data
    #     batch_features = features[:, start:end, :]
    #     batch_adj = adj_norm[start:end, start:end]
    #     batch_adj_label = adj_label[start:end, start:end]
    #
    #     # Yield the data
    #     print(batch_features.shape, batch_adj.shape, batch_adj_label.shape)
    # Example binary labels and predicted scores
    from sklearn.metrics import ndcg_score

    y_true = [[0, 1, 0]]
    print(type(y_true))
    y_pred = [[0.1, 0.2, 0.3]]
    ndcg= ndcg_score(y_true, y_pred)

    print(ndcg)