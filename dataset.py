'''
Build dataset for Synthetic, Amazon Product Reviews (Amazon-Beauty, Amazon-Appliances) and Epinion
    input: original dataset from official website
    output: for core_file in names = ['x', 'tx', 'allx', 'graph','test.index']:
                return dataset.core_file

    example shape:
    # allx = scipy.sparse.csr_matrix(x[:1708])
    # tx = scipy.sparse.csr_matrix(x[1708:])
    # x = scipy.sparse.csr_matrix(x[:140])
    # print(x.shape) # sample like:  (139, 1393)	1.0  # shape (140, 1433)
    # print(tx.shape) # sample like: (999, 65)	1.0  # shape (1000, 1433)
    # print(allx.shape) # sample like:  (1, 233)	1.0 # shape (1708, 1433)
    # print(len(graph)) # defaultdict(<class 'list'>, {0: [633, 1862, 2582], 1: [2, 652, 654], 2: [1986, 332, 1666, 1, 1454], # len: 2708

'''

import os
import json
import gzip
import pandas as pd
import scipy
import argparse
import pickle
import random
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import vstack
pd.options.mode.chained_assignment = None

def get_reviewer_features(reviewer_id, asins, feats_dict, brand_num, price_num):
    temp_encoding = [0] * (brand_num+ price_num)
    temp_encoding[0] = 1
    return [feats_dict.get(asin, temp_encoding) for asin in asins]

def allx_generator(graph, feats_dict, brand_num, price_num):
    for reviewer_id, asins in graph.items():
        yield reviewer_id, get_reviewer_features(reviewer_id, asins, feats_dict, brand_num, price_num)

def save_file(folder, prefix, name, obj):
    path = os.path.join(folder, '{}.{}'.format(prefix.lower(), name))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def synthetic():
    pass

def Amazon(dataset_dir, datset):

    print('Using {} dataset...'.format(datset))

    # load raw data
    interaction_data = []
    with gzip.open('{}/{}.json.gz'.format(dataset_dir, datset)) as f:
        for l in f:
            interaction_data.append(json.loads(l.strip()))
    print('total reviews:', len(interaction_data))

    meta_data = []
    with gzip.open('{}/meta_{}.json.gz'.format(dataset_dir, datset)) as f:
        for l in f:
            meta_data.append(json.loads(l.strip()))
    print('total meta data:', len(meta_data))

    # build interaction data
    interactions = pd.DataFrame.from_dict(interaction_data)
    print('user num:', len(interactions['reviewerID'].unique())) #324038
    print('item num:',len(interactions['asin'].unique())) # 32586
    graph = interactions.groupby('reviewerID')['asin'].apply(list).to_dict()

    # graph = defaultdict(list)
    # for i in range(len(interactions)):
    #     user_idx = interactions.iloc[i]['reviewerID']
    #     item_idx = interactions.iloc[i]['asin']
    #     graph[user_idx].append(item_idx)

    # build adjacency matrix
        # Create numerical IDs for the reviewerID and asin columns
    interactions['reviewerID_idx'],_ = interactions['reviewerID'].factorize()
    interactions['asin_idx'],_ = interactions['asin'].factorize()

        # Create a defaultdict to store the adj
    adj = interactions.groupby('reviewerID_idx')['asin_idx'].apply(list).to_dict()

    # build features
    meta = pd.DataFrame.from_dict(meta_data)
    feats = ['asin','brand', 'price']
    product_feats = meta[feats]

    # convert price to four categories
    product_feats["price"] = pd.to_numeric(product_feats["price"].str.replace("$", ""), errors='coerce')
    max_price = product_feats["price"].max()
    min_price = product_feats["price"].min()

    price_bin_width = (max_price - min_price) / 3
    price_bins = [min_price, min_price + price_bin_width, min_price + price_bin_width * 2, max_price]
    price_labels = ["low", "medium", "high"]
    product_feats["price_tag"] = pd.cut(product_feats["price"], bins=price_bins, labels=price_labels, right=False)
    product_feats['price_tag'] = product_feats['price_tag'].cat.add_categories('unknown')
    product_feats['price_tag'].fillna('unknown', inplace=True)

    # drop original numbered price
    product_feats = product_feats.drop(['price'], axis=1)
    brand_num = len(product_feats['brand'].unique())
    price_num = len(product_feats["price_tag"].unique())
    print('Unique brand ids:', brand_num) #7863
    print('Unique price ids:', price_num) #4

    # One-hot encode the categorical features
    feats_one_hot = pd.concat([product_feats['asin'], pd.get_dummies(product_feats['brand'], prefix='brand'),
                               pd.get_dummies(product_feats['price_tag'], prefix='price')], axis=1)  # (32892, 7868)

    # map asin features to reviewer features to build allx
    feats_dict = {}
    for index, row in feats_one_hot.iterrows():
        key = row['asin']
        values = row[1:].tolist()
        feats_dict[key] = values

    temp_encoding =[0] * (brand_num+ price_num)
    temp_encoding[123] = 1
    allx_dict = {reviewer_id: [feats_dict.get(asin,temp_encoding) for asin in asins] for reviewer_id, asins in graph.items()} # 324038
    # allx_dict_single = {reviewer_id: feats[0] for reviewer_id, feats in allx_dict.items()}
    allx = pd.DataFrame(list(allx_dict.items()), columns=['id', 'features'])
    allx['features'] = allx['features'].apply(lambda x: x[0])
    allx = allx.drop(['id'], axis=1)
    print(allx.shape)

    allx = vstack(allx['features'].values, format='csr')
    print(allx.shape)

    return allx, adj

def save(allx, adj,out_dataset_dir, datset):

    all_samples = allx.shape[0]

    x = scipy.sparse.csr_matrix(allx[:10000]) # training node features
    tx = scipy.sparse.csr_matrix(allx[20000:]) # tx: testing node features
    allx = scipy.sparse.csr_matrix(allx[:20000]) # allx: all node features
    test_index = random.sample(range(0, all_samples), 500)
    print('files shape', x.shape, tx.shape, allx.shape, len(adj), len(test_index))

    test_index = random.sample(range(0, all_samples), 500)
    with open('{}.test.index'.format(datset), 'w') as f:
        for line in test_index:
            f.write("%s\n" % line)


    names = [('x', x), ('tx', tx), ('allx', allx), ('graph', adj)]
    items = [save_file(out_dataset_dir, datset, name[0], name[1]) for name in names]

def Amazon_reduce(dataset_dir, dataset):
    print('Using {} dataset...'.format(dataset))

    '''
    Interaction data:
         reviewerID - ID of the reviewer, e.g.A2SUAM1J3GNN3B
         asin - ID of the product, e.g.0000013714
         overall - rating of the product

    Amazon beauty metadata:
    #['category', 'tech1', 'description', 'fit', 'title', 'also_buy', 'tech2',
       'brand', 'feature', 'rank', 'also_view', 'details', 'main_cat',
       'similar_item', 'date', 'price', 'asin', 'imageURL', 'imageURLHighRes']

    Amazon appliance metadata:
    ['category', 'tech1', 'description', 'fit', 'title', 'also_buy', 'tech2',
       'brand', 'feature', 'rank', 'also_view', 'details', 'main_cat',
       'similar_item', 'date', 'price', 'asin', 'imageURL', 'imageURLHighRes']
    '''

    # load raw data
    interaction_data = []
    with gzip.open('{}/{}.json.gz'.format(dataset_dir, dataset)) as f:
        for l in f:
            interaction_data.append(json.loads(l.strip()))
    print('total reviews:', len(interaction_data))

    meta_data = []
    with gzip.open('{}/meta_{}.json.gz'.format(dataset_dir, dataset)) as f:
        for l in f:
            meta_data.append(json.loads(l.strip()))
    print('total meta data:', len(meta_data))

    # build interaction data
    interactions = pd.DataFrame.from_dict(interaction_data)

    # select positive interactions while ratings >= 3
    interactions = interactions[interactions['overall'] >= 3]
    print('interactions', interactions.shape)
    print('user num:', len(interactions['reviewerID'].unique()))
    print('item num:', len(interactions['asin'].unique()))
    graph = interactions.groupby('reviewerID')['asin'].apply(list).to_dict()
    print(len(graph)) # 271036

    # build features
    meta = pd.DataFrame.from_dict(meta_data)
    feats = ['asin', 'brand', 'price']
    product_feats = meta[feats]

    # convert price to four categories
    product_feats["price"] = pd.to_numeric(product_feats["price"].str.replace("$", ""), errors='coerce').astype(
        'float32')
    max_price = product_feats["price"].max()
    min_price = product_feats["price"].min()

    price_bin_width = (max_price - min_price) / 3
    price_bins = [min_price, min_price + price_bin_width, min_price + price_bin_width * 2, max_price]
    price_labels = ["low", "medium", "high"]
    product_feats["price_tag"] = pd.cut(product_feats["price"], bins=price_bins, labels=price_labels, right=False)
    product_feats['price_tag'] = product_feats['price_tag'].cat.add_categories('unknown')
    product_feats['price_tag'].fillna('unknown', inplace=True)

    # drop original numbered price
    product_feats = product_feats.drop(['price'], axis=1)
    brand_num = len(product_feats['brand'].unique())
    price_num = len(product_feats["price_tag"].unique())
    print('Unique brand ids:', brand_num)  # 7863
    print('Unique price ids:', price_num)  # 4

    # One-hot encode the categorical features
    feats_one_hot = pd.concat([product_feats['asin'], pd.get_dummies(product_feats['brand'], prefix='brand',dtype='int32'),
                               pd.get_dummies(product_feats['price_tag'], prefix='price',dtype='int32')], axis=1)  # (32892, 7868)

    allx = pd.DataFrame(allx_generator(graph, feats_one_hot, brand_num, price_num), columns=['id', 'features'])  #(271036, 2)

    # allx = np.array(allx['features'].tolist(),dtype='int32') #(271036,)
    # allx = scipy.sparse.csr_matrix(allx)

    allx = vstack(allx['features'].values, format='csr')
    print(allx.shape) # (311791, 7867)

    # build adjacency matrix
    # Create numerical IDs for the reviewerID and asin columns
    interactions['reviewerID_idx'], _ = interactions['reviewerID'].factorize()
    interactions['asin_idx'], _ = interactions['asin'].factorize()

    # Create a defaultdict to store the adj
    adj = interactions.groupby('reviewerID_idx')['asin_idx'].apply(list).to_dict()
    print(len(adj))

    return allx, adj


def Convert(a):
    it = iter(a)
    res_dct = dict(zip(it, it))
    return res_dct

def epinions_allx_generator(graph, feats_dict, price_num):
    for reviewer_id, asins in graph.items():
        yield reviewer_id, epinions_get_reviewer_features(reviewer_id, asins, feats_dict, price_num)

def epinions_get_reviewer_features(reviewer_id, asins, feats_dict, price_num):
    temp_encoding = [0] * (price_num)
    temp_encoding[0] = 1
    return [feats_dict.get(asin, temp_encoding) for asin in asins]

def Epinion(dataset_dir, dataset):
    print('Using {} dataset...'.format(dataset))

    # load raw data
    interaction_data = []
    with open('{}/{}.json'.format(dataset_dir, dataset)) as f:
        for l in f:
            json_data = ast.literal_eval(json.dumps(l))
            json_data = json_data.replace("\'", "\"")
            try:
                jsons = json.loads(json_data.strip())
                interaction_data.append(jsons)
            except:
                continue
    print('total interaction:', len(interaction_data))

    #print(Convert(interaction_data[0])) #{'unixtime': 'paid', 'review': 'userId', 'itemId': 'stars'}

    interactions = pd.DataFrame.from_dict(interaction_data)

    # create label encoder objects for user and item IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    # fit the label encoders to the user and item IDs in the dataframe
    user_encoder.fit(interactions['userId'])
    item_encoder.fit(interactions['itemId'])

    # transform the user and item IDs in the dataframe to numerical IDs
    interactions['userId'] = user_encoder.transform(interactions['userId'])
    interactions['itemId'] = item_encoder.transform(interactions['itemId'])

    # select positive interactions while ratings >= 3
    interactions = interactions[interactions['stars'] >= 3]
    print('interactions', interactions.shape)
    print('user num:', len(interactions['userId'].unique()))
    print('item num:', len(interactions['itemId'].unique()))
    graph = interactions.groupby('userId')['itemId'].apply(list).to_dict()
    print(len(graph))  # 83520

    # build features
    feats = ['itemId', 'paid']
    product_feats = interactions[feats]

    # convert paid to four categories
    product_feats["paid"] = pd.to_numeric(product_feats["paid"], errors='coerce').astype(
        'float32')
    max_price = product_feats["paid"].max()
    min_price = product_feats["paid"].min()

    price_bin_width = (max_price - min_price) / 3
    price_bins = [min_price, min_price + price_bin_width, min_price + price_bin_width * 2, max_price]
    price_labels = ["low", "medium", "high"]
    product_feats["price_tag"] = pd.cut(product_feats["paid"], bins=price_bins, labels=price_labels, right=False)
    product_feats['price_tag'] = product_feats['price_tag'].cat.add_categories('unknown')
    product_feats['price_tag'].fillna('unknown', inplace=True)

    # drop original numbered price
    product_feats = product_feats.drop(['paid'], axis=1)
    price_num = len(product_feats["price_tag"].unique())
    print('Unique price ids:', price_num)

    # One-hot encode the categorical features
    feats_one_hot = pd.get_dummies(product_feats['price_tag'], prefix='price', dtype='int32') # (32892, 7868)

    print(feats_one_hot.shape)

    allx = pd.DataFrame(epinions_allx_generator(graph, feats_one_hot, price_num),
                        columns=['id', 'features'])  # (271036, 2)

    allx = vstack(allx['features'].values, format='csr')
    print(allx.shape)  # (311791, 7867)

    # Create a defaultdict to store the adj
    adj = interactions.groupby('userId')['itemId'].apply(list).to_dict()
    print(len(adj)) # 83520

    return allx, adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='epinions', help='dataset name.') #all_beauty
    parser.add_argument('--raw_dataset_dir', type=str, default='dataset_raw', help='dataset dictionary')
    parser.add_argument('--out_dataset_dir', type=str, default='data', help='processed dataset dictionary')

    args = parser.parse_args()
    #
    # allx, adj= Epinion(args.raw_dataset_dir,args.dataset)
    # print('ok')
    # save(allx, adj, args.out_dataset_dir,args.dataset)

    cora_content = np.loadtxt('cora.content', delimiter='\t', dtype=str)
    print
    cora_content.shape

