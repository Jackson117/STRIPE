import datetime
import numpy as np
from sklearn.cluster import SpectralClustering

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, dense_to_sparse


def anomaly_generation_dy(adjs, node_feas, anomaly_per, dataset, mode='train', seed =666):
    np.random.seed(seed)

    time_stamp = adjs.shape[0]
    times = np.arange(time_stamp)
    if mode == 'train':
        times = times[:int(time_stamp * 0.7)]
    else:
        times = times[int(time_stamp * 0.7):]
    print('{} data has time stamp number: {}'.format(mode, times.shape[0]))

    node_num = adjs.shape[1]
    anomaly_num = int(np.floor(anomaly_per * node_num))
    # an_dic = {'reddit':420, 'Brain':240, 'DBLP3':210, 'DBLP5':330, 'bitcoinOTC': 300}
    # anomaly_num = an_dic[dataset]

    node_feas = np.transpose(node_feas, (1, 0, 2))

    adj0 = torch.from_numpy(adjs[0, :, :])
    edge_index_0, edge_attrs = dense_to_sparse(adj0)
    node_deg = degree(edge_index_0[:, 0], node_num).int()
    # k_deg, outlier_idx = torch.topk(node_deg, k=anomaly_num, largest=False)
    outlier_idx = np.arange(node_num)
    np.random.shuffle(outlier_idx)
    outlier_idx = torch.LongTensor(outlier_idx[:anomaly_num])

    graph_list = []
    for t in times:
        adj, node_fea = torch.from_numpy(adjs[t, :, :]), node_feas[t, :, :]
        edge_index, edge_attr = dense_to_sparse(adj)
        edge_list = np.transpose(edge_index.numpy(), (1, 0))
        lbl = np.zeros(node_fea.shape[0], dtype=np.int64)

        if mode == 'test':
            node_fea, edge_list, lbl = gen_structural_outlier(node_fea, edge_list, anomaly_num,
                                                                     outlier_idx=outlier_idx, m=15, seed=seed)

            node_fea, edge_list, lbl = gen_attribute_outlier(node_fea, edge_list, anomaly_num, k=15,
                                  src_n_idx=outlier_idx, ano_n_label=lbl, seed=seed)

        x = torch.FloatTensor(node_fea)
        lbl = torch.LongTensor(lbl)
        edge_index = torch.transpose(torch.LongTensor(edge_list), 1, 0)

        data = Data(x, edge_index, y=lbl, t=t, num_nodes=node_num)
        graph_list.append(data)


    return graph_list

def gen_structural_outlier(node_fea, edge_list, anomaly_num, m, p=0, outlier_idx=None, ano_n_label=None, directed=True, seed=None):
    if seed:
        torch.manual_seed(seed)

    new_edges = []

    if outlier_idx is None:
        node_deg = degree(torch.tensor(edge_list[:, 0]), node_fea.shape[0]).int()
        k_deg, outlier_idx = torch.topk(node_deg, k=anomaly_num, largest=False)


    # np.random.shuffle(idx_node)

    n = int(np.floor(anomaly_num / m)) # number of dense cliques
    # outlier_idx = torch.tensor(idx_node, dtype=torch.long)[:anomaly_num]

    # connect all m nodes in each clique
    for i in range(n):
        new_edges.append(torch.combinations(outlier_idx[m * i: m * (i + 1)]))

    new_edges = torch.cat(new_edges)

    # drop edges with probability p
    if p != 0:
        indices = torch.randperm(len(new_edges))[:int((1-p) * len(new_edges))]
        new_edges = new_edges[indices]

    if ano_n_label == None:
        y_n_outlier = np.zeros(node_fea.shape[0], dtype=np.int64)
    else:
        y_n_outlier = ano_n_label
    y_n_outlier[outlier_idx] = 1

    if not directed:
        new_edges = torch.cat([new_edges, new_edges.flip(1)], dim=0)

    new_edges = new_edges.numpy()
    edge_list = np.concatenate((edge_list, new_edges), axis=0)
    np.random.shuffle(edge_list)

    return node_fea, edge_list, y_n_outlier

def gen_attribute_outlier(node_fea, edge_list, anomaly_num, k, src_n_idx=None, ano_n_label=None, directed=True, seed=None):
    if seed:
        torch.manual_seed(seed)

    idx_perm = np.unique(edge_list)

    if src_n_idx is None:
        node_deg = degree(torch.tensor(edge_list[:, 0]), idx_perm.shape[0]).int()
        k_deg, src_n_idx = torch.topk(node_deg, k=anomaly_num, largest=False)
        src_n_idx = src_n_idx.numpy()

    # np.random.shuffle(idx_perm)
    # src_n_idx = idx_perm[:n]

    # new_edges = []
    x = torch.FloatTensor(node_fea)
    for i, idx in enumerate(src_n_idx):
        np.random.shuffle(idx_perm)
        candidate_idx_n = idx_perm[:k]

        euclidean_dist_n = torch.cdist(x[idx].unsqueeze(0), x[
            candidate_idx_n])

        max_dist_idx = torch.argmax(euclidean_dist_n, dim=1)
        max_dist_node = candidate_idx_n[max_dist_idx]
        x[idx] = x[max_dist_node]

    # new_edges = torch.cat(new_edges, dim=0)
    # if not directed:
    #     new_edges = torch.cat([new_edges, new_edges.flip(0)], dim=1)

    # new_edges = new_edges.numpy()
    # edge_list = np.concatenate((edge_list, new_edges), axis=0)

    if ano_n_label is None:
        y_n_outlier = torch.zeros(x.shape[0], dtype=torch.long)
    else:
        y_n_outlier = ano_n_label
    y_n_outlier[src_n_idx] = 1

    return x.numpy(), edge_list, y_n_outlier

def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """

    data = tuple(map(tuple, data))

    n = max(max(user, item) for user, item in data)  # Get size of matrix
    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[user - 1][item - 1] = 1  # Convert to 0-based index.
        matrix[item - 1][user - 1] = 1  # Convert to 0-based index.
    return matrix

def processEdges(fake_edges, data):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    # b:list->set
    # Time cost rate is proportional to the size

    idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)

    tmp = fake_edges[idx_fake]
    tmp[:, [0, 1]] = tmp[:, [1, 0]]

    fake_edges[idx_fake] = tmp

    idx_remove_dups = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)

    fake_edges = fake_edges[idx_remove_dups]
    a = fake_edges.tolist()
    b = data.tolist()
    c = []

    for i in a:
        if i not in b:
            c.append(i)
    fake_edges = np.array(c)
    return fake_edges