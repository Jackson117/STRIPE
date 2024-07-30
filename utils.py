import torch
import numpy as np
import random
import time
import os
import scipy.sparse as sp
import scipy.io as sio

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import add_self_loops, to_dense_adj
from AnomalyGeneration import  anomaly_generation_dy
from torch_geometric.datasets import BitcoinOTC
from DeepWalk import DeepWalk
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

import networkx as nx

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class DynamicDataset(InMemoryDataset):
    def __init__(self, root, node_fea=None, edge_list=None, lbl=None, snap_size=None, anomaly_per=0.05,
                 mode='train',dataset='digg'):
        self.node_fea = node_fea
        self.edge_list = edge_list
        self.dataset = dataset
        self.snap_size = snap_size
        self.anomaly_per = anomaly_per
        self.mode = mode
        self.lbl = lbl
        
        super(DynamicDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['Dynamic_{}_{}_{}'.format(self.dataset, self.mode, self.anomaly_per) + '.pt']

    def process(self):
        dyn_graph_snapshots = []
        edge_num = self.edge_list.shape[0]
        snap_num = int(edge_num / self.snap_size) + 1
        print(self.mode + ' snapshot number: %d' % snap_num)

        for ii in range(snap_num):
            start_loc = ii * self.snap_size
            end_loc = (ii + 1) * (self.snap_size)

            if end_loc >= edge_num:
                end_loc = edge_num

            node_idx = np.unique(self.edge_list[start_loc: end_loc, :])
            mapping = {idx: i for i, idx in enumerate(node_idx)}
            x = torch.FloatTensor(self.node_fea[node_idx, :])

            lbl = torch.FloatTensor(self.lbl[node_idx])

            edge_snap = self.edge_list[start_loc: end_loc, :]
            edge_snap_mapped = []

            for edge in edge_snap:
                edge_snap_mapped.append([mapping[edge[0]], mapping[edge[1]]])

            edge_index = torch.transpose(torch.LongTensor(edge_snap_mapped), 1, 0)

            data = Data(x, edge_index, y=lbl, n_idx=node_idx, num_nodes=node_idx.shape[0])
            dyn_graph_snapshots.append(data)

        torch.save(self.collate(dyn_graph_snapshots), self.processed_paths[0])
        del dyn_graph_snapshots

def preprocessDataset(dataset):
    print('Preprocess dataset: ' + dataset)
    t0 = time.time()
    if dataset in ['digg', 'uci']:
        edges = np.loadtxt(
            'datasets/{}/raw/{}.txt'.format(dataset, dataset),
            dtype=float,
            comments='%',
            delimiter=' ')
        edges = edges[:, 0:2].astype(dtype=int)
    elif dataset in ['btc_alpha', 'btc_otc']:
        if dataset == 'btc_alpha':
            file_name = 'datasets/{}/raw/{}'.format(dataset, 'soc-sign-bitcoinalpha.csv')
        elif dataset == 'btc_otc':
            file_name = 'datasets/{}/raw/{}'.format(dataset, 'soc-sign-bitcoinotc.csv')
        with open(file_name) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = np.array(edges)
        edges = edges[edges[:, 3].argsort()]
        edges = edges[:, 0:2].astype(dtype=int)

    for ii in range(len(edges)):
        x0 = edges[ii][0]
        x1 = edges[ii][1]
        if x0 > x1:
            edges[ii][0] = x1
            edges[ii][1] = x0

    edges = edges[np.nonzero([x[0] != x[1] for x in edges])].tolist()
    aa, idx = np.unique(edges, return_index=True, axis=0)
    edges = np.array(edges)
    edges = edges[np.sort(idx)]

    vertexs, edges = np.unique(edges, return_inverse=True)
    edges = np.reshape(edges, [-1, 2])
    print('vertex:', len(vertexs), ' edge: ', len(edges))
    os.makedirs('datasets/{}/processed/'.format(dataset))
    np.savetxt(
        'datasets/{}/processed/{}.txt'.format(dataset,dataset),
        X=edges,
        delimiter=' ',
        comments='%',
        fmt='%d')
    print('Preprocess finished! Time: %.2f s' % (time.time() - t0))


def generate_dynamic_data(dataset, anomaly_per, mode):
    if dataset == 'bitcoinOTC':
        return generate_bitcoinotc(dataset, anomaly_per, mode)

    file = np.load('./datasets/{}.npz'.format(dataset))
    adjs = file['adjs']
    attmats = file['attmats']
    labels = file['labels']

    graph_list = anomaly_generation_dy(adjs, attmats, anomaly_per=anomaly_per, dataset=dataset, mode=mode)

    return graph_list

def generate_bitcoinotc(dataset, anomaly_per, mode):
    bitcoin = BitcoinOTC('./datasets/tg_data/bitcoinOTC')

    edge_list = [d.edge_index for d in bitcoin]
    edge_list = add_self_loops(torch.concat(edge_list, dim=1), num_nodes=bitcoin[0].num_nodes)[0]
    edge_list = torch.transpose(edge_list, 1, 0).numpy().tolist()
    fea, _, _ = generate_node_fea(edges=edge_list, embd_size=32)
    feas = [fea.reshape(fea.shape[0], 1, fea.shape[1]) for i in range(len(bitcoin))]
    feas = np.concatenate(feas, axis=1)
    print('Feature shape via DeepWalk: ', feas.shape)

    adjs = [to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes) for d in bitcoin]
    adjs = torch.concat(adjs, dim=0).numpy()
    print('Adjacency matrix shape: ', adjs.shape)

    graph_list = anomaly_generation_dy(adjs, feas, anomaly_per=anomaly_per, dataset=dataset, mode=mode)

    return graph_list



def load_mat(dataset):
    data = sio.loadmat("./mat_dataset/{}.mat".format(dataset))

    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    edge_index, edge_attr = to_edge_index(adj)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr)

    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)

    ano_labels = np.squeeze(np.array(label))

    num_node = label.shape[0]
    all_idx = list(range(num_node))
    random.shuffle(all_idx)

    ano_n_label = torch.LongTensor(ano_labels)
    feature = torch.FloatTensor(attr.todense())
    tg_data = Data(feature, edge_index, edge_attr, y=ano_n_label, num_nodes=num_node)

    return tg_data

def generate_node_fea(edges, embd_size):
    # Positional node feature
    nx_g = nx.from_edgelist(edges)
    deepwalk = DeepWalk(window_size=10, embedding_size=embd_size, walk_length=40, walks_per_node=5)
    walks = deepwalk.get_walks(nx_g)
    pos_fea = deepwalk.compute_embeddings(walks).vectors # node_num * embd_size ndarray

    # Structural node feature
    pr_score = list(nx.pagerank(nx_g).values())
    # node_num * embd_size ndarray
    stu_fea = np.broadcast_to(np.array(pr_score, dtype=float).reshape(len(pr_score),1), (len(pr_score), embd_size))

    # Temporal node feature
    tmp_fea = 0 # compute tmp_fea during datalist construction

    return pos_fea, stu_fea, tmp_fea


def to_edge_index(adj):
    if isinstance(adj, torch.Tensor):
        row, col, value = adj.to_sparse_coo().indices()[0], adj.to_sparse_coo().indices()[1], \
                        adj.to_sparse_coo().values()

    elif isinstance(adj, sp.csr_matrix):
        row, col, value = adj.tocoo().row, adj.tocoo().col, \
                          adj.tocoo().data
        row, col, value = torch.tensor(row, dtype=torch.long), torch.tensor(col, dtype=torch.long), \
                          torch.tensor(value, dtype=torch.float)
    else:
        raise RuntimeError("adj has to be either torch.sparse_csr_matrix or scipy.sparse.csr_matrix.")
    if value is None:
        value = torch.ones(row.size(0), device=row.device)

    return torch.stack([row, col], dim=0), value

def process_timestamps(edges, timestamps):
    min_time = np.min(timestamps)
    max_time = np.max(timestamps)

    period = (max_time - min_time) / 10
    # timestamps = (timestamps - min_time) / (max_time - min_time) * 100
    timestamps = np.sort(timestamps)
    tmp_time = min_time + period
    lis=[]
    for i, t in enumerate(timestamps):
        if t > tmp_time:
            lis.append(i)
            tmp_time += period

    print(lis)

def fea_reshape_3d(x, n_idx, batch, node_num):
    snap_num = torch.max(batch).item() + 1
    all_node_fea = torch.zeros((snap_num, node_num, x.size(1)), dtype=torch.float).to(x.get_device())
    for i, id in enumerate(n_idx):
        all_node_fea[i, id, :] = all_node_fea[i, id, :] + x[batch==i]

    n_idx = np.unique(np.concatenate(n_idx, axis=0))
    batch_node_fea = all_node_fea[:, n_idx, :]

    return batch_node_fea

def fea_reshape_2d(x, n_idx, batch, node_num):
    n_idx = np.concatenate(n_idx, axis=0)
    n_idx_u = np.unique(n_idx)
    n_idx = torch.from_numpy(n_idx).to(x.get_device())

    snap_num = x.size(0)
    all_node_fea = torch.zeros((snap_num, node_num, x.size(-1)), dtype=torch.float).to(x.get_device())

    all_node_fea[:, n_idx_u, :] = all_node_fea[:, n_idx_u, :] + x

    # x = all_node_fea.view(all_node_fea.size(1), -1)   # x.size(): N*(T*D)
    x = torch.mean(all_node_fea, dim=0) # x.size(): N * D

    n_idx = torch.broadcast_to(torch.unsqueeze(n_idx, 1), (n_idx.size(0), x.size(-1)))

    output = torch.gather(x, dim=0, index=n_idx)

    # del all_node_fea

    return output

def accuracy(preds, labels):
    # preds = torch.round(nn.Sigmoid()(logits))

    AUC = roc_auc_score(labels, preds)

    preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
    preds = np.where(preds >= 0.5, 1, 0)
    recall = recall_score(labels, preds, average='macro')
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')

    return recall, macro_f1, AUC, acc, precision

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def main():
    edges = np.loadtxt(
        'datasets/digg/digg.txt',
        dtype=float,
        comments='%',
        delimiter=' '
    )
    timestamps = edges[:, 3]
    process_timestamps(None, timestamps=timestamps)

if __name__ == '__main__':
    main()
