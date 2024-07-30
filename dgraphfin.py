from typing import Optional, Callable, List
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import degree


def read_dgraphfin(folder):
    print('read_dgraphfin')
    names = ['dgraphfin.npz']
    items = [np.load(folder+'/'+name) for name in names]
    
    x = items[0]['x']
    y = items[0]['y'].reshape(-1,1)
    edge_index = items[0]['edge_index']
    edge_type = items[0]['edge_type']
    edge_timestamp = items[0]['edge_timestamp']
    train_mask = items[0]['train_mask']
    valid_mask = items[0]['valid_mask']
    test_mask = items[0]['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    ano_n_label = torch.squeeze(torch.tensor(y, dtype=torch.int64))
    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.float)
    train_mask = torch.tensor(train_mask, dtype=torch.int64)
    valid_mask = torch.tensor(valid_mask, dtype=torch.int64)
    test_mask = torch.tensor(test_mask, dtype=torch.int64)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type)
    labeled_mask = torch.ones(ano_n_label.size(), dtype=torch.bool)
    # ano_n_label[ano_n_label==2] = 0
    # ano_n_label[ano_n_label==3] = 0

    labeled_mask[ano_n_label==2] = 0
    labeled_mask[ano_n_label==3] = 0
    data.labeled_mask = labeled_mask
    data.edge_timestamp = edge_timestamp
    data.y = ano_n_label

    return data

class DGraphFin(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"dgraphfin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ''

    def __init__(self, root: str, name: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['dgraphfin.npz']
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass
#         for name in self.raw_file_names:
#             download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_dgraphfin(self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'

def analyze_dgraph(data):
    print(data)
    row = data.edge_index[1, :]
    labeled_mask = data.labeled_mask
    y = data.y
    all_deg = degree(row, num_nodes=data.num_nodes)
    fraud_deg = torch.mean(all_deg[y == 1])
    norm_deg = torch.mean(all_deg[y == 0])
    print('fraud_avg_deg', fraud_deg)
    print('norm_avg_deg', norm_deg)

    edge_stamp = data.edge_timestamp
    max_edge_stamp = edge_stamp.max()
    min_edge_stamp = edge_stamp.min()
    print('max_edge_stamp', max_edge_stamp)
    print('min_edge_stamp', min_edge_stamp)

    # stamp_cnt = {}
    # for e in edge_stamp:
    #     if stamp_cnt.get(e) is None:
    #         stamp_cnt[e] = 1
    #     else:
    #         stamp_cnt[e] += 1
    # np.savetxt('./edge_stamp.txt', edge_stamp)

    edge_stamp = edge_stamp
    edge_index = data.edge_index
    snap_size = 20
    snap_num = int(np.ceil(max_edge_stamp/snap_size))
    snap_edges = []
    fraud_deg_l, norm_deg_l = [], []
    for i in range(snap_num):
        low_bound = i * snap_size
        high_bound = (i + 1) * snap_size if (i + 1) * snap_size < max_edge_stamp else max_edge_stamp
        snap_edge = edge_index[:, np.array(edge_stamp>=low_bound) & np.array(edge_stamp<high_bound)]
        snap_edges.append(snap_edge)
        print('Snap #{} size: {}'.format(i, snap_edge.shape))
        node_index = np.unique(snap_edge[1, :])

        snap_deg = all_deg[node_index]
        fraud_deg = torch.mean(snap_deg[(y==1)[node_index]])
        norm_deg = torch.mean(snap_deg[(y==0)[node_index]])
        if i == 40:
            np.savetxt('./toyexamp/fraud_deg_snap_{}'.format(i), snap_deg[(y==1)[node_index]])
            np.savetxt('./toyexamp/norm_deg_snap_{}'.format(i), snap_deg[(y==0)[node_index]])
        fraud_deg_l.append(fraud_deg.item())
        norm_deg_l.append(norm_deg.item())
        print('fraud_{}_deg'.format(i), fraud_deg)
        print('norm_{}_deg'.format(i), norm_deg)
        print('\t')

        np.savetxt('./toyexamp/fraud_deg_out.txt', fraud_deg_l)
        np.savetxt('./toyexamp/norm_deg_out.txt', norm_deg_l)
