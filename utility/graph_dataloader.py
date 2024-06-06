import json
import warnings
import torch
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch_geometric.transforms as T

from typing import Callable, Optional
from torch_sparse import coalesce, transpose
from torch_geometric.io import read_npz
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.datasets import WikiCS, StochasticBlockModelDataset, RandomPartitionGraphDataset
from torch_geometric_signed_directed.data import load_directed_real_data, DirectedData
from torch_geometric_signed_directed.utils import node_class_split


class Amazon(InMemoryDataset):
    r"""The Amazon Computers and Amazon Photo networks from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper.
    Nodes represent goods and edges represent that two goods are frequently
    bought together.
    Given product reviews as bag-of-words node features, the task is to
    map goods to their respective product category.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Computers"`,
            :obj:`"Photo"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - Computers
              - 13,752
              - 491,722
              - 767
              - 10
            * - Photo
              - 7,650
              - 238,162
              - 745
              - 8
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['computers', 'photo']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.lower(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.lower(), 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'amazon_electronics_{self.name.lower()}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name.capitalize()}()'

class CoraML(InMemoryDataset):
    r"""Data loader for the Cora_ML data set used in the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, 
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None,
                 is_undirected: Optional[bool] = None):
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behaviour.")
            is_undirected = True
        self.is_undirected = is_undirected
        self.url = (
            'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/cora_ml.npz')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['cora_ml.npz']

    @property
    def processed_file_names(self):
        return ['cora_ml.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        with np.load(self.raw_dir + '/cora_ml.npz', allow_pickle=True) as loader:
            loader = dict(loader)
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                      loader['attr_indptr']), shape=loader['attr_shape'])
            labels = loader.get('labels')

        coo = adj.tocoo()
        values = torch.from_numpy(coo.data).float()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        features = torch.from_numpy(features.todense()).float()
        labels = torch.from_numpy(labels).long()
        if self.is_undirected:
            indices = to_undirected(indices, num_nodes=features.size(0))
        data = Data(x=features, edge_index=indices,
                    edge_weight=values, y=labels)
        data = node_class_split(data, train_size_per_class=20, val_size=500)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

class CiteSeerDir(InMemoryDataset):
    r"""Data loader for the CiteSeer data set used in the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, 
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None,
                 is_undirected: Optional[bool] = None):
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behaviour.")
            is_undirected = True
        self.is_undirected = is_undirected
        self.url = (
            'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/citeseer.npz')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['citeseer.npz']

    @property
    def processed_file_names(self):
        return ['citeseer.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        with np.load(self.raw_dir+'/citeseer.npz', allow_pickle=True) as loader:
            loader = dict(loader)
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                      loader['attr_indptr']), shape=loader['attr_shape'])
            labels = loader.get('labels')

        coo = adj.tocoo()
        values = torch.from_numpy(coo.data)
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        features = torch.from_numpy(features.todense()).float()
        labels = torch.from_numpy(labels).long()
        if self.is_undirected:
            indices = to_undirected(indices, num_nodes=features.size(0))
        data = Data(x=features, edge_index=indices,
                    edge_weight=values, y=labels)
        data = node_class_split(data, train_size_per_class=20, val_size=500)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

class Telegram(InMemoryDataset):

    url = 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/telegram'

    def __init__(self, root: str, 
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None,
                 is_undirected: Optional[bool] = None):
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behaviour.")
            is_undirected = True
        self.is_undirected = is_undirected
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['telegram_adj.npz', 'telegram_labels.npy']

    @property
    def processed_file_names(self):
        return ['telegram.pt']

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        A = sp.load_npz(self.raw_paths[0])
        label = np.load(self.raw_paths[1])
        rs = np.random.RandomState(seed=0)

        test_ratio = 0.2
        train_ratio = 0.6
        val_ratio = 1 - train_ratio - test_ratio

        label = torch.from_numpy(label).long()
        s_A = sp.csr_matrix(A)
        coo = s_A.tocoo()
        values = coo.data

        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        features = torch.from_numpy(
            rs.normal(0, 1.0, (s_A.shape[0], 1))).float()

        if self.is_undirected:
            indices = to_undirected(indices, num_nodes=features.size(0))
        data = Data(x=features, edge_index=indices,
                    edge_weight=torch.FloatTensor(values), y=label)
        data = node_class_split(
            data, train_size_per_class=train_ratio, val_size_per_class=val_ratio)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

class WikipediaNetwork(InMemoryDataset):
    r"""The code is modified from torch_geometric.datasets.WikipediaNetwork (v1.6.3)
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Chameleon"` :obj:`"Squirrel"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, 
                 name: str, 
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None,
                 is_undirected: Optional[bool] = None):
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behaviour.")
            is_undirected = True
        self.is_undirected = is_undirected
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']
        self.url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                    'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')
        super(WikipediaNetwork, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'] + [
            '{}_split_0.6_0.2_{}.npz'.format(self.name, i) for i in range(10)
        ]

    @property
    def processed_file_names(self):
        return 'wikipedianetwork.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.float)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        if self.is_undirected:
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

def data_loader(name):
    name = name.lower()

    # Undirected Graphs
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='./data/', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['computers', 'photo']:
        dataset = Amazon(root='./data/', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['film']:
        dataset = Actor(root='./data/film/', transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root='./data/', name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
    # Directed Graphs
    elif name == 'wikics':
        dataset = WikiCS(root='./data/wikics/', transform=T.NormalizeFeatures(), is_undirected=False)
        data = dataset[0]
        std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
        data.x = (data.x - mean) / std
    elif name == 'cora_ml':
        dataset = CoraML(root='./data/cora_ml/', transform=T.NormalizeFeatures(), is_undirected=False)
        data = dataset[0]
    elif name == 'citeseer_dir':
        dataset = CiteSeerDir(root='./data/citeseer_dir', transform=T.NormalizeFeatures(), is_undirected=False)
        data = dataset[0]
    elif name == 'telegram':
        dataset = Telegram(root='./data/telegram/', transform=T.NormalizeFeatures(), is_undirected=False)
        data = dataset[0]
    # Dataset 'crocodile' is not included.
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='./data/', name=name, transform=T.NormalizeFeatures(), is_undirected=False)
        data = dataset[0]
    else:
        raise ValueError(f'Dataset {name} is not included.')

    return dataset, data