# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import NeighborLoader


def load_graph(dataset_name):

    edge_index = torch.load(r"./processed/edge_index.pt".format(dataset_name))
    edge_type = torch.load(r"./processed/edge_type.pt".format(dataset_name))

    # if dataset_name in ['TW-20', "MGTAB"]:
    #     edge_index = torch.load(r"./processed/edge_index_homo.pt".format(dataset_name))
    #     edge_type = torch.load(r"./processed/edge_type_homo.pt".format(dataset_name))
    return edge_index, edge_type


def load_split_index(dataset_name):
    train_idx = torch.load(r"./processed/train_idx.pt".format(dataset_name))
    val_idx = torch.load(r"./processed/valid_idx.pt".format(dataset_name))
    test_idx = torch.load(r"./processed/test_idx.pt".format(dataset_name))
    return train_idx, val_idx, test_idx


def load_labels(dataset_name):
    labels = torch.load(r"./processed/label.pt".format(dataset_name))
    return labels


def load_features(dataset_name):
    # des_tensor = torch.load('./des_tensor.pt'.format(dataset_name))
    tweets_tensor = torch.load('./processed/tweets_tensor.pt'.format(dataset_name))
    num_prop = torch.load('./processed/num_properties_tensor.pt'.format(dataset_name))
    category_prop = torch.load('./processed/cat_properties_tensor.pt'.format(dataset_name))
    return category_prop, num_prop, tweets_tensor        # , des_tensor,


class Dataset:
    def __init__(self, dataset_name, batch_size, model_name=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.model_name = model_name

        self.train_idx, self.val_idx, self.test_idx = load_split_index(dataset_name)
        self.labels = load_labels(dataset_name)
        self.edge_index, self.edge_type = load_graph(dataset_name)

        self.category_prop, self.num_prop, self.tweets_tensor = load_features(dataset_name)
        # Concatenate features in the order of cat, num, text
        self.features = torch.cat((self.category_prop, self.num_prop, self.tweets_tensor), dim=1)

        self.features_size = [self.category_prop.size(1), self.num_prop.size(1), self.tweets_tensor.size(1)]

        train_mask = index_to_mask(self.train_idx, size=self.features.size(0))
        val_mask = index_to_mask(self.val_idx, size=self.features.size(0))
        test_mask = index_to_mask(self.test_idx, size=self.features.size(0))

        self.data = Data(x=self.features,
                         y=self.labels,
                         edge_index=self.edge_index,
                         edge_type=self.edge_type,
                         train_mask=train_mask,
                         val_mask=val_mask,
                         test_mask=test_mask)
        if dataset_name in ['MGTAB', 'TW-22'] and model_name in ['RGT', 'SimpleHGT']:
            # Random sampling of edges
            all_index = torch.cat((self.train_idx, self.val_idx, self.test_idx), dim=0)
            loader = NeighborLoader(self.data,
                                    num_neighbors=[256] * 2,
                                    batch_size=all_index.size(0),
                                    input_nodes=all_index,
                                    shuffle=False)
            for batch in loader:
                self.data = batch
                break

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = True
    return mask

