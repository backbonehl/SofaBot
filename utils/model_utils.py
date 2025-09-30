# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import logging
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
import json
import torch.nn as nn


def save_model(best_path, model):
    torch.save({'model_state_dict': model.state_dict()}, best_path)


def save_feature(best_path, features):
    torch.save(features, best_path)


def save_result(best_path, result):
    with open(best_path, "w") as f:
        json.dump(result, f)


def load_model(best_path, model, device):
    device = torch.device(device)
    state_dict = torch.load(best_path, map_location=device)["model_state_dict"]
    model.load_state_dict(state_dict)
    return model


def cos_distance(input1, input2):
    norm1 = torch.norm(input1, dim=-1)
    norm2 = torch.norm(input2, dim=-1)

    norm1 = torch.unsqueeze(norm1, dim=1)
    norm2 = torch.unsqueeze(norm2, dim=0)

    cos_matrix = torch.matmul(input1, input2.t())

    cos_matrix /= norm1
    cos_matrix /= norm2

    return cos_matrix


def generate_one_hot_label(labels):
    num_labels = torch.max(labels).item() + 1
    num_nodes = labels.shape[0]
    label_onehot = torch.zeros((num_nodes, num_labels)).cuda()
    label_onehot = F.one_hot(labels, num_labels).float().squeeze(1)

    return label_onehot


def generate_normalized_adjs(adj, D_isqrt):
    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def process_adj(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return adj, deg_inv_sqrt


def cal_metrics(y_true, y_pred, compare='acc'):
    res = {
        'acc': accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average='micro'),
        "f1_macro": f1_score(y_true, y_pred, average='macro')
    }
    try:
        res['f1_human'], res['f1_bot'] = f1_score(y_true, y_pred, average=None)
    except ValueError:
        res['f1_human'], res['f1_bot'] = -1, -1

    res['recall'] = recall_score(y_true, y_pred)
    try:
        res['recall_human'], res['recall_bot'] = recall_score(y_true, y_pred, average=None)
    except ValueError:
        res['recall_human'], res['recall_bot'] = -1, -1

    res['precision'] = precision_score(y_true, y_pred)
    try:
        res['precision_human'], res['precision_bot'] = precision_score(y_true, y_pred, average=None)
    except ValueError:
        res['precision_human'], res['precision_bot'] = -1, -1

    return res[compare], res


def init_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
