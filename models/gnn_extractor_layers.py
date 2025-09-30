# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv, TransformerConv, HGTConv
from models.blocks import SimpleHGN, RGTLayer


class GNNExtractorLayer(nn.Module):
    """
    GNN-based feature extraction
    """
    def __init__(self, feature_dim, hidden_dims, gnn_model='gcn', dropout=0.3, head_num=4, relation_num=2, batchNorm=False):
        """
        :param feature_dim: 
        :param
        :param hidden_dims: [128, 64, 32]
        :param dropout:
        :param head_num: Need in GAT and Graph Transformer 
        :param relation_num：
        :param batchNorm: 
        """
        super(GNNExtractorLayer, self).__init__()

        assert len(hidden_dims) >= 2 and feature_dim == hidden_dims[0]
        self.layer_count = len(hidden_dims) - 1
        self.batch_norm = batchNorm
        self.dropout_rate = dropout
        self.output_dim = hidden_dims[-1]
        self.gnn_model = gnn_model
        self.conv_layers, self.activate_layer, self.norm_layer = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.hidden_dims = hidden_dims

        for i in range(self.layer_count):
            if self.gnn_model == "GCN":
                self.conv_layers.append(
                    GCNConv(in_channels=self.hidden_dims[i], out_channels=self.hidden_dims[i + 1])
                )
            elif self.gnn_model == "SAGE":
                self.conv_layers.append(
                    SAGEConv(in_channels=self.hidden_dims[i], out_channels=self.hidden_dims[i + 1])
                )
            elif self.gnn_model == "GAT":
                self.conv_layers.append(
                    GATConv(in_channels=self.hidden_dims[i],
                            out_channels=self.hidden_dims[i + 1] // head_num,
                            heads=head_num, dropout=self.dropout_rate)
                )
            elif self.gnn_model == "RGCN":
                self.conv_layers.append(
                    RGCNConv(in_channels=self.hidden_dims[i], out_channels=self.hidden_dims[i + 1],
                             num_relations=relation_num)
                )
            elif self.gnn_model == "SimpleHGT":
                self.conv_layers.append(
                    SimpleHGN(in_channels=self.hidden_dims[i],
                              out_channels=self.hidden_dims[i + 1],
                              num_edge_type=relation_num,
                              rel_dim=100,
                              beta=0.05)
                )
            elif self.gnn_model == "GT":
                self.conv_layers.append(
                    TransformerConv(in_channels=self.hidden_dims[i],
                                    out_channels=self.hidden_dims[i + 1],
                                    heads=head_num,
                                    dropout=dropout,
                                    concat=False)
                )
            elif self.gnn_model == "RGT":
                self.conv_layers.append(
                    RGTLayer(in_channel=self.hidden_dims[i],
                             out_channel=self.hidden_dims[i + 1],
                             num_edge_type=relation_num,
                             trans_heads=head_num,
                             semantic_head=head_num,
                             dropout=dropout)
                )
            elif self.gnn_model == 'HGT':
                self.conv_layers.append(
                    HGTConv(in_channels=self.hidden_dims[i],
                            out_channels=self.hidden_dims[i + 1],
                            metadata=(['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')]))
                )

            self.activate_layer.append(nn.PReLU())
            self.norm_layer.append(nn.BatchNorm1d(self.hidden_dims[i + 1]))

            # if batchNorm:
            #     self.conv_layers.append(nn.Sequential(
            #         RGTLayer(in_channel=self.hidden_dims[i],
            #                  out_channel=self.hidden_dims[i + 1],
            #                  num_edge_type=relation_num,
            #                  trans_heads=head_num,
            #                  semantic_head=head_num,
            #                  dropout=dropout),
            #         nn.BatchNorm1d(self.hidden_dims[i + 1]),
            #         nn.PReLU()
            #     ))
            # else:
            #     self.conv_layers.append(nn.Sequential(
            #         RGTLayer(in_channel=self.hidden_dims[i],
            #                  out_channel=self.hidden_dims[i + 1],
            #                  num_edge_type=relation_num,
            #                  trans_heads=head_num,
            #                  semantic_head=head_num,
            #                  dropout=dropout),
            #         nn.PReLU()
            #     ))

        # self.init_weights()

    def forward(self, x, edge_idx, edge_type=None, mask=None):
        if self.gnn_model in ["RGCN", "SimpleHGT", "RGT"]:  # 考虑多类型关系
            for i in range(self.layer_count):
                if self.batch_norm:
                    x = self.activate_layer[i](
                        self.norm_layer[i](
                            self.conv_layers[i](x, edge_idx, edge_type)
                        )
                    )
                else:
                    x = self.activate_layer[i](
                        self.conv_layers[i](x, edge_idx, edge_type)
                    )
        elif self.gnn_model in ['HGT']:
            for i in range(self.layer_count):
                following_edge_index = edge_idx[:, edge_type == 0]
                follower_edge_index = edge_idx[:, edge_type == 1]
                x_dict = {"user": x}
                edge_index_dict = {('user', 'follower', 'user'): follower_edge_index,
                                   ('user', 'following', 'user'): following_edge_index}
                if self.batch_norm:
                    x = self.activate_layer[i](
                        self.norm_layer[i](
                            self.conv_layers[i](x_dict, edge_index_dict)["user"]
                        )
                    )
                else:
                    x = self.activate_layer[i](
                        self.conv_layers[i](x_dict, edge_index_dict)["user"]
                    )
        else:
            for i in range(self.layer_count):
                if self.batch_norm:
                    x = self.activate_layer[i](
                        self.norm_layer[i](
                            self.conv_layers[i](x, edge_idx)
                        )
                    )
                else:
                    x = self.activate_layer[i](
                        self.conv_layers[i](x, edge_idx)
                    )

        if mask:
            x = x[mask]
        return x
