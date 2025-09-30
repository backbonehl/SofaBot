# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import torch
import torch.nn as nn


class InitFeatureEmbeddingLayer(nn.Module):
    """
    Node initial feature fusion (attributes, text, etc.), including MLP, Self-Attention, etc.
    """
    def __init__(self, feature_dim, features_size, strategy='mlp', dropout=0.2):
        """
        :param feature_dim: 
        :param features_size: [cat, num, tweet]
        :param dropout:
        :param strategy: mlp, att
        """
        super(InitFeatureEmbeddingLayer, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.strategy = strategy
        self.features_size = features_size

        if self.strategy == 'mlp':
            self.features_linear = nn.ModuleList()
            for i, f_size in enumerate(features_size):
                self.features_linear.append(
                    nn.Sequential(
                        nn.Linear(f_size, feature_dim // len(features_size)),
                        self.activation
                    )
                )
            self.init_feature_linear = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                self.activation
            )
        elif self.strategy == 'att':
            self.features_linear = nn.ModuleList()
            self.feature_self_attention = nn.MultiheadAttention(feature_dim//len(features_size), num_heads=4)
            self.feature_self_fusion = nn.Linear(feature_dim, feature_dim)
            for i, f_size in enumerate(features_size):
                self.features_linear.append(
                    nn.Sequential(
                        nn.Linear(f_size, feature_dim // len(features_size)),
                        self.activation
                    )
                )
        else:
            raise ValueError("Not Support Strategy {} for Initial Features".format(self.strategy))

        # self.init_weights()

    def forward(self, all_features):
        assert len(all_features) == len(self.features_size)
        if self.strategy == 'mlp':
            return self.init_feature_with_mlp(all_features)
        else:
            return self.init_feature_with_att(all_features)

    def init_feature_with_mlp(self, all_features):
        """
        :param all_features: [f1, f2], f1=batch_size*dim_size
        :return:
        """
        feats = []
        for i, feat in enumerate(all_features):
            tmp_feat = self.features_linear[i](feat)
            feats.append(tmp_feat)
        input_feats = torch.cat(feats, dim=1)
        output_features = self.dropout(self.init_feature_linear(input_feats))
        return output_features

    def init_feature_with_att(self, all_features):
        feats = []
        for i, feat in enumerate(all_features):
            tmp_feat = self.features_linear[i](feat)
            feats.append(tmp_feat.unsqueeze(0))

        input_feats = torch.cat(feats, dim=0)

        output_features, weights = self.feature_self_attention(input_feats, input_feats, input_feats)
        output_features = self.dropout(self.feature_self_fusion(output_features.permute(1, 0, 2).reshape(input_feats.shape[1], -1)))
        return output_features
