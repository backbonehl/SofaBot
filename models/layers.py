# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import torch
import torch.nn as nn


class FeatureFusionLayer(nn.Module):
    """
    Feature fusion layer
    """
    def __init__(self, input_dims, output_dim, strategy='mlp', dropout=0.2):
        """
        :param input_dims: List of input feature dimensions
        :param output_dim: 
        :param dropout:
        :param strategy: mlp, att
        """
        super(FeatureFusionLayer, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.strategy = strategy

        self.input_dims = input_dims
        self.output_dim = output_dim

        if self.strategy == 'mlp':
            self.init_feature_linear = nn.Sequential(
                nn.Linear(sum(input_dims), output_dim),
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

    def fusion_with_mlp(self, inputs):
        """
        :param inputs: [f1, f2], f1=batch_size*dim_size
        :return:
        """
        input_feats = torch.cat(inputs, dim=1)
        output_features = self.dropout(self.init_feature_linear(input_feats))
        return output_features

    def fusion_with_att(self, all_features):
        feats = []
        for i, feat in enumerate(all_features):
            tmp_feat = self.features_linear[i](feat)
            feats.append(tmp_feat.unsqueeze(0))

        input_feats = torch.cat(feats, dim=0)
        output_features, weights = self.feature_self_attention(input_feats, input_feats, input_feats)
        output_features = self.dropout(self.feature_self_fusion(output_features.permute(1, 0, 2).reshape(input_feats.shape[1], -1)))
        return output_features