# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import torch.nn as nn
import torch
from models.init_feature_embedding import InitFeatureEmbeddingLayer
from models.gnn_extractor_layers import GNNExtractorLayer
from models.classifiers import Classifier
from models.adapt_layers import *


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask, samples


def augmentation(features_1, adj_1, features_2, adj_2, args, training):
    # view 1
    mask_1, _ = get_feat_mask(features_1, args.maskfeat_rate_1)
    features_1 = features_1 * (1 - mask_1)
    if not args.sparse:
        adj_1 = F.dropout(adj_1, p=args.dropedge_rate_1, training=training)
    else:
        adj_1.edata['w'] = F.dropout(adj_1.edata['w'], p=args.dropedge_rate_1, training=training)

    # # view 2
    mask_2, _ = get_feat_mask(features_1, args.maskfeat_rate_2)
    features_2 = features_2 * (1 - mask_2)
    if not args.sparse:
        adj_2 = F.dropout(adj_2, p=args.dropedge_rate_2, training=training)
    else:
        adj_2.edata['w'] = F.dropout(adj_2.edata['w'], p=args.dropedge_rate_2, training=training)

    return features_1, adj_1, features_2, adj_2


class AdaptBotDetector(nn.Module):
    """
    无监督学习目标域知识
    """
    def __init__(self, args):
        super(AdaptBotDetector, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.init_feature_size = args.category_feature_size + args.numerical_feature_size + args.textual_feature_size

        # 图拆分
        self.edge_discriminator = EdgeDiscriminator(self.init_feature_size, args.alpha, args.sparse)
        # 图对比
        self.gcl_module = GCL(nlayers=2, nlayers_proj=1, in_dim=self.init_feature_size, emb_dim=args.emb_dim,
                              proj_dim=args.hidden_dim, dropout=args.dropout, sparse=args.sparse,
                              batch_size=args.cl_batch_size)

        # self.init_weights()

    def forward(self, user_features, edge_index, dataset, training=True):

        adj_1, adj_2, weights_lp, weights_hp = self.edge_discriminator(user_features, edge_index)  # 每次使用的是最新融合的特征
        # 数据增强，mask特征，mask边
        if training:
            x_1, adj_1, x_2, adj_2 = augmentation(user_features, adj_1, user_features, adj_2, self.args, training=True)
        else:
            x_1, adj_1, x_2, adj_2 = user_features, adj_1, user_features, adj_2

        # 在两个图上获取h1_1和h1_2
        self.gcl_module.set_mask_knn(user_features.cpu(), k=self.args.k, dataset=dataset)
        h1_1, h1_2, cl_loss = self.gcl_module(x_1, adj_1, x_2, adj_2)

        return h1_1, h1_2, cl_loss

    def get_homo_hete(self, user_features, edge_index):
        adj_1, adj_2, _, _ = self.edge_discriminator(user_features, edge_index)
        return adj_1, adj_2

    # def init_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             nn.init.kaiming_normal_(module.weight.data)
    #             if module.bias is not None:
    #                 module.bias.data.zero_()
    #         elif isinstance(module, nn.LayerNorm):
    #             module.weight.data.fill_(1.0)
    #             module.bias.data.zero_()


class SRCBotDetector(nn.Module):
    """
    保留源域知识
    """
    def __init__(self, args):
        super(SRCBotDetector, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.feature_size = [args.category_feature_size, args.numerical_feature_size, args.textual_feature_size]
        self.hidden_dims = [int(args.hidden_dim + i*(args.feature_dim - args.hidden_dim)/args.gnn_layers)
                            for i in range(args.gnn_layers)]
        self.hidden_dims.append(args.feature_dim)
        self.hidden_dims = self.hidden_dims[::-1]
        # 初始特征融合
        self.Encoder = InitFeatureEmbeddingLayer(args.feature_dim, self.feature_size, args.agg_strategy, args.dropout)
        # GNN融合
        self.Extractor = GNNExtractorLayer(args.feature_dim, self.hidden_dims, args.gnn_model,
                                           dropout=args.dropout, head_num=args.head_nums, batchNorm=args.batch_norm)
        # if args.with_extra:
            # 外部特征融合
        self.f1 = nn.Linear(args.extra_feature_size, args.hidden_dim)
        self.f2 = nn.Linear(args.hidden_dim*2, args.hidden_dim)

        self.Classifier = Classifier(args.hidden_dim, args.classifier_out)

        # self.init_weights()

    def forward(self, user_features, features_size, edge_index, edge_type, extra_feature):

        cat_feat, num_feat, text_feat = torch.split(user_features, features_size, dim=1)
        x_0 = self.Encoder([cat_feat, num_feat, text_feat])
        h_0 = self.Extractor(x_0, edge_index, edge_type)

        if self.args.with_extra == 1:
            h_1 = self.f1(extra_feature)
            z = self.f2(torch.cat((h_0, h_1), dim=1))
        elif self.args.with_extra == 2:
            z = self.f1(extra_feature)
            z = nn.PReLU().cuda()(z)
        else:
            z = h_0
        if self.args.with_activate == 1:
            z = nn.PReLU().cuda()(z)

        prob = self.Classifier(z)

        if prob.shape[1] == 1:
            pred_labels = (prob > 0.5).long()
        else:
            pred_labels = torch.argmax(prob, dim=1)

        return prob, pred_labels, h_0
    #
    # def init_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             nn.init.kaiming_normal_(module.weight.data)
    #             if module.bias is not None:
    #                 module.bias.data.zero_()
    #         elif isinstance(module, nn.LayerNorm):
    #             module.weight.data.fill_(1.0)
    #             module.bias.data.zero_()


class BotClassifier(nn.Module):
    """
    特征融合加分类器
    """
    def __init__(self, args, hidden_dim, extra_feature_size):
        super(BotClassifier, self).__init__()
        # 外部特征融合
        self.f1 = nn.Linear(extra_feature_size, hidden_dim)
        self.f2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.Classifier = Classifier(args.hidden_dim*2, args.classifier_out)

    def forward(self, x, extra_feature):

        x_e = self.f1(extra_feature)
        z = self.f2(torch.cat((x, x_e), dim=1))

        prob = self.Classifier(z)

        if prob.shape[1] == 1:
            pred_labels = (prob > 0.5).long()
        else:
            pred_labels = torch.argmax(prob, dim=1)

        return prob, pred_labels
