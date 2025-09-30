# -*- coding: utf-8 -*-
# @Author  : 
# @Time    : 
# @Function:
import os.path
import torch
from models.models import *
import torch.nn as nn
from utils.model_utils import load_model, save_model, save_result, cal_metrics, init_weights
from sf_uda.soga import SOGATrainer
from sf_uda.cta2 import CTATrainer
import torch.optim as optim
from utils.dataset import Dataset
from tqdm import tqdm
import time
from sf_uda.NCE_utilies import RandomWalker, NegativeSampler, Entropy
from torch_geometric.utils.convert import to_networkx


class NeighborPropagate(MessagePassing):
    def __init__(self, aggr: str = 'mean', **kwargs,):
        kwargs['aggr'] = aggr if aggr != 'lstm' else None
        super().__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, size=size)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, x: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        return scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)


class TargetTrainer:
    def __init__(self, args, logger):
        super(TargetTrainer, self).__init__()

        self.negative_samples_neigh = None
        self.positive_samples_neigh = None
        self.center_nodes_neigh = None
        self.negative_samples_struct = None
        self.positive_samples_struct = None
        self.center_nodes_struct = None
        self.num_negative_samples = 5
        self.num_positive_samples = 2
        self.num_target_nodes = -1

        self.mem_cls = None
        self.probs_th = torch.tensor([0.90, 0.5])
        self.probs_th_b = 1-0.9

        self.neigh_prop = NeighborPropagate()

        self.logger = logger
        self.args = args
        self.device = torch.device(args.device)

        self.baseline_bot_model = SRCBotDetector(args)
        init_weights(self.baseline_bot_model)
        self.baseline_bot_model.to(self.device)
        self.extra_bot_model = AdaptBotDetector(args)
        init_weights(self.extra_bot_model)
        self.extra_bot_model.to(self.device)

        self.tgt_bm_opt = optim.Adam(self.baseline_bot_model.parameters(), lr=args.target_lr0,
                                     weight_decay=args.target_weight_decay0)
        self.tgt_em_opt = optim.Adam(self.extra_bot_model.parameters(), lr=args.target_lr1,
                                     weight_decay=args.target_weight_decay1)

        # self.source_data = Dataset(args.source_dataset, args.batch_size, args.gnn_model)
        # self.source_data.data.to(device=self.device)
        self.target_data = Dataset(args.target_dataset, args.batch_size, args.gnn_model)
        self.target_data.data.to(device=self.device)

        if self.args.classifier_out == 1:
            self.CrossEntropy = nn.BCEWithLogitsLoss()
        else:
            self.CrossEntropy = nn.CrossEntropyLoss()

    def generate_pseudo_labels_0(self, target_data, probs):
        return target_data.data.y, target_data.data.train_mask

    def generate_pseudo_labels_self_train(self, target_data, probs):
        """
        :param target_data:
        :param probs
        :return:
        """
        softmax_out = F.softmax(probs, dim=1)

        if self.mem_cls is None:
            self.mem_cls = softmax_out.clone()
        else:
            self.mem_cls = (1.0 - self.args.momentum) * self.mem_cls + self.args.momentum * softmax_out.clone()

        max_class_probs, _ = torch.max(self.mem_cls, dim=0)
        self.logger.info(max_class_probs)
        final_mask = torch.zeros(probs.size(0), dtype=torch.bool).to(self.args.device)
        num_target_classes = len(np.unique(target_data.data.y.cpu().numpy()))
        self.probs_th += (self.probs_th_b / self.args.target_epochs)
        for i in range(num_target_classes):
            class_mask = (self.mem_cls[:, i] > self.probs_th[i] * max_class_probs[i])
            final_mask |= class_mask

        pseudo_prob = self.mem_cls
        _, pseudo_pred = torch.max(pseudo_prob, dim=1)

        pseudo_acc, pseudo_res = cal_metrics(target_data.data.y.cpu(), pseudo_pred.cpu())
        self.logger.info("[target training][pseudo labeling]pseudo_acc: {}\t\t{}".format(pseudo_acc, pseudo_res))

        selected_pseudo_pred = pseudo_pred[final_mask]
        self.logger.info(
            "[target training][pseudo labeling]Human: {}, Bot: {}".format((selected_pseudo_pred == 0).sum().item(),
                                                                          (selected_pseudo_pred == 1).sum().item()))
        return pseudo_pred, final_mask

    def generate_pseudo_labels_homo_hete(self, target_data, probs, homo_adj, hete_adj):
        """
        Use two views to generate pseudo labels
        """
        softmax_out = F.softmax(probs, dim=1)
        _, model_pred_labels = torch.max(softmax_out, dim=1)
        edge_index_0 = homo_adj.nonzero(as_tuple=False).t()
        edge_index_1 = hete_adj.nonzero(as_tuple=False).t()
        homo_pseudo_prob = self.neigh_prop(softmax_out, edge_index_0)
        _, homo_pred_labels = torch.max(homo_pseudo_prob, dim=1)
        hete_pseudo_prob = self.neigh_prop(softmax_out, edge_index_1)
        _, hete_pred_labels = torch.max(hete_pseudo_prob, dim=1)

        model_pseudo_acc, _ = cal_metrics(target_data.data.y.cpu(), model_pred_labels.cpu())
        homo_pseudo_acc, _ = cal_metrics(target_data.data.y.cpu(), homo_pred_labels.cpu())
        hete_pseudo_acc, _ = cal_metrics(target_data.data.y.cpu(), hete_pred_labels.cpu())

        self.logger.info(
            "[target training][pseudo labeling]model_pseudo_acc: {}; homo_pseudo_acc: {}; hete_pseudo_acc: {}".format(
                model_pseudo_acc,
                homo_pseudo_acc,
                hete_pseudo_acc))
        return model_pred_labels, torch.ones(probs.size(0), dtype=torch.bool).to(self.args.device)

    def train_target(self, baseline_model, extra_model, target_data, epoch):
        baseline_model.eval()
        for k, v in baseline_model.named_parameters():
            v.requires_grad = False
        extra_model.train()
        for k, v in extra_model.named_parameters():
            v.requires_grad = True

        if self.args.with_extra:
            for t in range(self.args.loop_0):
                self.tgt_em_opt.zero_grad()

                _, _, cl_loss = extra_model(target_data.data.x, target_data.data.edge_index, target_data.dataset_name)

                cl_loss.backward()
                self.tgt_em_opt.step()
                self.logger.info(
                    "=====> [target training][extra model training] epoch: {}, loop: {}, cl loss: {}".format(epoch, t,
                                                                                                             cl_loss.item()))
        self.logger.info("\n")
        baseline_model.train()
        for k, v in baseline_model.named_parameters():
            v.requires_grad = True
        extra_model.eval()
        for k, v in extra_model.named_parameters():
            v.requires_grad = False
        h1_1, h1_2, _ = extra_model(target_data.data.x, target_data.data.edge_index, target_data.dataset_name)
        extra_feature = torch.cat((h1_1, h1_2), dim=1)

        probs, pred_labels, h_0 = baseline_model(target_data.data.x,
                                                 target_data.features_size,
                                                 target_data.data.edge_index,
                                                 target_data.data.edge_type,
                                                 extra_feature)
        true_labels = target_data.data.y

        im_loss = (self.ent(probs))         # - self.div(probs)
        NCE_loss_struct = self.NCE_loss(probs,
                                        self.center_nodes_struct,
                                        self.positive_samples_struct,
                                        self.negative_samples_struct)
        NCE_loss_neigh = self.NCE_loss(probs,
                                       self.center_nodes_neigh,
                                       self.positive_samples_neigh,
                                       self.negative_samples_neigh)
        loss = self.args.im_lambda*im_loss + self.args.struct_lambda * NCE_loss_struct + self.args.neigh_lambda * NCE_loss_neigh
        # ce_loss = self.CrossEntropy(probs, pseudo_labels)
        # loss = ce_loss

        self.tgt_bm_opt.zero_grad()
        loss.backward()
        self.tgt_bm_opt.step()
        self.logger.info(
            "=====> [target training][extra model training] epoch: {}, loss: {}".format(epoch, loss.item()))

        metric, result = cal_metrics(true_labels.cpu(), pred_labels.cpu())

        return loss.item(), metric, result

    @torch.no_grad()
    def test(self, baseline_model, extra_model, dataset, stage):
        baseline_model.eval()
        extra_model.eval()
        h1_1, h1_2, _ = extra_model(dataset.data.x, dataset.data.edge_index, dataset.dataset_name, training=False)
        extra_feature = torch.cat((h1_1, h1_2), dim=1)

        probs, pred_labels, h_0 = baseline_model(dataset.data.x,
                                                 dataset.features_size,
                                                 dataset.data.edge_index,
                                                 dataset.data.edge_type,
                                                 extra_feature)

        if stage == 'valid':
            probs = probs[dataset.data.val_mask]
            pred_labels = pred_labels[dataset.data.val_mask]
            labels = dataset.data.y[dataset.data.val_mask]
        elif stage == 'test':
            probs = probs[dataset.data.test_mask]
            pred_labels = pred_labels[dataset.data.test_mask]
            labels = dataset.data.y[dataset.data.test_mask]
        else:
            labels = dataset.data.y

        loss = self.CrossEntropy(probs, labels)
        metric, result = cal_metrics(labels.cpu(), pred_labels.cpu())
        return loss.item(), metric, result

    def train_procedure(self):

        self.logger.info(self.args)
        src_dataset = self.args.source_dataset
        src_flag = "{}_{}_drp-{}".format(
            src_dataset,
            self.args.gnn_model,
            self.args.dropout
        )
        src_best_dir = os.path.join(self.args.best_source_dir, src_dataset, src_flag)
        src_best_model_0 = os.path.join(src_best_dir, "model_b.pt")
        src_best_model_1 = os.path.join(src_best_dir, "model_e.pt")

        assert os.path.exists(src_best_model_0) and os.path.exists(src_best_model_1)

        self.baseline_bot_model = load_model(src_best_model_0, self.baseline_bot_model, self.args.device)
        self.logger.info("Load Best Source Model For Target Data: {}".format(src_best_model_0))
        if self.args.reset_model:
            init_weights(self.extra_bot_model)
            self.logger.info("Reset Model For Target Data")
        else:
            self.extra_bot_model = load_model(src_best_model_1, self.extra_bot_model, self.args.device)
            self.logger.info("Load Best Source Model For Target Data: {}".format(src_best_model_1))
        self.logger.info("\n\n")

        target_structure_data = self.target_data.data.clone()
        self.init_target(target_structure_data, self.target_data.data)

        target_dataset = self.args.target_dataset
        tgt_flag = "gnn-{}_ly-{}_ft-{}_dim-{}_drp-{}_bz-{}_lr0-{}_lr1-{}_w0-{}_w1-{}".format(
            self.args.gnn_model,
            self.args.gnn_layers,
            self.args.agg_strategy,
            self.args.hidden_dim,
            self.args.dropout,
            self.args.batch_size,
            self.args.target_lr0,
            self.args.target_lr1,
            self.args.target_weight_decay0,
            self.args.target_weight_decay1
        )
        tgt_best_dir = os.path.join(self.args.best_target_dir, "{}_{}".format(src_dataset, target_dataset), tgt_flag)
        if not os.path.exists(tgt_best_dir):
            os.makedirs(tgt_best_dir)
        tgt_best_model_0 = os.path.join(tgt_best_dir, "model_b.pt")
        tgt_best_model_1 = os.path.join(tgt_best_dir, "model_e.pt")
        tgt_best_result = os.path.join(tgt_best_dir, "result.json")
        tgt_best_config = os.path.join(tgt_best_dir, "config.json")

        best_valid_acc = 0.0
        best_valid_result = None

        _, test_test_acc, test_test_res = self.test(self.baseline_bot_model, self.extra_bot_model, self.target_data,
                                                    stage="test")

        self.logger.info("Test Target Baseline (Test Data): {}".format(test_test_acc))
        self.logger.info(test_test_res)
        self.logger.info("===========================================================================")
        self.logger.info("\n\n")

        for epoch in range(self.args.target_epochs):
            train_loss, train_acc, train_res = self.train_target(self.baseline_bot_model, self.extra_bot_model,
                                                                 self.target_data, epoch)
            val_loss, val_acc, val_result = self.test(self.baseline_bot_model, self.extra_bot_model, self.target_data,
                                                      stage="valid")
            # test_loss, test_acc, test_result = self.test(self.baseline_bot_model, self.extra_bot_model, self.target_data,
            #                                           stage="test")
            self.logger.info(
                'Epoch\t{:03d}\ttrain: acc\t{:.6f}, loss\t{:.6f}; valid: acc\t{:.6f}, loss\t{:.6f}; test: acc\t{:.6f}, loss\t{:.6f}'.format(
                    epoch,
                    train_acc,
                    train_loss,
                    val_acc,
                    val_loss,
                    val_acc, val_loss))
            self.logger.info(train_res)
            self.logger.info(val_result)
            self.logger.info(val_result)
            if val_acc >= best_valid_acc:
                best_valid_acc = val_acc
                best_valid_result = val_result
                save_model(tgt_best_model_0, self.baseline_bot_model)
                save_model(tgt_best_model_1, self.extra_bot_model)
                save_result(tgt_best_result, best_valid_result)
                save_result(tgt_best_config, vars(self.args))

        # self.logger.info(self.target_config)
        self.logger.info('Best test on valid data acc\t{:.6f}'.format(best_valid_acc))
        self.logger.info(best_valid_result)
        self.baseline_bot_model = load_model(tgt_best_model_0, self.baseline_bot_model, self.args.device)
        self.extra_bot_model = load_model(tgt_best_model_1, self.extra_bot_model, self.args.device)
        _, best_test_acc, best_test_result = self.test(self.baseline_bot_model, self.extra_bot_model, self.target_data,
                                                       stage="test")  
        self.logger.info('Best test on test data acc\t{:.6f}'.format(best_test_acc))
        self.logger.info(best_test_result)
        self.logger.info("===========================================================================")

        self.logger.info(self.args)

    def ent(self, probs):
        softmax_output = F.softmax(probs, dim=-1)
        entropy_loss = torch.mean(Entropy(softmax_output))
        return entropy_loss

    def div(self, probs):
        softmax_output = F.softmax(probs, dim=-1)
        mean_softmax_output = softmax_output.mean(dim=0)
        diversity_loss = torch.sum(-mean_softmax_output * torch.log(mean_softmax_output + 1e-8))
        return diversity_loss

    def NCE_loss(self, outputs, center_nodes, positive_samples, negative_samples):
        outputs = F.softmax(outputs, dim=-1)
        negative_embedding = F.embedding(negative_samples, outputs)
        positive_embedding = F.embedding(positive_samples, outputs)
        center_embedding = F.embedding(center_nodes, outputs)

        positive_embedding = positive_embedding.permute([0, 2, 1])
        positive_score = torch.bmm(center_embedding, positive_embedding).squeeze()
        exp_positive_score = torch.exp(positive_score).squeeze()

        negative_embedding = negative_embedding.permute([0, 2, 1])
        negative_score = torch.bmm(center_embedding, negative_embedding).squeeze()
        exp_negative_score = torch.exp(negative_score).squeeze()

        exp_negative_score = torch.sum(exp_negative_score, dim=1)

        loss = -torch.log(exp_positive_score / exp_negative_score)
        loss = loss.mean()

        return loss

    def init_target(self, graph_struct, graph_neigh):

        assert graph_struct.x.size(0) == graph_neigh.x.size(0)
        self.num_target_nodes = graph_struct.x.size(0)

        target_g_struct = to_networkx(graph_struct)
        target_g_neigh = to_networkx(graph_neigh)

        positive_sampler = RandomWalker(target_g_struct, p=0.25, q=2, use_rejection_sampling=1)
        negative_sampler = NegativeSampler(target_g_struct)
        self.center_nodes_struct, self.positive_samples_struct = self.generate_positive_samples(positive_sampler,
                                                                                                self.num_positive_samples)
        self.negative_samples_struct = self.generate_negative_samples(negative_sampler,
                                                                      self.num_target_nodes,
                                                                      self.num_negative_samples)
        sort_index = torch.argsort(self.center_nodes_struct.squeeze(1))
        self.center_nodes_struct = self.center_nodes_struct[sort_index]
        self.positive_samples_struct = self.positive_samples_struct[sort_index]
        self.negative_samples_struct = self.negative_samples_struct[sort_index]


        positive_sampler = RandomWalker(target_g_neigh, p=0.25, q=2, use_rejection_sampling=1)
        negative_sampler = NegativeSampler(target_g_neigh)
        self.center_nodes_neigh, self.positive_samples_neigh = self.generate_positive_samples(positive_sampler,
                                                                                              self.num_positive_samples)
        self.negative_samples_neigh = self.generate_negative_samples(negative_sampler,
                                                                     self.num_target_nodes,
                                                                     self.num_negative_samples)

        sort_index = torch.argsort(self.center_nodes_neigh.squeeze(1))
        self.center_nodes_neigh = self.center_nodes_neigh[sort_index]
        self.positive_samples_neigh = self.positive_samples_neigh[sort_index]
        self.negative_samples_neigh = self.negative_samples_neigh[sort_index]

    def generate_positive_samples(self, sampler, positive_nums):
        sampler.preprocess_transition_probs()
        positive_samples_t = sampler.simulate_walks(num_walks=1, walk_length=positive_nums, workers=1, verbose=1)
        for i in range(len(positive_samples_t)):
            if len(positive_samples_t[i]) != 2:
                positive_samples_t[i].append(positive_samples_t[i][0])

        samples = torch.tensor(positive_samples_t).cuda()

        center_nodes = torch.unsqueeze(samples[:, 0], dim=-1)
        positive_samples = torch.unsqueeze(samples[:, 1], dim=-1)

        return center_nodes, positive_samples

    def generate_negative_samples(self, sampler, nodes_nums, negative_nums):
        negative_samples = torch.tensor([sampler.sample() for _ in
                                         range(negative_nums * nodes_nums)]).view(
            [nodes_nums, negative_nums]).cuda()

        return negative_samples