# -*- coding: utf-8 -*-
# @Author  : 
# @Time    : 
# @Function:
import json
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


class SourceTrainer:
    def __init__(self, args, logger):
        super(SourceTrainer, self).__init__()

        self.logger = logger
        self.args = args
        self.device = torch.device(args.device)

        self.baseline_bot_model = SRCBotDetector(args)
        init_weights(self.baseline_bot_model)
        self.baseline_bot_model.to(self.device)
        self.extra_bot_model = AdaptBotDetector(args)
        init_weights(self.extra_bot_model)
        self.extra_bot_model.to(self.device)

        self.src_bm_opt = optim.Adam(self.baseline_bot_model.parameters(), lr=args.source_lr0,
                                     weight_decay=args.source_weight_decay0)
        self.src_em_opt = optim.Adam(self.extra_bot_model.parameters(), lr=args.source_lr1,
                                     weight_decay=args.source_weight_decay1)

        self.source_data = Dataset(args.source_dataset, args.batch_size, args.gnn_model)
        self.source_data.data.to(device=self.device)
        self.target_data = Dataset(args.target_dataset, args.batch_size, args.gnn_model)
        self.target_data.data.to(device=self.device)

        # For supervised training on source domain  
        if self.args.classifier_out == 1:
            self.CrossEntropy = nn.BCEWithLogitsLoss()
        else:
            self.CrossEntropy = nn.CrossEntropyLoss()

    def train_source(self, baseline_model, extra_model, source_data, epoch):
        
        baseline_model.eval()
        for k, v in baseline_model.named_parameters():
            v.requires_grad = False
        extra_model.train()
        for k, v in extra_model.named_parameters():
            v.requires_grad = True

        for t in range(self.args.loop_0):
            self.src_em_opt.zero_grad()

            _, _, cl_loss = extra_model(source_data.data.x, source_data.data.edge_index, source_data.dataset_name)

            cl_loss.backward()
            self.src_em_opt.step()
            self.logger.info(
                "=====> [source training][extra model training] epoch: {}, loop: {}, cl loss: {}".format(epoch, t,
                                                                                                         cl_loss.item()))
        self.logger.info("\n")
        
        baseline_model.train()
        for k, v in baseline_model.named_parameters():
            v.requires_grad = True
        extra_model.eval()
        for k, v in extra_model.named_parameters():
            v.requires_grad = False
        h1_1, h1_2, _ = extra_model(source_data.data.x, source_data.data.edge_index, source_data.dataset_name)
        extra_feature = torch.cat((h1_1, h1_2), dim=1)

        probs, pred_labels, h_0 = baseline_model(source_data.data.x,
                                                 source_data.features_size,
                                                 source_data.data.edge_index,
                                                 source_data.data.edge_type,
                                                 extra_feature)

        labels = source_data.data.y[source_data.data.train_mask]
        probs = probs[source_data.data.train_mask]
        pred_labels = pred_labels[source_data.data.train_mask]

        
        ce_loss = self.CrossEntropy(probs, labels)
        self.src_bm_opt.zero_grad()
        ce_loss.backward()
        self.src_bm_opt.step()
        self.logger.info(
            "=====> [source training][extra model training] epoch: {}, ce loss: {}".format(epoch, ce_loss.item()))

        metric, result = cal_metrics(labels.cpu(), pred_labels.cpu())

        return ce_loss.item(), metric, result

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
        # training
        best_val_loss = 1e8

        src_dataset = self.args.source_dataset
        src_flag = "gnn-{}_ly-{}_ft-{}_dim-{}_drp-{}_bz-{}_lr0-{}_lr1-{}_w0-{}_w1-{}_{}".format(
            self.args.gnn_model,
            self.args.gnn_layers,
            self.args.agg_strategy,
            self.args.hidden_dim,
            self.args.dropout,
            self.args.batch_size,
            self.args.source_lr0,
            self.args.source_lr1,
            self.args.source_weight_decay0,
            self.args.source_weight_decay1,
            time.strftime("%Y%m%d-%H%M%S", time.localtime())
        )
        src_best_dir = os.path.join(self.args.best_source_dir, src_dataset, src_flag)
        src_best_model_0 = os.path.join(src_best_dir, "model_b.pt")
        src_best_model_1 = os.path.join(src_best_dir, "model_e.pt")
        src_best_result = os.path.join(src_best_dir, "result.json")
        src_best_config = os.path.join(src_best_dir, "config.json")

        if not os.path.exists(src_best_model_0) or not os.path.exists(src_best_model_1): 
            if not os.path.exists(src_best_dir):
                os.makedirs(src_best_dir)
            self.logger.info("========================Train On Source Datasets========================")
            best_val_acc = 0.0
            best_val_result = None
            for epoch in range(self.args.source_epochs):
                train_loss, train_accuracy, _ = self.train_source(self.baseline_bot_model, self.extra_bot_model,
                                                                  self.source_data, epoch)
                val_loss, val_acc, val_res = self.test(self.baseline_bot_model, self.extra_bot_model,
                                                       self.source_data, stage='valid')
                test_loss, test_acc, test_res = self.test(self.baseline_bot_model, self.extra_bot_model,
                                                       self.source_data, stage='test')
                self.logger.info(
                    'Epoch\t{:03d}\ttrain:acc\t{:.6f}\tce_loss\t{:.6f}\tvalid:acc\t{:.6f}\tce_loss\t{:.6f}\ttest:acc\t{:.6f}\tce_loss\t{:.6f}'.format(
                        epoch, train_accuracy, train_loss, val_acc, val_loss, test_acc, test_loss))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_val_result = val_res
                    save_model(src_best_model_0, self.baseline_bot_model)
                    save_model(src_best_model_1, self.extra_bot_model)
                    save_result(src_best_result, best_val_result)
                    save_result(src_best_config, vars(self.args))
            self.logger.info('Best valid acc\t{:.6f}\t Best valid loss\t{:.6f}'.format(best_val_acc, best_val_loss))
            self.logger.info(vars(self.args))
            self.logger.info(best_val_result)

        self.baseline_bot_model = load_model(src_best_model_0, self.baseline_bot_model, self.args.device)
        self.extra_bot_model = load_model(src_best_model_1, self.extra_bot_model, self.args.device)
        self.logger.info("Load Best Source Model For Source Test: {}".format(src_best_model_0))

        _, test_accuracy, test_res = self.test(self.baseline_bot_model, self.extra_bot_model,
                                               self.source_data, stage='test')
        self.logger.info(vars(self.args))
        self.logger.info('Test On Source Data acc\t{:.6f}'.format(test_accuracy))
        self.logger.info(test_res)
        self.logger.info("===========================================================================")
        self.logger.info("\n\n")

        self.logger.info("========================Test On Target Datasets without UDA================")
        _, test_test_acc, test_test_res = self.test(self.baseline_bot_model, self.extra_bot_model, self.target_data,
                                                    stage="test")  
        self.logger.info("Test Target Baseline (Test Data): {}".format(test_test_acc))
        self.logger.info(test_test_res)
        self.logger.info("============================================================================")
        self.logger.info("\n\n")

        self.logger.info(self.args)

    def test_procedure(self, model_dir):

        src_dataset = self.args.source_dataset
        src_best_dir = os.path.join(self.args.best_source_dir, src_dataset, model_dir)
        src_best_model_0 = os.path.join(src_best_dir, "model_b.pt")
        src_best_model_1 = os.path.join(src_best_dir, "model_e.pt")
        src_best_result = os.path.join(src_best_dir, "result.json")
        src_best_config = os.path.join(src_best_dir, "config.json")
        with open(src_best_config, "r") as f:
            config_info = json.load(f)
            self.logger.info(config_info)
        with open(src_best_result, "r") as f:
            val_res_info = json.load(f)
            self.logger.info(val_res_info)
        self.baseline_bot_model = load_model(src_best_model_0, self.baseline_bot_model, self.args.device)
        self.extra_bot_model = load_model(src_best_model_1, self.extra_bot_model, self.args.device)
        self.logger.info("Load Best Source Model For Source Test: {}".format(src_best_model_0))

        _, test_accuracy, test_res = self.test(self.baseline_bot_model, self.extra_bot_model,
                                               self.source_data, stage='test')
        self.logger.info('Test On Source Data acc\t{:.6f}'.format(test_accuracy))
        self.logger.info(test_res)
        self.logger.info("\n\n")
        _, test_test_acc, test_test_res = self.test(self.baseline_bot_model, self.extra_bot_model, self.target_data, stage="test")

        self.logger.info("Test Target Baseline (Test Data): {}".format(test_test_acc))
        self.logger.info(test_test_res)
        self.logger.info("============================================================================")
        self.logger.info("\n\n\n\n")


