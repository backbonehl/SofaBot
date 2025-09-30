# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
from src_trainner import SourceTrainer
from tag_trainner import TargetTrainer
from config import parse_train_args, init_logger
import argparse
import torch
import numpy as np
import random

args = parse_train_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                                description="BotDetection")).parse_args()

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

# Logger
logger = init_logger(args)

tgt_trainner = TargetTrainer(args, logger)
tgt_trainner.train_procedure()
