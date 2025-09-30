# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=1, with_active=False):
        """
        :param input_dim: 输入特征维度，最后一层特征维度
        :param output_dim:  1表示单分类任务，否则表示转为多分类任务
        :param with_active: 是否使用sigmoid或者softmax，取决于外面使用的损失函数是否自带
        """
        super(Classifier, self).__init__()
        self.fc = nn.Sequential()
        self.fc.append(nn.Linear(input_dim, output_dim))
        # self.fc = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
        #                         nn.ReLU(),
        #                         nn.Linear(input_dim // 2, output_dim))
        if with_active:
            if output_dim == 1:
                self.fc.append(nn.Sigmoid())
            else:
                self.fc.append(nn.Softmax())

    def forward(self, x):
        return self.fc(x)
