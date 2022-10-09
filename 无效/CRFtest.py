#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:CRFtest.py
@time:2022/10/04/21：46
"""

import torch
import torch.nn as nn
import numpy as np
import random
from torchcrf import CRF
from torch.optim import Adam

seed = 100


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


num_tags = 5
model = CRF(num_tags, batch_first=True)  # 这里根据情况而定
seq_len = 3
batch_size = 50
seed_everything()
trainset = torch.randn(batch_size, seq_len, num_tags)  # features
traintags = (torch.rand([batch_size, seq_len]) * 4).floor().long()  # (batch_size, seq_len)
testset = torch.randn(5, seq_len, num_tags)  # features
testtags = (torch.rand([5, seq_len]) * 4).floor().long()  # (batch_size, seq_len)
print('trainset\n', trainset.size())
print('traintags\n', traintags.size())
print('testset\n', testset.size())
print('testtags\n', testtags.size())

# 训练阶段
for e in range(50):
    optimizer = Adam(model.parameters(), lr=0.05)
    model.train()
    optimizer.zero_grad()
    loss = -model(trainset, traintags)
    print('epoch{}: loss score is {}'.format(e, loss))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

# 测试阶段
model.eval()
loss = model(testset, testtags)
model.decode(testset)
