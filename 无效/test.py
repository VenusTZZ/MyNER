#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:test.py
@time:2022/10/06/20：06
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torchcrf import CRF
from transformers import BertModel


class MyModelLstm(nn.Module):
    def __init__(self):
        super(MyModelLstm, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, batch_first=True, bidirectional=True, num_layers=2)
        self.linear = nn.Linear(1024, 6)
        # self.crf = CRF(7)

    def forward(self, x, y):
        output = self.bert(x).last_hidden_state
        output, _ = self.lstm(output)
        # output = self.linear(output)
        # output = self.crf(output, y)
        return output


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    pass
    model = MyModelLstm().to('cuda')
    print(model.state_dict())
    print(get_parameter_number(model))
