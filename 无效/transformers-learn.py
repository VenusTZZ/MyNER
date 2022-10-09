#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:transformers-learn.py
@time:2022/10/03/15：58
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from transformers import BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        # [xxx,yyy]
        self.data = data
        # [0,1]
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.label[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=30, truncation=True)
        input_ids = inputs.input_ids.squeeze(0)
        token_type_ids = inputs.token_type_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return input_ids, token_type_ids, attention_mask, label

    def __len__(self):
        return len(self.data)


class MyModelConv(nn.Module):
    def __init__(self):
        super(MyModelConv, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # x:[batch,channel,width,height]
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(2, 768))
        self.linear = nn.Linear(27, 3)

    def forward(self, input_ids, token_type_ids, attention_mask):
        batch = input_ids.size(0)
        # 1  output=[batch,seq,hidden_size]
        output = self.bert(input_ids, token_type_ids, attention_mask).last_hidden_state
        # 2  output=[batch,channel=1,width=seq_length,height=hidden_size]
        output = output.unsqueeze(1)
        # 3  output=[batch,channel=3,width,height]
        # width=((seq_length+2*padding-kernel_size)/stride)+1=((10+2*0-2)/1)+1=9
        # height=(768-768)/1+1=1
        output = self.conv1(output)  # output=[batch,channel=3,width=9,height=1]
        output = output.view(batch, -1)  # output=[batch,3*9*1]
        output = self.linear(output)
        return output


class MyModelLstm(nn.Module):
    def __init__(self):
        super(MyModelLstm, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(1024, 3)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # x=[batch,seq]
        # x=[batch,seq,dim]
        batch = input_ids.size(0)
        # 1  output=[batch,seq,hidden_size]
        output = self.bert(input_ids, token_type_ids, attention_mask).last_hidden_state
        # output, h_n, c_n = self.lstm(output)
        output, _ = self.lstm(output)
        # print(output.shape)
        output = output[:, -1, :]  # output=[batch,hidden_size*2] => bidirectional=True
        return output


if __name__ == '__main__':
    pass
    data, label = [], []
    with open("../example.train", encoding="utf-8") as f:
        for line in f.readlines():
            print(line.strip("/n"))
            print("xx")
            if len(line) > 0:
                lineData, lineLabel = line.strip().split("	")
                data.append(lineData)
                label.append(lineLabel)
            else:
                continue
    datasets = MyDataSet(data, label)
    dataloader = Data.DataLoader(datasets, batch_size=2, shuffle=True)
    # 创建模型
    model = MyModelConv().to(device)
    # model = MyModelLstm().to(device)
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # 训练
    for epoch in range(10):
        for input_ids, token_type_ids, attention_mask, label in dataloader:
            # 把数据放到GPU上
            input_ids, token_type_ids, attention_mask, label = input_ids.to(device), token_type_ids.to(device), attention_mask.to(
                device), label.to(device)
            # 前向传播
            pred = model(input_ids, token_type_ids, attention_mask)
            # 计算损失
            loss = loss_fn(pred, label)
            print(loss.item())
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()
