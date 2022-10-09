#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:20221004.py
@time:2022/10/04/20：30
"""
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertTokenizerFast
import numpy as np
from torchcrf import CRF

# 参数设置
epoches = 50
max_length = 128
data_dir = r'example.train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        # [xxx,yyy]
        self.data = data
        # [0,1]
        self.label = label
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.label[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)
        # inputs = self.tokenizer(train_texts,
        #                         is_split_into_words=True,
        #                         return_offsets_mapping=True,
        #                         padding=True,
        #                         truncation=True,
        #                         max_length=128)
        print(inputs)
        input_ids = inputs.input_ids.squeeze(0)
        token_type_ids = inputs.token_type_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return input_ids, token_type_ids, attention_mask, torch.tensor(label)

    def __len__(self):
        return len(self.data)


class MyModelLstm(nn.Module):
    def __init__(self):
        super(MyModelLstm, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.config = self.bert.config
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, batch_first=True,
                            bidirectional=True)
        # self.layer_norm = nn.LayerNorm(512 * 2)
        self.classifier = nn.Linear(1024, 6)
        self.crf = CRF(6, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # x=[batch,seq]
        # x=[batch,seq,dim]
        # print('input_ids\n', input_ids.size())
        # print('token_type_ids\n', token_type_ids.size())
        # print('attention_mask\n', attention_mask.size())
        batch = input_ids.size(0)
        # 1  output=[batch,seq,hidden_size]
        output = self.bert(input_ids, token_type_ids, attention_mask).last_hidden_state
        # output, h_n, c_n = self.lstm(output)
        output, _ = self.lstm(output)
        # print('1output\n', output.size())
        # print(output.shape)
        # output = output[:, -1, :]  # output=[batch,hidden_size*2] => bidirectional=True
        output = self.classifier(output)
        # print('2output\n', output.size())
        output = self.crf.decode(output)
        output = torch.tensor(output)
        # print('3output\n', output.size())
        return output


def read_data(file_path):
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding='UTF-8').strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        tags.append('[CLS]')
        for line in doc.split('\n'):
            token, tag = line.split('   ')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tags.append('[SEP]')
        print(tags)
        tag_docs.append(tags)
    return token_docs, tag_docs


# def encode_tags(tags, encodings):
#     labels = [[tag2id[tag] for tag in doc] for doc in tags]
#     # print(labels)
#     encoded_labels = []
#     for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
#         # 创建全由-100组成的矩阵
#         doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
#         arr_offset = np.array(doc_offset)
#         # set labels whose first offset position is 0 and the second is not 0
#         # if len(doc_labels) >= 510:  # 防止异常
#         #     doc_labels = doc_labels[:510]
#         doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
#         encoded_labels.append(doc_enc_labels.tolist())
#
#     return encoded_labels


# --读取数据
print('1.读取数据')
train_texts, train_tags = read_data(data_dir)
# --标签转换为数字
# print('标签转换为数字')
unique_tags = set(tag for doc in train_tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
label_list = list(unique_tags)
print('unique_tags\n', unique_tags)
print('tag2id\n', tag2id)
print('id2tag\n', id2tag)
print('label_list\n', label_list)
labels = []
for label in train_tags:
    r = [tag2id[x] for x in label]
    if len(r) < max_length:
        r += [tag2id['O']] * (max_length - len(r))
    labels.append(r)
# print(labels)
datasets = MyDataSet(train_texts, labels)
dataloader = Data.DataLoader(datasets, batch_size=2, shuffle=True)
# 查看数据
# print(dataloader.dataset[0])
print(datasets.data[0])
print(datasets.label[0])

# --创建模型
print('2.创建模型')
# model = MyModelConv().to(device)
model = MyModelLstm().to(device)
# print(model.parameters())
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# --训练
print('3.训练')
for epoch in range(epoches):
    for input_ids, token_type_ids, attention_mask, label in dataloader.dataset:
        # 把数据放到GPU上
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        print('input_ids\n', input_ids[1])
        # print(token_type_ids.size())
        # print(attention_mask.size())
        # print(label.size())
        # --
        # 前向传播
        pred = model(input_ids, token_type_ids, attention_mask).to(device)
        # print(pred[0])
        # print(label)
        # 计算损失
        loss = loss_fn(pred, label)
        print(loss.item())
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()
        # --
