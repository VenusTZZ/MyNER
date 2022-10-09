#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:从故障描述到检修建议的推荐.py
@time:2022/09/18/16：18
@description: 从故障描述到检修建议的推荐,讲现象输入神经网络，输出检修建议的multi-hot编码,根据编码找到检修建议的词
把输入通过一个bert可以理解为通过一个word2vec,都是对输入进行特征提取
# --------------------------------------------------
# 可以先在控制台测试一下
# import torch
# import transformers
# from transformers import BertTokenizer, BertModel
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertModel.from_pretrained('bert-base-chinese')
# inputs = tokenizer("雪邦山风电场XBS-A3-032风电机组（EN87-1500型）齿轮箱中速齿底部外壳开裂。", return_tensors="pt", padding='max_length', max_length=30)
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)
# --------------------------------------------------
"""
# 1.
# 把所有的现象进行分词，去除停用词，得到所有的词汇表。
# 根据输入词汇表对输入进行multi-hot编码，得到输入向量。
# 2.
# 把所有的检修方法进行分词，去除停用词，得到所有的词汇表。
# 3.
# 根据检修方法词汇表对检修方法进行multi-hot编码，得到检修方法向量。
# 把向量送入Roberta(或者别的bert衍生模型？)模型，得到一个向量表示。
# 把学习到的向量送入卷积网络，得到卷积结果。卷积结果也是一种multi-hot编码的方式
# 然后看multi-hot编码的结果,其中那个位置的数字为1，则代表推荐这个检修方法；如果为0，则不推荐这个检修方法。
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from transformers import BertTokenizer, BertTokenizerFast, BertModel
from datasets import load_metric
import numpy as np
from torchcrf import CRF
# 参数设置
epoches = 50
max_length = 64
data_dir = r'D:\大论文\开题\MyNER\train.txt'
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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.label[idx]
        # train_encodings = self.tokenizer(text,
        #                                  is_split_into_words=True,
        #                                  return_offsets_mapping=True,
        #                                  padding=True,
        #                                  truncation=True,
        #                                  max_length=512)
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)
        input_ids = inputs.input_ids.squeeze(0)
        token_type_ids = inputs.token_type_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return input_ids, token_type_ids, attention_mask, torch.tensor(label)

    def __len__(self):
        return len(self.data)


class MyModelConv(nn.Module):
    def __init__(self):
        super(MyModelConv, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # x:[batch,channel,width,height]
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(2, 768))
        self.linear = nn.Linear(27, 6)

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
        self.config = self.bert.config
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(1024, 6)
        # self.crf = CRF(6)

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
        output = self.linear(output)
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
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    # print(labels)
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # 创建全由-100组成的矩阵
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        if len(doc_labels) >= 510:  # 防止异常
            doc_labels = doc_labels[:510]
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#
#     # 不要管-100那些，剔除掉
#     true_predictions = [
#         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#
#     results = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }


if __name__ == '__main__':

    # --读取数据
    print('1.读取数据')
    train_texts, train_tags = read_data(data_dir)
    # --标签转换为数字
    # print('标签转换为数字')
    # unique_tags={'B-Met', 'B-Cau', 'O', 'B-Phe', 'I-Phe', 'I-Met', 'I-Cau'}
    # tag2id={'B-Met': 0, 'B-Cau': 1, 'O': 2, 'B-Phe': 3, 'I-Phe': 4, 'I-Met': 5, 'I-Cau': 6}
    # id2tag={0: 'B-Met', 1: 'B-Cau', 2: 'O', 3: 'B-Phe', 4: 'I-Phe', 5: 'I-Met', 6: 'I-Cau'}
    # label_list=['B-Met', 'B-Cau', 'O', 'B-Phe', 'I-Phe', 'I-Met', 'I-Cau']
    unique_tags = set(tag for doc in train_tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    label_list = list(unique_tags)
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
    # print(datasets.data[0])
    # print(datasets.label[0])
    # 判断数据和标签是否一一对应
    # for i in range(len(datasets.data)):
    #     print(len(datasets.data[i]))
    #     print(len(datasets.label[i]))
    #     if len(datasets.data[i]) != len(datasets.label[i]):
    #         print(i)

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
            # 前向传播
            pred = model(input_ids, token_type_ids, attention_mask)
            print('pred', pred.shape)
            print('label', label.shape)
            # 计算损失
            loss = loss_fn(pred, label)
            print(loss.item())
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()
    # --评估标准
    # metric = load_metric("seqeval")
