#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:Huggingface-BiLSTM-crf.py
@time:2022/09/22/23：20
"""
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
# import transformers
# from transformers import BertTokenizer, BertTokenizerFast, BertModel
# from datasets import load_metric
import numpy as np
from transformers import BertModel, AdamW, BertTokenizer
from torchcrf import CRF

data_dir = r'D:\大论文\开题\MyNER\test.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class Model(nn.Module):

    def __init__(self, tag_num, max_length):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        config = self.bert.config
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=config.hidden_size, hidden_size=config.hidden_size // 2,
                            batch_first=True)
        self.crf = CRF(tag_num)
        self.fc = nn.Linear(config.hidden_size, tag_num)

    def forward(self, x, y):
        with torch.no_grad():
            bert_output = self.bert(input_ids=x.input_ids, attention_mask=x.attention_mask, token_type_ids=x.token_type_ids)[0]
        lstm_output, _ = self.lstm(bert_output)  # (1,30,768)
        fc_output = self.fc(lstm_output)  # (1,30,7)
        # fc_output -> (seq_length, batch_size, n tags) y -> (seq_length, batch_size)
        print('fc_output', fc_output)
        print('y', y)
        loss = self.crf(fc_output, y)
        tag = self.crf.decode(fc_output)
        return loss, tag


if __name__ == '__main__':
    # 1.训练参数
    epoches = 50
    max_length = 30
    # 2.数据读取
    #  ---测试数据-----------------------------------------------------------------
    x = ["我 和 小 明 今 天 去 了 北 京".split(),
         "普 京 在  昨 天 进 攻 了 乌 克 拉 ， 造 成 了 大 量 人 员 的 伤 亡".split()
         ]
    y = ["O O B-PER I-PER O O O O B-LOC I-LOC".split(),
         "B-PER I-PER O O O O O O B-LOC I-LOC I-LOC O O O O O O O O O O O".split()
         ]

    tag_to_ix = {"B-PER": 0, "I-PER": 1, "O": 2, "[CLS]": 3, "[SEP]": 4, "B-LOC": 5, "I-LOC": 6}
    # ---
    # ---实验数据-----------------------------------------------------------------
    # x, y = read_data(data_dir)
    # unique_tags = set(tag for doc in y for tag in doc)
    # tag_to_ix = {tag: id for id, tag in enumerate(unique_tags)}
    # # tag_to_ix = {'B-Met': 0, 'B-Cau': 1, 'O': 2, 'B-Phe': 3, 'I-Phe': 4, 'I-Met': 5, 'I-Cau': 6}
    # # -------------------------------------------------------------------------
    labels = []
    for label in y:
        r = [tag_to_ix[x] for x in label]
        if len(r) < max_length:
            r += [tag_to_ix['O']] * (max_length - len(r))
        labels.append(r)
    # 3.模型构建
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # input_ids,attention_mask,token_type_ids
    tokenizer_result = tokenizer.encode_plus(x[1],
                                             return_token_type_ids=True,
                                             return_attention_mask=True, return_tensors='pt',
                                             padding='max_length', max_length=max_length)
    # training
    model = Model(len(tag_to_ix), max_length)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    model.train()
    for i in range(epoches):
        for j in range(len(x)):
            loss, _ = model(tokenizer_result, torch.tensor(labels[1]).unsqueeze(dim=0))
            loss = abs(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'loss : {loss}')
    # evaluating
    model.eval()
    for i in range(len(x)):
        with torch.no_grad():
            _, tag = model(tokenizer_result, torch.tensor(labels[1]).unsqueeze(dim=0))
        print(f' ori tag: {labels[1]} \n predict tag : {tag}')
    # save model
    # torch.save(model.state_dict(),f'model.pt')
