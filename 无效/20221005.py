#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:20221005.py
@time:2022/10/05/15：34
"""

from datasets import list_datasets
from datasets import load_dataset
from pathlib import Path
import re
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    # raw_docs = file_path.read_text().strip()
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        # tags.append('[CLS]')
        for line in doc.split('\n'):
            # token, tag = line.split('\t')
            token, tag = line.split(' ')
            tokens.append(token)
            tags.append(tag)
        # tags.append('[SEP]')
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


texts, tags = read_wnut('../data/train.txt')
train_texts, train_tags = read_wnut('../data/train.txt')
test_texts, test_tags = read_wnut('../data/test.txt')
# val_texts, val_tags = read_wnut('./data/val.txt')
val_texts, val_tags = read_wnut('../data/dev.txt')


unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
label_list = list(unique_tags)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

import numpy as np

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    #print(labels)
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)


#
# print(train_tags[0])
# print(train_encodings['input_ids'][0])

import torch
import torch.utils.data as Data


class WNUTDataset(Data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_encodings.pop("offset_mapping")
val_encodings.pop("offset_mapping")
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)
# print(train_dataset[0]['input_ids'].size())
# print(train_dataset[0]['labels'].size())

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(unique_tags)).to(device)

# from datasets import load_dataset, load_metric
#
# metric = load_metric("seqeval")
#
# import numpy as np
#
#
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#
#     # Remove ignored index (special tokens)
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


training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=10,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,  # the instantiated   Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    # compute_metrics=compute_metrics
)

trainer.train().to(device)

# trainer.evaluate()
