#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:Huggingface-NER.py
@time:2022/09/22/20：20
"""
# 导入相关库
import re
from pathlib import Path
from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from transformers import BertTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import torch
from datasets import load_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. 数据集读取
# 这里自己写了一个函数用来读取数据，data_dir改成自己的路径
data_dir = r'D:\大论文\开题\MyNER\test.txt'


def read_data(file_path):
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding='UTF-8').strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    # raw_docs = file_path.read_text().strip()
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


train_texts, train_tags = read_data(data_dir)
test_texts, test_tags = read_data(data_dir)
val_texts, val_tags = read_data(data_dir)

# unique_tags 代表有多少种标签，tag2id表示每种标签对应的id，id2tag表示每种id对应的标签。后面需要。
unique_tags = set(tag for doc in train_tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
label_list = list(unique_tags)

# 2.Bert Tokenizer
# is_split_into_words表示已经分词好了
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
train_encodings = tokenizer(train_texts,
                            is_split_into_words=True,
                            return_offsets_mapping=True,
                            padding=True,
                            truncation=True,
                            max_length=512)
val_encodings = tokenizer(val_texts,
                          is_split_into_words=True,
                          return_offsets_mapping=True,
                          padding=True,
                          truncation=True,
                          max_length=512)


# 3. 标签对齐
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


train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)


# 4. 构建数据集
class NerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_encodings.pop("offset_mapping")  # 训练不需要这个
val_encodings.pop("offset_mapping")
train_dataset = NerDataset(train_encodings, train_labels)
val_dataset = NerDataset(val_encodings, val_labels)

# 5. 导入预训练模型
model = AutoModelForTokenClassification.from_pretrained('bert-base-chinese',
                                                        num_labels=5,
                                                        ignore_mismatched_sizes=True,
                                                        id2label=id2tag,
                                                        label2id=tag2id
                                                        )
# 6. 评估标准
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 不要管-100那些，剔除掉
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# 7. 训练
checkpoint = 'bert-base-chinese'
num_train_epochs = 1000
per_device_train_batch_size = 8
per_device_eval_batch_size = 8

training_args = TrainingArguments(
    output_dir='./output',  # 输入路径
    num_train_epochs=num_train_epochs,  # 训练epoch数量
    per_device_train_batch_size=per_device_train_batch_size,  # 每个GPU的BATCH
    per_device_eval_batch_size=per_device_eval_batch_size,
    warmup_steps=500,  # warmup次数
    weight_decay=0.01,  # 限制权重的大小
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=1,
    evaluation_strategy='steps',
    eval_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


def run():
    trainer.train()
    trainer.evaluate()
    model.save_pretrained("./checkpoint/model/%s-%sepoch" % (checkpoint, num_train_epochs))


if __name__ == '__main__':
    run()
