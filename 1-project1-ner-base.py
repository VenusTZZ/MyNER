#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:1-project1-ner-base.py
@time:2022/10/05/19：33
"""
from torch.utils.data import Dataset
# from transformers import AutoTokenizer
# import torch
from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# import numpy as np
from torch import nn
# from transformers import AutoModel
# from tqdm.auto import tqdm
# # from seqeval.metrics import classification_report
# # from seqeval.scheme import IOB2
# from transformers import AdamW, get_scheduler

# --------------------------------------------------------------------------------
# checkpoint = "bert-base-chinese"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using {device} device')


# --------------------------------------------------------------------------------
class ReadData(Dataset):
    def __init__(self, data_file, categories):
        self.data = self.load_data(data_file, categories)

    def load_data(self, data_file, categories):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, tags = '', []
                for i, c in enumerate(line.split('\n')):
                    word, tag = c.split('   ')
                    sentence += word
                    if tag[0] == 'B':
                        tags.append([i, i, word, tag[2:]])  # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag[0] == 'I':
                        tags[-1][1] = i
                        tags[-1][2] += word
                Data[idx] = {
                    'sentence': sentence,
                    'tags': tags
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    batch_sentence, batch_tags = [], []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_tags.append(sample['tags'])
    batch_inputs = tokenizer(
        batch_sentence,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for s_idx, sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence, truncation=True)
        batch_label[s_idx][0] = -100
        batch_label[s_idx][len(encoding.tokens()) - 1:] = -100
        for char_start, char_end, _, tag in batch_tags[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
            batch_label[s_idx][token_start + 1:token_end + 1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(768, len(id2label))

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        logits = self.classifier(bert_output.last_hidden_state)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0, 2, 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1)
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in y]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, y)
            ]
    # print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))


# --------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------------
    categories = set()
    # train_data = ReadData('example.train', categories)
    train_data = ReadData('example.train', categories)
    # valid_data = ReadData('dev.txt', categories)
    # test_data = ReadData('./example.test', categories)

    print(train_data[321])
    #
    # id2label = {0: 'O'}
    # for c in list(sorted(categories)):
    #     id2label[len(id2label)] = f"B-{c}"
    #     id2label[len(id2label)] = f"I-{c}"
    # label2id = {v: k for k, v in id2label.items()}
    #
    # print(id2label)
    # print(label2id)
    # # --------------------------------------------------------------------------------
    # # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    #
    # # sentence = '主机厂家已机组提供高电压耐受能力情况说明（未说明具体耐受能力范围）'
    # # tags = [[9, 13, '高电压耐受', 'Phe']]
    # #
    # # encoding = tokenizer(sentence, truncation=True)
    # # tokens = encoding.tokens()
    # # label = np.zeros(len(tokens), dtype=int)
    # # for char_start, char_end, word, tag in tags:
    # #     token_start = encoding.char_to_token(char_start)
    # #     token_end = encoding.char_to_token(char_end)
    # #     label[token_start] = label2id[f"B-{tag}"]
    # #     label[token_start + 1:token_end + 1] = label2id[f"I-{tag}"]
    # # print(tokens)
    # # print(label)
    # # print([id2label[id] for id in label])
    #
    # # checkpoint = "bert-base-chinese"
    # # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    #
    # train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    # valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
    # test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
    #
    # batch_X, batch_y = next(iter(train_dataloader))
    # # print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    # # print('batch_y shape:', batch_y.shape)
    # # print(batch_X)
    # # print(batch_y)
    #
    # # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #
    # model = NeuralNetwork().to(device)
    # # print(model)
    #
    # learning_rate = 1e-5
    # epoch_num = 70
    #
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=epoch_num * len(train_dataloader),
    # )
    #
    # total_loss = 0.
    # loss_list = []
    # for t in range(epoch_num):
    #     print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
    #     total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss)
    #     test_loop(valid_dataloader, model)
    # print("Done!")
    #
    # sentence = '6.发电设备功率特性曲线出现整体右移、左移或最大功率达不到额定功率或出现其它异常的情况。1）对于功率曲线多线的机组，对机组主控程序版本和参数、风速计、叶片零位角等进行检查2）对于风速数据异常的机组，对风速计和风速传递函数进行检查，并对问题风速计已更换3）对于功率曲线性能低下的机组，从风向标安装角度、风速计、主控程序参数、叶片初始零位角、是否进行精维护、是否进行相关技改或大修等方面检查5）对于数据异常的机组，从停机时长、通信异常、存储异常等方面进行排查。'
    # results = []
    # with torch.no_grad():
    #     inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    #     inputs = inputs.to(device)
    #     pred = model(inputs)
    #     probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].tolist()
    #     predictions = pred.argmax(dim=-1)[0].tolist()
    #
    #     pred_label = []
    #     inputs_with_offsets = tokenizer(sentence, return_offsets_mapping=True)
    #     tokens = inputs_with_offsets.tokens()
    #     offsets = inputs_with_offsets["offset_mapping"]
    #
    #     idx = 0
    #     while idx < len(predictions):
    #         pred = predictions[idx]
    #         label = id2label[pred]
    #         if label != "O":
    #             label = label[2:]  # Remove the B- or I-
    #             start, end = offsets[idx]
    #             all_scores = [probabilities[idx][pred]]
    #             # Grab all the tokens labeled with I-label
    #             while (
    #                     idx + 1 < len(predictions) and
    #                     id2label[predictions[idx + 1]] == f"I-{label}"
    #             ):
    #                 all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
    #                 _, end = offsets[idx + 1]
    #                 idx += 1
    #
    #             score = np.mean(all_scores).item()
    #             word = sentence[start:end]
    #             pred_label.append(
    #                 {
    #                     "entity_group": label,
    #                     "score": score,
    #                     "word": word,
    #                     "start": start,
    #                     "end": end,
    #                 }
    #             )
    #         idx += 1
    #     print(pred_label)
