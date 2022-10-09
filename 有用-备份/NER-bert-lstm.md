```python
# !pip install pytorch-crf
# !pip install seqeval
# !pip install transformers
```


```python
from torch.utils.data import Dataset

categories = set()

class ReadData(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, tags = '', []
                for i, c in enumerate(line.split('\n')):
                    word, tag = c.split('\t')
                    sentence += word
                    if tag[0] == 'B':
                        tags.append([i, i, word, tag[2:]]) # Remove the B- or I-
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
```


```python
train_data = ReadData('./example.train')
valid_data = ReadData('./example.dev')
test_data = ReadData('./example.test')

print(train_data[0])
```

    {'sentence': '主机厂家已机组提供高电压耐受能力情况说明（未说明具体耐受能力范围），缺少对应的报告文件支持。3.常用标准、规程、措施、制度、技术资料和各种记录缺失。主机厂家已提供符合要求的高电压耐受能力证明报告及对应的支持文件', 'tags': [[9, 13, '高电压耐受', 'Phe'], [34, 44, '缺少对应的报告文件支持', 'Phe'], [67, 72, '各种记录缺失', 'Cau'], [79, 96, '提供符合要求的高电压耐受能力证明报告', 'Met']]}
    


```python
categories
```




    {'Cau', 'Met', 'Phe'}




```python
id2label = {0:'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}

# print(id2label)
# print(label2id)
```


```python
from transformers import AutoTokenizer
import numpy as np

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence = '主机厂家已机组提供高电压耐受能力情况说明（未说明具体耐受能力范围）'
tags = [[9, 13, '高电压耐受', 'Phe']]

encoding = tokenizer(sentence, truncation=True)
tokens = encoding.tokens()
label = np.zeros(len(tokens), dtype=int)
for char_start, char_end, word, tag in tags:
    token_start = encoding.char_to_token(char_start)
    token_end = encoding.char_to_token(char_end)
    label[token_start] = label2id[f"B-{tag}"]
    label[token_start+1:token_end+1] = label2id[f"I-{tag}"]

# print(tokens)
# print(label)
# print([id2label[id] for id in label])
```


```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collote_fn(batch_samples):
    batch_sentence, batch_tags  = [], []
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
        batch_label[s_idx][len(encoding.tokens())-1:] = -100
        for char_start, char_end, _, tag in batch_tags[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
            batch_label[s_idx][token_start+1:token_end+1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

batch_X, batch_y = next(iter(train_dataloader))
# print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
# print('batch_y shape:', batch_y.shape)
# print(batch_X)
# print(batch_y)
```


```python
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 1.
        # self.bert = AutoModel.from_pretrained(checkpoint)
        # self.Linear = nn.Linear(768, len(id2label))
        # 2.
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.config = self.bert.config
        self.BiLstm=nn.LSTM(input_size=self.config.hidden_size,hidden_size=512,batch_first=True,bidirectional=True)
        # self.crf = CRF(len(id2label),batch_first=True)
        self.Linear = nn.Linear(512*2, len(id2label))
    # 1.
    def forward(self, x):
    # 2.
    # def forward(self, x, y):
        # 1.
        # output = self.bert(**x)
        # output = self.Linear(bert_output.last_hidden_state)
        # return output
        # 2.
        output = self.bert(**x).last_hidden_state
        output, _ = self.BiLstm(output)
        output = self.Linear(output)
        # loss = self.crf(output, y)
        # output = self.crf.decode(output)
        # output=torch.tensor(output)
        return output
        # return loss, output
    

model = NeuralNetwork().to(device)
print(model)
```

    Using cuda device
    

    Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertModel: ['cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    

    NeuralNetwork(
      (bert): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(21128, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (2): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (3): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (4): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (5): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (6): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (7): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (8): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (9): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (10): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (11): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (BiLstm): LSTM(768, 512, batch_first=True, bidirectional=True)
      (Linear): Linear(in_features=1024, out_features=7, bias=True)
    )
    


```python
from tqdm.auto import tqdm

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
# def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # loss, pred = model(X, y)
        #  通过 pred.permute(0, 2, 1) 交换后两维，将模型预测结果从(batch,seq,7) 调整为 (batch,7,seq)。
        loss = loss_fn(pred.permute(0, 2, 1), y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss
```


```python
# !pip install seqeval
# from seqeval.metrics import classification_report
# from seqeval.scheme import IOB2

# y_true = [['O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O'], ['B-PER', 'I-PER', 'O']]
# y_pred = [['O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O'], ['B-PER', 'I-PER', 'O']]

# print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
```


```python
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def test_loop(dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # loss, pred = model(X, y)
            predictions = pred.argmax(dim=-1)
            # predictions = pred
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in y]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, y)
            ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
```


```python
from transformers import AdamW, get_scheduler

learning_rate = 1e-5
epoch_num = 70

loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.LogSoftmax() 

optimizer = AdamW(model.parameters(), lr=learning_rate)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
loss_list=[]
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    # total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    test_loop(valid_dataloader, model)
print("Done!")
```

    Epoch 1/70
    -------------------------------
    

    /root/miniconda3/lib/python3.8/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 2/70
    -------------------------------
    

    /root/miniconda3/lib/python3.8/site-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /root/miniconda3/lib/python3.8/site-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 3/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 4/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 5/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 6/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 7/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 8/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 9/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 10/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 11/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.64      0.03      0.07       201
    
       micro avg       0.64      0.02      0.04       347
       macro avg       0.21      0.01      0.02       347
    weighted avg       0.37      0.02      0.04       347
    
    Epoch 12/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.48      0.05      0.10       201
    
       micro avg       0.48      0.03      0.06       347
       macro avg       0.16      0.02      0.03       347
    weighted avg       0.28      0.03      0.06       347
    
    Epoch 13/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.80      0.32      0.46       201
    
       micro avg       0.80      0.18      0.30       347
       macro avg       0.27      0.11      0.15       347
    weighted avg       0.46      0.18      0.26       347
    
    Epoch 14/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.80      0.34      0.48       201
    
       micro avg       0.80      0.20      0.32       347
       macro avg       0.27      0.11      0.16       347
    weighted avg       0.46      0.20      0.28       347
    
    Epoch 15/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.71      0.34      0.46       201
    
       micro avg       0.71      0.20      0.31       347
       macro avg       0.24      0.11      0.15       347
    weighted avg       0.41      0.20      0.27       347
    
    Epoch 16/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.85      0.44      0.58       201
    
       micro avg       0.85      0.26      0.39       347
       macro avg       0.28      0.15      0.19       347
    weighted avg       0.49      0.26      0.34       347
    
    Epoch 17/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.85      0.58      0.69       201
    
       micro avg       0.85      0.34      0.48       347
       macro avg       0.28      0.19      0.23       347
    weighted avg       0.49      0.34      0.40       347
    
    Epoch 18/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.78      0.53      0.63       201
    
       micro avg       0.78      0.31      0.44       347
       macro avg       0.26      0.18      0.21       347
    weighted avg       0.45      0.31      0.36       347
    
    Epoch 19/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.80      0.57      0.67       201
    
       micro avg       0.80      0.33      0.47       347
       macro avg       0.27      0.19      0.22       347
    weighted avg       0.46      0.33      0.39       347
    
    Epoch 20/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       1.00      0.04      0.08        74
             Phe       0.82      0.59      0.69       201
    
       micro avg       0.81      0.35      0.49       347
       macro avg       0.61      0.21      0.26       347
    weighted avg       0.69      0.35      0.42       347
    
    Epoch 21/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.80      0.17      0.28        72
             Met       0.80      0.05      0.10        74
             Phe       0.84      0.59      0.69       201
    
       micro avg       0.83      0.39      0.53       347
       macro avg       0.81      0.27      0.36       347
    weighted avg       0.82      0.39      0.48       347
    
    Epoch 22/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.74      0.24      0.36        72
             Met       0.88      0.28      0.43        74
             Phe       0.86      0.60      0.70       201
    
       micro avg       0.84      0.46      0.59       347
       macro avg       0.82      0.37      0.50       347
    weighted avg       0.84      0.46      0.57       347
    
    Epoch 23/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.75      0.33      0.46        72
             Met       0.89      0.34      0.49        74
             Phe       0.82      0.63      0.72       201
    
       micro avg       0.82      0.51      0.63       347
       macro avg       0.82      0.43      0.56       347
    weighted avg       0.82      0.51      0.61       347
    
    Epoch 24/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.77      0.32      0.45        72
             Met       0.86      0.24      0.38        74
             Phe       0.89      0.66      0.75       201
    
       micro avg       0.86      0.50      0.63       347
       macro avg       0.84      0.41      0.53       347
    weighted avg       0.86      0.50      0.61       347
    
    Epoch 25/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.84      0.43      0.57        72
             Met       0.86      0.50      0.63        74
             Phe       0.88      0.67      0.76       201
    
       micro avg       0.87      0.59      0.70       347
       macro avg       0.86      0.53      0.65       347
    weighted avg       0.87      0.59      0.69       347
    
    Epoch 26/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.82      0.44      0.58        72
             Met       0.82      0.50      0.62        74
             Phe       0.87      0.68      0.76       201
    
       micro avg       0.85      0.59      0.70       347
       macro avg       0.84      0.54      0.65       347
    weighted avg       0.85      0.59      0.69       347
    
    Epoch 27/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.79      0.43      0.56        72
             Met       0.85      0.46      0.60        74
             Phe       0.87      0.69      0.77       201
    
       micro avg       0.85      0.59      0.70       347
       macro avg       0.84      0.53      0.64       347
    weighted avg       0.85      0.59      0.69       347
    
    Epoch 28/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.86      0.53      0.66        72
             Met       0.91      0.65      0.76        74
             Phe       0.90      0.74      0.81       201
    
       micro avg       0.89      0.67      0.77       347
       macro avg       0.89      0.64      0.74       347
    weighted avg       0.89      0.67      0.77       347
    
    Epoch 29/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.83      0.53      0.64        72
             Met       0.90      0.62      0.74        74
             Phe       0.89      0.70      0.79       201
    
       micro avg       0.88      0.65      0.75       347
       macro avg       0.87      0.62      0.72       347
    weighted avg       0.88      0.65      0.75       347
    
    Epoch 30/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.87      0.56      0.68        72
             Met       0.86      0.59      0.70        74
             Phe       0.87      0.74      0.80       201
    
       micro avg       0.87      0.67      0.76       347
       macro avg       0.87      0.63      0.73       347
    weighted avg       0.87      0.67      0.75       347
    
    Epoch 31/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.89      0.68      0.77        72
             Met       0.91      0.70      0.79        74
             Phe       0.90      0.74      0.81       201
    
       micro avg       0.90      0.72      0.80       347
       macro avg       0.90      0.71      0.79       347
    weighted avg       0.90      0.72      0.80       347
    
    Epoch 32/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.85      0.62      0.72        72
             Met       0.91      0.68      0.78        74
             Phe       0.89      0.77      0.82       201
    
       micro avg       0.88      0.72      0.79       347
       macro avg       0.88      0.69      0.77       347
    weighted avg       0.88      0.72      0.79       347
    
    Epoch 33/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.89      0.75      0.81        72
             Met       0.91      0.70      0.79        74
             Phe       0.89      0.77      0.83       201
    
       micro avg       0.89      0.75      0.82       347
       macro avg       0.90      0.74      0.81       347
    weighted avg       0.89      0.75      0.82       347
    
    Epoch 34/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.83      0.67      0.74        72
             Met       0.91      0.70      0.79        74
             Phe       0.88      0.77      0.82       201
    
       micro avg       0.88      0.73      0.80       347
       macro avg       0.87      0.71      0.78       347
    weighted avg       0.88      0.73      0.80       347
    
    Epoch 35/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.91      0.74      0.82        72
             Met       0.89      0.68      0.77        74
             Phe       0.89      0.77      0.82       201
    
       micro avg       0.89      0.74      0.81       347
       macro avg       0.90      0.73      0.80       347
    weighted avg       0.89      0.74      0.81       347
    
    Epoch 36/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.88      0.74      0.80        72
             Met       0.95      0.72      0.82        74
             Phe       0.92      0.77      0.84       201
    
       micro avg       0.92      0.75      0.83       347
       macro avg       0.92      0.74      0.82       347
    weighted avg       0.92      0.75      0.83       347
    
    Epoch 37/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.86      0.75      0.80        72
             Met       0.91      0.70      0.79        74
             Phe       0.90      0.80      0.84       201
    
       micro avg       0.89      0.77      0.82       347
       macro avg       0.89      0.75      0.81       347
    weighted avg       0.89      0.77      0.82       347
    
    Epoch 38/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.85      0.76      0.80        72
             Met       0.93      0.72      0.81        74
             Phe       0.91      0.78      0.84       201
    
       micro avg       0.90      0.76      0.82       347
       macro avg       0.90      0.75      0.82       347
    weighted avg       0.90      0.76      0.82       347
    
    Epoch 39/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.89      0.76      0.82        72
             Met       0.95      0.73      0.82        74
             Phe       0.92      0.79      0.85       201
    
       micro avg       0.92      0.77      0.84       347
       macro avg       0.92      0.76      0.83       347
    weighted avg       0.92      0.77      0.84       347
    
    Epoch 40/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.86      0.90        72
             Met       0.91      0.69      0.78        74
             Phe       0.88      0.86      0.87       201
    
       micro avg       0.90      0.82      0.86       347
       macro avg       0.91      0.80      0.85       347
    weighted avg       0.90      0.82      0.86       347
    
    Epoch 41/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.92      0.82      0.87        72
             Met       0.91      0.70      0.79        74
             Phe       0.94      0.79      0.86       201
    
       micro avg       0.93      0.78      0.85       347
       macro avg       0.92      0.77      0.84       347
    weighted avg       0.93      0.78      0.85       347
    
    Epoch 42/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.85      0.89        72
             Met       0.92      0.73      0.81        74
             Phe       0.90      0.83      0.87       201
    
       micro avg       0.91      0.81      0.86       347
       macro avg       0.92      0.80      0.86       347
    weighted avg       0.91      0.81      0.86       347
    
    Epoch 43/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.89      0.82      0.86        72
             Met       0.92      0.74      0.82        74
             Phe       0.89      0.84      0.86       201
    
       micro avg       0.90      0.82      0.85       347
       macro avg       0.90      0.80      0.85       347
    weighted avg       0.90      0.82      0.85       347
    
    Epoch 44/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.91      0.88      0.89        72
             Met       0.92      0.73      0.81        74
             Phe       0.91      0.85      0.88       201
    
       micro avg       0.91      0.83      0.87       347
       macro avg       0.91      0.82      0.86       347
    weighted avg       0.91      0.83      0.87       347
    
    Epoch 45/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.89      0.91        72
             Met       0.93      0.74      0.83        74
             Phe       0.90      0.86      0.88       201
    
       micro avg       0.91      0.84      0.88       347
       macro avg       0.92      0.83      0.87       347
    weighted avg       0.91      0.84      0.87       347
    
    Epoch 46/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.89      0.91        72
             Met       0.88      0.69      0.77        74
             Phe       0.88      0.88      0.88       201
    
       micro avg       0.89      0.84      0.87       347
       macro avg       0.90      0.82      0.85       347
    weighted avg       0.89      0.84      0.86       347
    
    Epoch 47/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.88      0.90        72
             Met       0.95      0.76      0.84        74
             Phe       0.89      0.90      0.90       201
    
       micro avg       0.91      0.86      0.89       347
       macro avg       0.92      0.84      0.88       347
    weighted avg       0.91      0.86      0.89       347
    
    Epoch 48/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.91      0.86      0.89        72
             Met       0.93      0.74      0.83        74
             Phe       0.90      0.87      0.89       201
    
       micro avg       0.91      0.84      0.87       347
       macro avg       0.92      0.83      0.87       347
    weighted avg       0.91      0.84      0.87       347
    
    Epoch 49/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.92      0.73      0.81        74
             Phe       0.90      0.85      0.87       201
    
       micro avg       0.91      0.83      0.87       347
       macro avg       0.92      0.82      0.86       347
    weighted avg       0.91      0.83      0.87       347
    
    Epoch 50/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.90      0.93        72
             Met       0.93      0.76      0.84        74
             Phe       0.90      0.89      0.89       201
    
       micro avg       0.92      0.86      0.89       347
       macro avg       0.93      0.85      0.89       347
    weighted avg       0.92      0.86      0.89       347
    
    Epoch 51/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.90      0.93        72
             Met       0.97      0.78      0.87        74
             Phe       0.90      0.91      0.90       201
    
       micro avg       0.92      0.88      0.90       347
       macro avg       0.94      0.86      0.90       347
    weighted avg       0.92      0.88      0.90       347
    
    Epoch 52/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.88      0.90        72
             Met       0.97      0.78      0.87        74
             Phe       0.90      0.89      0.90       201
    
       micro avg       0.92      0.86      0.89       347
       macro avg       0.93      0.85      0.89       347
    weighted avg       0.92      0.86      0.89       347
    
    Epoch 53/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.92      0.94        72
             Met       0.97      0.78      0.87        74
             Phe       0.89      0.89      0.89       201
    
       micro avg       0.92      0.87      0.89       347
       macro avg       0.94      0.86      0.90       347
    weighted avg       0.92      0.87      0.89       347
    
    Epoch 54/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.78      0.87        74
             Phe       0.88      0.91      0.89       201
    
       micro avg       0.91      0.88      0.90       347
       macro avg       0.94      0.87      0.90       347
    weighted avg       0.92      0.88      0.90       347
    
    Epoch 55/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.81      0.88        74
             Phe       0.91      0.90      0.91       201
    
       micro avg       0.93      0.89      0.91       347
       macro avg       0.95      0.88      0.91       347
    weighted avg       0.93      0.89      0.91       347
    
    Epoch 56/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.92      0.94        72
             Met       0.97      0.81      0.88        74
             Phe       0.91      0.90      0.90       201
    
       micro avg       0.93      0.88      0.91       347
       macro avg       0.94      0.88      0.91       347
    weighted avg       0.93      0.88      0.91       347
    
    Epoch 57/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.95      0.80      0.87        74
             Phe       0.91      0.91      0.91       201
    
       micro avg       0.92      0.89      0.91       347
       macro avg       0.94      0.88      0.91       347
    weighted avg       0.93      0.89      0.91       347
    
    Epoch 58/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.95      0.82      0.88        74
             Phe       0.90      0.91      0.90       201
    
       micro avg       0.92      0.89      0.91       347
       macro avg       0.94      0.89      0.91       347
    weighted avg       0.92      0.89      0.91       347
    
    Epoch 59/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.95      0.84      0.89        74
             Phe       0.89      0.91      0.90       201
    
       micro avg       0.92      0.90      0.91       347
       macro avg       0.93      0.89      0.91       347
    weighted avg       0.92      0.90      0.91       347
    
    Epoch 60/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.90      0.92        72
             Met       0.95      0.82      0.88        74
             Phe       0.88      0.91      0.89       201
    
       micro avg       0.91      0.89      0.90       347
       macro avg       0.93      0.88      0.90       347
    weighted avg       0.91      0.89      0.90       347
    
    Epoch 61/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.85      0.91        74
             Phe       0.91      0.91      0.91       201
    
       micro avg       0.93      0.90      0.91       347
       macro avg       0.94      0.90      0.92       347
    weighted avg       0.93      0.90      0.91       347
    
    Epoch 62/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.95      0.84      0.89        74
             Phe       0.91      0.91      0.91       201
    
       micro avg       0.93      0.90      0.91       347
       macro avg       0.94      0.89      0.92       347
    weighted avg       0.93      0.90      0.91       347
    
    Epoch 63/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.95      0.84      0.89        74
             Phe       0.92      0.91      0.91       201
    
       micro avg       0.93      0.90      0.91       347
       macro avg       0.94      0.89      0.92       347
    weighted avg       0.93      0.90      0.91       347
    
    Epoch 64/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.84      0.90        74
             Phe       0.92      0.91      0.91       201
    
       micro avg       0.93      0.90      0.92       347
       macro avg       0.95      0.89      0.92       347
    weighted avg       0.94      0.90      0.92       347
    
    Epoch 65/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.85      0.91        74
             Phe       0.90      0.92      0.91       201
    
       micro avg       0.93      0.91      0.92       347
       macro avg       0.94      0.90      0.92       347
    weighted avg       0.93      0.91      0.92       347
    
    Epoch 66/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.85      0.91        74
             Phe       0.92      0.92      0.92       201
    
       micro avg       0.93      0.90      0.92       347
       macro avg       0.95      0.90      0.92       347
    weighted avg       0.94      0.90      0.92       347
    
    Epoch 67/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.85      0.91        74
             Phe       0.92      0.91      0.91       201
    
       micro avg       0.93      0.90      0.92       347
       macro avg       0.95      0.90      0.92       347
    weighted avg       0.94      0.90      0.92       347
    
    Epoch 68/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.85      0.91        74
             Phe       0.92      0.91      0.91       201
    
       micro avg       0.93      0.90      0.92       347
       macro avg       0.95      0.90      0.92       347
    weighted avg       0.94      0.90      0.92       347
    
    Epoch 69/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.85      0.91        74
             Phe       0.92      0.91      0.91       201
    
       micro avg       0.93      0.90      0.92       347
       macro avg       0.95      0.90      0.92       347
    weighted avg       0.94      0.90      0.92       347
    
    Epoch 70/70
    -------------------------------
    


      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.93      0.94        72
             Met       0.97      0.85      0.91        74
             Phe       0.92      0.91      0.91       201
    
       micro avg       0.93      0.90      0.92       347
       macro avg       0.95      0.90      0.92       347
    weighted avg       0.94      0.90      0.92       347
    
    Done!
    


```python
sentence = '在使用过程中若发现油位指示窗内出现油面，说明波纹囊有渗漏，绝缘油进入空气腔。发现指示窗有油应马上通知厂家处理，并采取临时措施。'
results = []
with torch.no_grad():
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    pred = model(inputs)
    probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].tolist()
    predictions = pred.argmax(dim=-1)[0].tolist()

    pred_label = []
    inputs_with_offsets = tokenizer(sentence, return_offsets_mapping=True)
    tokens = inputs_with_offsets.tokens()
    offsets = inputs_with_offsets["offset_mapping"]

    idx = 0
    while idx < len(predictions):
        pred = predictions[idx]
        label = id2label[pred]
        if label != "O":
            label = label[2:] # Remove the B- or I-
            start, end = offsets[idx]
            all_scores = [probabilities[idx][pred]]
            # Grab all the tokens labeled with I-label
            while (
                idx + 1 < len(predictions) and 
                id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1

            score = np.mean(all_scores).item()
            word = sentence[start:end]
            pred_label.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    print(pred_label)
```

    [{'entity_group': 'Phe', 'score': 0.8737103276782565, 'word': '位指示窗内出现油面', 'start': 10, 'end': 19}, {'entity_group': 'Phe', 'score': 0.9347905218601227, 'word': '波纹囊有渗漏', 'start': 22, 'end': 28}, {'entity_group': 'Phe', 'score': 0.9199935048818588, 'word': '绝缘油进入空气腔', 'start': 29, 'end': 37}, {'entity_group': 'Met', 'score': 0.8187771836916605, 'word': '现指示窗有油', 'start': 39, 'end': 45}, {'entity_group': 'Met', 'score': 0.7216312363743782, 'word': '马上通知厂家处理', 'start': 46, 'end': 54}, {'entity_group': 'Met', 'score': 0.8945693842002324, 'word': '并采取临时措施', 'start': 55, 'end': 62}]
    


```python
sentence = '气体继电器保护装置的信号动作时，值班员应立即停止报警信号，并检查变压器，查明信号动作的原因，是否因空气侵入变压器内，或是油位降低，或是二次回路故障。'
results = []
with torch.no_grad():
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    pred = model(inputs)
    probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].tolist()
    predictions = pred.argmax(dim=-1)[0].tolist()

    pred_label = []
    inputs_with_offsets = tokenizer(sentence, return_offsets_mapping=True)
    tokens = inputs_with_offsets.tokens()
    offsets = inputs_with_offsets["offset_mapping"]

    idx = 0
    while idx < len(predictions):
        pred = predictions[idx]
        label = id2label[pred]
        if label != "O":
            label = label[2:] # Remove the B- or I-
            start, end = offsets[idx]
            all_scores = [probabilities[idx][pred]]
            # Grab all the tokens labeled with I-label
            while (
                idx + 1 < len(predictions) and 
                id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1

            score = np.mean(all_scores).item()
            word = sentence[start:end]
            pred_label.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    print(pred_label)
```

    [{'entity_group': 'Phe', 'score': 0.9662598818540573, 'word': '信号动作', 'start': 10, 'end': 14}, {'entity_group': 'Cau', 'score': 0.652740404009819, 'word': '气侵入变压器', 'start': 50, 'end': 56}, {'entity_group': 'Phe', 'score': 0.89470574259758, 'word': '油位降低', 'start': 60, 'end': 64}, {'entity_group': 'Cau', 'score': 0.9158064872026443, 'word': '回路故障', 'start': 69, 'end': 73}]
    


```python

```
