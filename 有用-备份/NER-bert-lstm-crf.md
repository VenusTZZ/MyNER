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

print(id2label)
print(label2id)
```

    {0: 'O', 1: 'B-Cau', 2: 'I-Cau', 3: 'B-Met', 4: 'I-Met', 5: 'B-Phe', 6: 'I-Phe'}
    {'O': 0, 'B-Cau': 1, 'I-Cau': 2, 'B-Met': 3, 'I-Met': 4, 'B-Phe': 5, 'I-Phe': 6}



```python
# from transformers import AutoTokenizer
# import numpy as np

# checkpoint = "bert-base-chinese"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# sentence = '主机厂家已机组提供高电压耐受能力情况说明（未说明具体耐受能力范围）'
# tags = [[9, 13, '高电压耐受', 'Phe']]

# encoding = tokenizer(sentence, truncation=True)
# tokens = encoding.tokens()
# label = np.zeros(len(tokens), dtype=int)
# for char_start, char_end, word, tag in tags:
#     token_start = encoding.char_to_token(char_start)
#     token_end = encoding.char_to_token(char_end)
#     label[token_start] = label2id[f"B-{tag}"]
#     label[token_start+1:token_end+1] = label2id[f"I-{tag}"]

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
    # batch_sentence, batch_tags,mask = [], [], []
    batch_sentence, batch_tags = [], []
    for sample in batch_samples:
        # print(sample)
        batch_sentence.append(sample['sentence'])
        batch_tags.append(sample['tags'])
        # mask.append(sample['mask_tensor'])
    batch_inputs = tokenizer(
        batch_sentence, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        # max_length=256
    )
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for s_idx, sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence, truncation=True)
        batch_label[s_idx][0] = 0
        batch_label[s_idx][len(encoding.tokens())-1:] = 0
        for char_start, char_end, _, tag in batch_tags[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
            batch_label[s_idx][token_start+1:token_end+1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)

# train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.config = self.bert.config
        self.BiLstm=nn.LSTM(input_size=self.config.hidden_size,hidden_size=512,batch_first=True,bidirectional=True,num_layers=2)
        self.Linear = nn.Linear(512*2, len(id2label))
        self.crf = CRF(len(id2label),batch_first=True)
            
    # def forward(self, x):
    def forward(self, x, y):
        # 1.
        # output = self.bert(**x).last_hidden_state
        # output, _ = self.BiLstm(output)
        # output = self.Linear(output)
        # return output
        # 2.
        output = self.bert(**x).last_hidden_state
        output, _ = self.BiLstm(output)
        output = self.Linear(output)
        # loss = self.crf(emissions=output,tags=y,mask=mask_tensor)
        # tag = self.crf.decode(emissions=output,,mask=mask_tensor)
        loss = self.crf(emissions=output,tags=y)
        tag = self.crf.decode(emissions=output)
        tag=torch.tensor(tag)
        return loss, tag
    
    def decode(self,x):
        output = self.bert(**x).last_hidden_state
        output, _ = self.BiLstm(output)
        output = self.Linear(output)
        tag = self.crf.decode(emissions=output)
        tag=torch.tensor(tag)
        return tag
    
model = model().to(device)
# print(model)
```

    Using cuda device


    Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertModel: ['cls.predictions.decoder.weight', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



```python
from tqdm.auto import tqdm

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
# def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        loss, tag = model(X, y)
        #  通过 pred.permute(0, 2, 1) 交换后两维，将模型预测结果从(batch,seq,7) 调整为 (batch,7,seq)。
        # loss = loss_fn(pred.permute(0, 2, 1), y)
        loss = abs(loss)
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
            
            # pred = model(X)
            loss, tag = model(X, y)
            
            # predictions = pred.argmax(dim=-1)
            predictions = tag
            
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
epoch_num = 80
loss_fn = nn.CrossEntropyLoss()
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

    Epoch 1/80
    -------------------------------


    /root/miniconda3/lib/python3.8/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(



      0%|          | 0/39 [00:00<?, ?it/s]


    /root/miniconda3/lib/python3.8/site-packages/torchcrf/__init__.py:249: UserWarning: where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead. (Triggered internally at  ../aten/src/ATen/native/TensorCompare.cpp:333.)
      score = torch.where(mask[i].unsqueeze(1), next_score, score)



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.00      0.00      0.00       201
    
       micro avg       0.00      0.00      0.00       347
       macro avg       0.00      0.00      0.00       347
    weighted avg       0.00      0.00      0.00       347
    
    Epoch 2/80
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
    
    Epoch 3/80
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
    
    Epoch 4/80
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
    
    Epoch 5/80
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
    
    Epoch 6/80
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
    
    Epoch 7/80
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
    
    Epoch 8/80
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
    
    Epoch 9/80
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
    
    Epoch 10/80
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
    
    Epoch 11/80
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
    
    Epoch 12/80
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
    
    Epoch 13/80
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
    
    Epoch 14/80
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
    
    Epoch 15/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.54      0.07      0.12       201
    
       micro avg       0.54      0.04      0.08       347
       macro avg       0.18      0.02      0.04       347
    weighted avg       0.31      0.04      0.07       347
    
    Epoch 16/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.56      0.10      0.17       201
    
       micro avg       0.56      0.06      0.10       347
       macro avg       0.19      0.03      0.06       347
    weighted avg       0.32      0.06      0.10       347
    
    Epoch 17/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.69      0.24      0.35       201
    
       micro avg       0.69      0.14      0.23       347
       macro avg       0.23      0.08      0.12       347
    weighted avg       0.40      0.14      0.21       347
    
    Epoch 18/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.69      0.27      0.39       201
    
       micro avg       0.69      0.16      0.25       347
       macro avg       0.23      0.09      0.13       347
    weighted avg       0.40      0.16      0.22       347
    
    Epoch 19/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.74      0.39      0.51       201
    
       micro avg       0.74      0.22      0.34       347
       macro avg       0.25      0.13      0.17       347
    weighted avg       0.43      0.22      0.29       347
    
    Epoch 20/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.72      0.42      0.53       201
    
       micro avg       0.72      0.24      0.37       347
       macro avg       0.24      0.14      0.18       347
    weighted avg       0.42      0.24      0.31       347
    
    Epoch 21/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.74      0.45      0.56       201
    
       micro avg       0.74      0.26      0.38       347
       macro avg       0.25      0.15      0.19       347
    weighted avg       0.43      0.26      0.32       347
    
    Epoch 22/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.00      0.00      0.00        74
             Phe       0.63      0.29      0.40       201
    
       micro avg       0.63      0.17      0.27       347
       macro avg       0.21      0.10      0.13       347
    weighted avg       0.37      0.17      0.23       347
    
    Epoch 23/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       1.00      0.04      0.08        74
             Phe       0.68      0.34      0.46       201
    
       micro avg       0.69      0.21      0.32       347
       macro avg       0.56      0.13      0.18       347
    weighted avg       0.61      0.21      0.28       347
    
    Epoch 24/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.75      0.12      0.21        74
             Phe       0.74      0.44      0.55       201
    
       micro avg       0.74      0.28      0.41       347
       macro avg       0.50      0.19      0.25       347
    weighted avg       0.59      0.28      0.36       347
    
    Epoch 25/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.00      0.00      0.00        72
             Met       0.65      0.15      0.24        74
             Phe       0.74      0.46      0.57       201
    
       micro avg       0.73      0.30      0.43       347
       macro avg       0.46      0.20      0.27       347
    weighted avg       0.57      0.30      0.38       347
    
    Epoch 26/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.75      0.17      0.27        72
             Met       0.72      0.24      0.36        74
             Phe       0.74      0.52      0.61       201
    
       micro avg       0.74      0.39      0.51       347
       macro avg       0.74      0.31      0.42       347
    weighted avg       0.74      0.39      0.49       347
    
    Epoch 27/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.75      0.17      0.27        72
             Met       0.70      0.26      0.38        74
             Phe       0.76      0.53      0.62       201
    
       micro avg       0.75      0.39      0.52       347
       macro avg       0.74      0.32      0.42       347
    weighted avg       0.74      0.39      0.50       347
    
    Epoch 28/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.75      0.29      0.42        72
             Met       0.67      0.27      0.38        74
             Phe       0.81      0.56      0.66       201
    
       micro avg       0.78      0.44      0.56       347
       macro avg       0.74      0.37      0.49       347
    weighted avg       0.76      0.44      0.55       347
    
    Epoch 29/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.68      0.35      0.46        72
             Met       0.81      0.39      0.53        74
             Phe       0.85      0.62      0.72       201
    
       micro avg       0.81      0.52      0.63       347
       macro avg       0.78      0.45      0.57       347
    weighted avg       0.80      0.52      0.62       347
    
    Epoch 30/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.79      0.51      0.62        72
             Met       0.88      0.57      0.69        74
             Phe       0.85      0.66      0.75       201
    
       micro avg       0.84      0.61      0.71       347
       macro avg       0.84      0.58      0.69       347
    weighted avg       0.84      0.61      0.71       347
    
    Epoch 31/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.74      0.44      0.56        72
             Met       0.81      0.57      0.67        74
             Phe       0.76      0.60      0.67       201
    
       micro avg       0.77      0.56      0.65       347
       macro avg       0.77      0.54      0.63       347
    weighted avg       0.77      0.56      0.64       347
    
    Epoch 32/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.78      0.53      0.63        72
             Met       0.83      0.46      0.59        74
             Phe       0.83      0.68      0.75       201
    
       micro avg       0.82      0.60      0.69       347
       macro avg       0.81      0.55      0.65       347
    weighted avg       0.82      0.60      0.69       347
    
    Epoch 33/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.78      0.60      0.68        72
             Met       0.83      0.59      0.69        74
             Phe       0.84      0.68      0.75       201
    
       micro avg       0.82      0.65      0.72       347
       macro avg       0.82      0.62      0.71       347
    weighted avg       0.82      0.65      0.72       347
    
    Epoch 34/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.81      0.64      0.71        72
             Met       0.81      0.59      0.69        74
             Phe       0.84      0.65      0.73       201
    
       micro avg       0.83      0.63      0.72       347
       macro avg       0.82      0.63      0.71       347
    weighted avg       0.83      0.63      0.72       347
    
    Epoch 35/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.83      0.61      0.70        72
             Met       0.77      0.55      0.65        74
             Phe       0.81      0.64      0.72       201
    
       micro avg       0.81      0.62      0.70       347
       macro avg       0.81      0.60      0.69       347
    weighted avg       0.81      0.62      0.70       347
    
    Epoch 36/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.79      0.61      0.69        72
             Met       0.80      0.58      0.67        74
             Phe       0.88      0.72      0.79       201
    
       micro avg       0.84      0.67      0.74       347
       macro avg       0.82      0.64      0.72       347
    weighted avg       0.84      0.67      0.74       347
    
    Epoch 37/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.82      0.64      0.72        72
             Met       0.79      0.55      0.65        74
             Phe       0.83      0.66      0.74       201
    
       micro avg       0.82      0.63      0.72       347
       macro avg       0.81      0.62      0.70       347
    weighted avg       0.82      0.63      0.71       347
    
    Epoch 38/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.79      0.62      0.70        72
             Met       0.80      0.61      0.69        74
             Phe       0.83      0.71      0.76       201
    
       micro avg       0.82      0.67      0.74       347
       macro avg       0.81      0.65      0.72       347
    weighted avg       0.82      0.67      0.73       347
    
    Epoch 39/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.83      0.67      0.74        72
             Met       0.81      0.64      0.71        74
             Phe       0.89      0.72      0.80       201
    
       micro avg       0.86      0.69      0.77       347
       macro avg       0.84      0.67      0.75       347
    weighted avg       0.86      0.69      0.77       347
    
    Epoch 40/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.83      0.68      0.75        72
             Met       0.86      0.69      0.77        74
             Phe       0.90      0.78      0.83       201
    
       micro avg       0.88      0.74      0.80       347
       macro avg       0.87      0.72      0.78       347
    weighted avg       0.88      0.74      0.80       347
    
    Epoch 41/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.88      0.74      0.80        72
             Met       0.86      0.69      0.77        74
             Phe       0.90      0.75      0.82       201
    
       micro avg       0.89      0.73      0.80       347
       macro avg       0.88      0.72      0.80       347
    weighted avg       0.89      0.73      0.80       347
    
    Epoch 42/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.89      0.71      0.79        72
             Met       0.86      0.69      0.77        74
             Phe       0.89      0.74      0.81       201
    
       micro avg       0.89      0.72      0.79       347
       macro avg       0.88      0.71      0.79       347
    weighted avg       0.89      0.72      0.79       347
    
    Epoch 43/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.82      0.64      0.72        72
             Met       0.83      0.68      0.75        74
             Phe       0.92      0.75      0.83       201
    
       micro avg       0.88      0.71      0.79       347
       macro avg       0.86      0.69      0.76       347
    weighted avg       0.88      0.71      0.79       347
    
    Epoch 44/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.88      0.71      0.78        72
             Met       0.87      0.70      0.78        74
             Phe       0.89      0.74      0.81       201
    
       micro avg       0.88      0.73      0.80       347
       macro avg       0.88      0.72      0.79       347
    weighted avg       0.88      0.73      0.80       347
    
    Epoch 45/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.84      0.65      0.73        72
             Met       0.85      0.68      0.75        74
             Phe       0.89      0.73      0.80       201
    
       micro avg       0.87      0.70      0.78       347
       macro avg       0.86      0.69      0.76       347
    weighted avg       0.87      0.70      0.78       347
    
    Epoch 46/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.86      0.68      0.76        72
             Met       0.85      0.68      0.75        74
             Phe       0.90      0.75      0.82       201
    
       micro avg       0.88      0.72      0.79       347
       macro avg       0.87      0.70      0.78       347
    weighted avg       0.88      0.72      0.79       347
    
    Epoch 47/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.91      0.82      0.86        72
             Met       0.85      0.68      0.75        74
             Phe       0.93      0.78      0.85       201
    
       micro avg       0.91      0.76      0.83       347
       macro avg       0.89      0.76      0.82       347
    weighted avg       0.91      0.76      0.83       347
    
    Epoch 48/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.92      0.81      0.86        72
             Met       0.84      0.70      0.76        74
             Phe       0.92      0.76      0.83       201
    
       micro avg       0.90      0.76      0.82       347
       macro avg       0.89      0.76      0.82       347
    weighted avg       0.90      0.76      0.82       347
    
    Epoch 49/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.90      0.78      0.84        72
             Met       0.82      0.69      0.75        74
             Phe       0.93      0.78      0.85       201
    
       micro avg       0.90      0.76      0.82       347
       macro avg       0.88      0.75      0.81       347
    weighted avg       0.90      0.76      0.82       347
    
    Epoch 50/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.92      0.79      0.85        72
             Met       0.84      0.69      0.76        74
             Phe       0.89      0.74      0.81       201
    
       micro avg       0.89      0.74      0.81       347
       macro avg       0.88      0.74      0.81       347
    weighted avg       0.89      0.74      0.81       347
    
    Epoch 51/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.90      0.79      0.84        72
             Met       0.87      0.73      0.79        74
             Phe       0.89      0.85      0.87       201
    
       micro avg       0.89      0.81      0.85       347
       macro avg       0.89      0.79      0.84       347
    weighted avg       0.89      0.81      0.85       347
    
    Epoch 52/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.91      0.85      0.88        72
             Met       0.87      0.73      0.79        74
             Phe       0.93      0.79      0.85       201
    
       micro avg       0.91      0.79      0.85       347
       macro avg       0.90      0.79      0.84       347
    weighted avg       0.91      0.79      0.85       347
    
    Epoch 53/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.88      0.79      0.83        72
             Met       0.85      0.72      0.78        74
             Phe       0.93      0.78      0.85       201
    
       micro avg       0.90      0.77      0.83       347
       macro avg       0.89      0.76      0.82       347
    weighted avg       0.91      0.77      0.83       347
    
    Epoch 54/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.89      0.82      0.86        72
             Met       0.86      0.73      0.79        74
             Phe       0.92      0.82      0.87       201
    
       micro avg       0.90      0.80      0.85       347
       macro avg       0.89      0.79      0.84       347
    weighted avg       0.90      0.80      0.85       347
    
    Epoch 55/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.88      0.90        72
             Met       0.89      0.76      0.82        74
             Phe       0.88      0.74      0.80       201
    
       micro avg       0.89      0.77      0.83       347
       macro avg       0.90      0.79      0.84       347
    weighted avg       0.89      0.77      0.83       347
    
    Epoch 56/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.88      0.90        72
             Met       0.89      0.78      0.83        74
             Phe       0.94      0.80      0.86       201
    
       micro avg       0.93      0.81      0.86       347
       macro avg       0.92      0.82      0.87       347
    weighted avg       0.93      0.81      0.86       347
    
    Epoch 57/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.91      0.81      0.85        72
             Met       0.86      0.74      0.80        74
             Phe       0.96      0.78      0.86       201
    
       micro avg       0.92      0.78      0.84       347
       macro avg       0.91      0.77      0.84       347
    weighted avg       0.93      0.78      0.84       347
    
    Epoch 58/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.92      0.81      0.86        72
             Met       0.86      0.73      0.79        74
             Phe       0.95      0.78      0.85       201
    
       micro avg       0.92      0.77      0.84       347
       macro avg       0.91      0.77      0.83       347
    weighted avg       0.92      0.77      0.84       347
    
    Epoch 59/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.86      0.77      0.81        74
             Phe       0.94      0.81      0.87       201
    
       micro avg       0.92      0.81      0.86       347
       macro avg       0.91      0.82      0.86       347
    weighted avg       0.92      0.81      0.86       347
    
    Epoch 60/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.86      0.89        72
             Met       0.90      0.81      0.85        74
             Phe       0.94      0.83      0.88       201
    
       micro avg       0.93      0.83      0.88       347
       macro avg       0.92      0.83      0.88       347
    weighted avg       0.93      0.83      0.88       347
    
    Epoch 61/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.86      0.90        72
             Met       0.90      0.81      0.85        74
             Phe       0.95      0.84      0.89       201
    
       micro avg       0.94      0.84      0.88       347
       macro avg       0.93      0.84      0.88       347
    weighted avg       0.94      0.84      0.88       347
    
    Epoch 62/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.92      0.85      0.88        72
             Met       0.91      0.82      0.87        74
             Phe       0.94      0.77      0.85       201
    
       micro avg       0.93      0.80      0.86       347
       macro avg       0.93      0.81      0.87       347
    weighted avg       0.93      0.80      0.86       347
    
    Epoch 63/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.86      0.90        72
             Met       0.92      0.82      0.87        74
             Phe       0.95      0.82      0.88       201
    
       micro avg       0.94      0.83      0.88       347
       macro avg       0.94      0.84      0.88       347
    weighted avg       0.94      0.83      0.88       347
    
    Epoch 64/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.86      0.90        72
             Met       0.93      0.84      0.88        74
             Phe       0.94      0.83      0.88       201
    
       micro avg       0.94      0.84      0.88       347
       macro avg       0.93      0.84      0.89       347
    weighted avg       0.94      0.84      0.88       347
    
    Epoch 65/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.91      0.81      0.85        72
             Met       0.91      0.81      0.86        74
             Phe       0.96      0.81      0.88       201
    
       micro avg       0.94      0.81      0.87       347
       macro avg       0.92      0.81      0.86       347
    weighted avg       0.94      0.81      0.87       347
    
    Epoch 66/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.92      0.85      0.88        72
             Met       0.93      0.84      0.88        74
             Phe       0.95      0.81      0.87       201
    
       micro avg       0.94      0.82      0.88       347
       macro avg       0.93      0.83      0.88       347
    weighted avg       0.94      0.82      0.88       347
    
    Epoch 67/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.86      0.89        72
             Met       0.92      0.82      0.87        74
             Phe       0.94      0.83      0.88       201
    
       micro avg       0.93      0.83      0.88       347
       macro avg       0.93      0.84      0.88       347
    weighted avg       0.93      0.83      0.88       347
    
    Epoch 68/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.86      0.89        72
             Met       0.92      0.82      0.87        74
             Phe       0.94      0.83      0.88       201
    
       micro avg       0.93      0.83      0.88       347
       macro avg       0.93      0.84      0.88       347
    weighted avg       0.93      0.83      0.88       347
    
    Epoch 69/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.94      0.84      0.89        74
             Phe       0.95      0.87      0.90       201
    
       micro avg       0.94      0.86      0.90       347
       macro avg       0.94      0.86      0.90       347
    weighted avg       0.94      0.86      0.90       347
    
    Epoch 70/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.86      0.89        72
             Met       0.92      0.82      0.87        74
             Phe       0.94      0.83      0.88       201
    
       micro avg       0.94      0.83      0.88       347
       macro avg       0.93      0.84      0.88       347
    weighted avg       0.94      0.83      0.88       347
    
    Epoch 71/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.93      0.86      0.89        72
             Met       0.91      0.81      0.86        74
             Phe       0.94      0.85      0.90       201
    
       micro avg       0.93      0.84      0.89       347
       macro avg       0.93      0.84      0.88       347
    weighted avg       0.93      0.84      0.89       347
    
    Epoch 72/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.91      0.81      0.86        74
             Phe       0.95      0.86      0.90       201
    
       micro avg       0.94      0.85      0.89       347
       macro avg       0.93      0.85      0.89       347
    weighted avg       0.94      0.85      0.89       347
    
    Epoch 73/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.89      0.92        72
             Met       0.91      0.81      0.86        74
             Phe       0.95      0.86      0.90       201
    
       micro avg       0.94      0.85      0.89       347
       macro avg       0.94      0.85      0.89       347
    weighted avg       0.94      0.85      0.89       347
    
    Epoch 74/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.96      0.89      0.92        72
             Met       0.91      0.81      0.86        74
             Phe       0.95      0.84      0.89       201
    
       micro avg       0.94      0.84      0.89       347
       macro avg       0.94      0.85      0.89       347
    weighted avg       0.94      0.84      0.89       347
    
    Epoch 75/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.91      0.81      0.86        74
             Phe       0.93      0.85      0.89       201
    
       micro avg       0.93      0.84      0.89       347
       macro avg       0.93      0.84      0.88       347
    weighted avg       0.93      0.84      0.89       347
    
    Epoch 76/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.91      0.81      0.86        74
             Phe       0.94      0.85      0.90       201
    
       micro avg       0.94      0.85      0.89       347
       macro avg       0.93      0.85      0.89       347
    weighted avg       0.94      0.85      0.89       347
    
    Epoch 77/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.91      0.81      0.86        74
             Phe       0.95      0.85      0.90       201
    
       micro avg       0.94      0.85      0.89       347
       macro avg       0.93      0.85      0.89       347
    weighted avg       0.94      0.85      0.89       347
    
    Epoch 78/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.91      0.81      0.86        74
             Phe       0.95      0.85      0.90       201
    
       micro avg       0.94      0.85      0.89       347
       macro avg       0.93      0.85      0.89       347
    weighted avg       0.94      0.85      0.89       347
    
    Epoch 79/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.91      0.81      0.86        74
             Phe       0.95      0.86      0.90       201
    
       micro avg       0.94      0.85      0.89       347
       macro avg       0.93      0.85      0.89       347
    weighted avg       0.94      0.85      0.89       347
    
    Epoch 80/80
    -------------------------------



      0%|          | 0/39 [00:00<?, ?it/s]



      0%|          | 0/39 [00:00<?, ?it/s]


                  precision    recall  f1-score   support
    
             Cau       0.94      0.88      0.91        72
             Met       0.91      0.81      0.86        74
             Phe       0.95      0.86      0.90       201
    
       micro avg       0.94      0.85      0.89       347
       macro avg       0.93      0.85      0.89       347
    weighted avg       0.94      0.85      0.89       347
    
    Done!



```python
sentence = '在使用过程中若发现油位指示窗内出现油面，说明波纹囊有渗漏，绝缘油进入空气腔。发现指示窗有油应马上通知厂家处理，并采取临时措施。'

results = []
with torch.no_grad():
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    pred = model.decode(inputs)
    predictions = pred[0].tolist()
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
            while (
                idx + 1 < len(predictions) and 
                id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                # all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1
            word = sentence[start:end]
            pred_label.append(
                {
                    "entity_group": label,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    print(pred_label)
```

    [{'entity_group': 'Phe', 'word': '油位指示窗内出现油面', 'start': 9, 'end': 19}, {'entity_group': 'Phe', 'word': '纹囊有渗漏', 'start': 23, 'end': 28}, {'entity_group': 'Met', 'word': '指示窗有油', 'start': 40, 'end': 45}, {'entity_group': 'Met', 'word': '采取临时措施', 'start': 56, 'end': 62}]



```python
sentence = '气体继电器保护装置的信号动作时，值班员应立即停止报警信号，并检查变压器，查明信号动作的原因，是否因空气侵入变压器内，或是油位降低，或是二次回路故障。'
results = []
with torch.no_grad():
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    pred = model.decode(inputs)
    predictions = pred[0].tolist()
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
            while (
                idx + 1 < len(predictions) and 
                id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                # all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1
            word = sentence[start:end]
            pred_label.append(
                {
                    "entity_group": label,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    print(pred_label)
```

    [{'entity_group': 'Phe', 'word': '体继电器保', 'start': 1, 'end': 6}, {'entity_group': 'Phe', 'word': '装置', 'start': 7, 'end': 9}, {'entity_group': 'Phe', 'word': '信号动作', 'start': 10, 'end': 14}, {'entity_group': 'Phe', 'word': '报警信', 'start': 24, 'end': 27}, {'entity_group': 'Cau', 'word': '空气侵入变压器', 'start': 49, 'end': 56}, {'entity_group': 'Cau', 'word': '油位降低', 'start': 60, 'end': 64}, {'entity_group': 'Cau', 'word': '二次回路故障', 'start': 67, 'end': 73}]



```python

```
