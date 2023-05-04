import os
## importing relevant functions from transformers library that will be used
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report


MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)}

model_type = 'bert' ###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
model_name = 'law-ai/InLegalBERT'
tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def write_binary_data(data='train'):
    os.makedirs('datasets/binary/', exist_ok=True)
    doc = open(f'datasets/pubmed-20k-lsp/{data}_scibert.txt', 'r').read().split('\n')
    result = []
    prev_tag = ''
    prev_sent = ''
    for i, line in enumerate(doc):
        flag = False
        if line.strip().startswith('###') or line.strip() == '':
            result.append(line)
            prev_tag = ''
            prev_sent = ''
            continue
        if prev_sent == '':
            flag = True
        new_line = ''
        splitted_line = line.split('\t')
        tag, sent = splitted_line[0], splitted_line[1]
        if tag == prev_tag:
            new_line += '0\t'
        else:
            new_line += '1\t'
        new_line += prev_sent + ' ' + ' '.join(sent.split()[1:])
        new_line += '\t' + ' '.join([str(0)]*len(prev_sent.split())) + ' ' + ' '.join([str(1)]*len(sent.split()[1:]))
        prev_tag = tag
        prev_sent = sent
        if flag:
            continue
        result.append(new_line)
    open(f'datasets/binary/{data}.txt', 'w').write('\n'.join(result))

class BinaryDataset(Dataset):
    def __init__(self, dataset='train'):
        super().__init__()
        self.load_data(dataset)

    def load_data(self, dataset):
        self.x = []
        self.y = []
        self.t = []
        data = open(f'datasets/binary/{dataset}.txt', 'r').read().split('\n')
        for i, line in enumerate(data):
            if line.strip() == '' or line.startswith('###'):
                continue
            tag, sent, token_type = line.split('\t')
            sent = [int(t) for t in sent.split()]
            token_type = [int(t) for t in token_type.split()]
            self.y.append(int(tag))
            self.x.append(sent)
            self.t.append(token_type)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.t[index]
    

def collate(batch):
    sent = []
    label = []
    token_type = []
    for (x,  y, t) in batch:
        sent.append(torch.tensor(x))
        token_type.append(torch.tensor(t))
        label.append(y)
    sent = pad_sequence(sent, batch_first=True)
    b, n = sent.shape

    token_type = pad_sequence(token_type).transpose(1,0)
    label = torch.tensor(label)
    return sent, token_type, label

def flat_accuracy(preds, labels):
    pred_flat = torch.as_tensor(np.argmax(preds, axis=1).flatten())
    labels_flat = labels.flatten().cpu()
    return (torch.sum(pred_flat == labels_flat) / len(labels_flat)).item()

if __name__ == '__main__':
    BATCH_SIZE=32
    EPOCH = 10
    lr = 1e-6
    max_grad_norm = 1.0

    write_binary_data('train')
    write_binary_data('dev')
    write_binary_data('test')

    train_dataset = BinaryDataset('train')
    dev_dataset = BinaryDataset('dev')
    test_dataset = BinaryDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)

    num_total_steps = len(train_loader)*EPOCH
    num_warmup_steps = 1000
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)
    max_acc = -1
    p = 0
    for epoch in range(EPOCH):
        total_loss = 0
        for (x, t, y) in tqdm(train_loader):
            x = x.to(device)
            t = t.to(device)
            y = y.to(device)
            m = x!=0

            model.zero_grad()        

            outputs = model(x, token_type_ids=t, attention_mask=m, labels=y)

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()


        avg_train_loss = total_loss / len(train_loader)            
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        print("")
        print("Running Validation...")

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for (x, t, y) in dev_loader:
            x = x.to(device)
            t = t.to(device)
            y = y.to(device)

            m = x!=0        

            with torch.no_grad():   
                outputs = model(x, token_type_ids=t, attention_mask=m)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, y)
            eval_accuracy += tmp_eval_accuracy

            nb_eval_steps += 1
        avg = eval_accuracy/nb_eval_steps
        if max_acc < avg:
            p = 0
            os.makedirs('results/bertsc/', exist_ok=True)
            model.save_pretrained('results/bertsc/')
            with open('results/bertsc/acc.txt', 'w') as f:
                f.write(str(avg))
        else:
            p += 1
        if p == 5:
            break
        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

    print("")
    print("Training complete!")

    print('Predicting labels for test sentences...')
    model.eval()

    predictions , true_labels = [], []

    for (x, t, y) in dev_loader:
        x = x.to(device)
        t = t.to(device)
        y = y.to(device)

        m = x!=0        

        with torch.no_grad():   
            outputs = model(x, token_type_ids=t, attention_mask=m)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = y.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_labels.flatten()
    pred_flat = torch.as_tensor(np.argmax(predictions, axis=1).flatten())
    labels_flat = true_labels.flatten()
    print('test accuracy: ', (np.sum(pred_flat == labels_flat) / len(labels_flat)))
    print(classification_report(labels_flat, pred_flat))
