import torch
import torch.nn as nn
import pickle as pkl
import os
from transformers import BertForSequenceClassification
from tqdm import tqdm
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BertForSequenceClassification.from_pretrained('results/bertsc/', output_hidden_states=True).to(device)
def load_data(dataset='train'):
    data = open(f'datasets/binary/{dataset}.txt', 'r').read().split('\n')
    samples = []
    sent = []
    sample = []
    for line in data:
        if line.strip() == '':
            continue
        if line.startswith('###'):
            samples.append(sample)
            sample = []
            continue
        tag, sent, token_type = line.split('\t')
        sent = [int(t) for t in sent.split()]
        token_type = [int(t) for t in token_type.split()]
        tag = int(tag)
        sample.append((sent, token_type, tag))
    samples = samples[1:]
    samples.append(sample)
    return samples

def write_embedding(data='train'):
    train = load_data(data)
    embedd = []
    all_tag = []
    for i, samples in tqdm(enumerate(train)):
        doc_embed = []
        doc_tag = []
        for (sent, token_type, tag) in samples:
            sent = torch.tensor(sent).unsqueeze(0).to(device)
            token_type = torch.tensor(token_type).unsqueeze(0).to(device)
            mask = sent != 0
            with torch.no_grad():   
                outputs = model(sent, token_type_ids=token_type, attention_mask=mask)
            vec = outputs["hidden_states"][12][0][0]
            vec = vec.detach().cpu().numpy()
            
            doc_embed.append(vec)
            doc_tag.append(tag)
        doc_embed = [np.zeros_like(vec)] + doc_embed + [np.zeros_like(vec)]
        doc_tag = [0] + doc_tag + [0]
        
        embedd.append(doc_embed)
        all_tag.append(doc_tag)
        
    print(f'num of doc embedding in {data} set: ', len(embedd))
    os.makedirs('embedding/lsp_embedding/', exist_ok=True)
    with open(f'embedding/lsp_embedding/{data}.pkl', 'wb') as f:
        pkl.dump(embedd, f)
    os.makedirs('embedding/tag/', exist_ok=True)
    with open(f'embedding/tag/{data}.pkl', 'wb') as f:
        pkl.dump(all_tag, f)
        
if __name__ == '__main__':
    write_embedding('train')
    write_embedding('dev')
    write_embedding('test')


