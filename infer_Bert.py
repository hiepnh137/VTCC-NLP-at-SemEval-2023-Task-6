import torch
import torch.nn as nn
import pickle as pkl
import os
from transformers import BertModel
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BertModel.from_pretrained("/storage-nlp/huggingface/law-ai/InLegalBERT").to(device)

def load_data(dataset='train'):
    data = open(f'datasets/pubmed-20k-lsp/{dataset}_scibert.txt', 'r').read().split('\n')
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
        tag, sent = line.split('\t')
        sent = [int(t) for t in sent.split()]
        sample.append(sent)
    samples = samples[1:]
    samples.append(sample)
    return samples


def write_embedding(data='train'):
    train = load_data(data)

    embedd = []
    for i, samples in tqdm(enumerate(train)):
        doc_embed = []
        for sent in samples:
            sent = torch.tensor(sent).unsqueeze(0).to(device)
            token_type = torch.ones_like(sent).to(device)
            mask = sent != 0
            with torch.no_grad():   
                outputs = model(sent, token_type_ids=token_type, attention_mask=mask)
            vec = outputs[0].squeeze()[0].flatten()
            vec = vec.detach().cpu().numpy()
            doc_embed.append(vec)
        embedd.append(doc_embed)
    print(f'num of doc embedding in {data} set: ', len(embedd))
    os.makedirs('embedding/sentence_embedding/', exist_ok=True)
    with open(f'embedding/sentence_embedding/{data}.pkl', 'wb') as f:
        pkl.dump(embedd, f)
if __name__ == '__main__':
    write_embedding(data='train')
    write_embedding(data='dev')
    write_embedding(data='test')