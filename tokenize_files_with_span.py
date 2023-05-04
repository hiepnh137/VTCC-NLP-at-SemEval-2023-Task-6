import os
import numpy as np
from transformers import AutoTokenizer
import pickle as pkl 
import json
# import en_legal_ner_sm
from tqdm import tqdm
import re
import spacy
config = pkl.load(open('spacy/config.pkl', 'rb'))
bytes_data = pkl.load(open('spacy/bytes_data.pkl', 'rb'))
lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
nlp = lang_cls.from_config(config)
nlp.from_bytes(bytes_data)
# nlp = en_legal_ner_sm.load()
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
def get_data(data='train'):
    dataset = json.loads(open(f'data/{data}.json', 'r').read())
    filtered_dataset = []
    for i, doc in tqdm(enumerate(dataset)):
        if len(doc['annotations'][0]['result']) == 0 or len(doc['annotations'][0]['result']) > 350:
            print(f'error doc in set: ', i)
            continue
        filtered_dataset.append(doc)
    return filtered_dataset


def get_sentence_corpus(data='train'):
    with open(f'datasets/subwords/{data}.pkl', 'rb') as f:
        words = pkl.load(f)
    words = words['words']
    corpus = []
    for doc in words:
        d = []
        for sent in doc:
            d.append(' '.join(sent))
        corpus.append(d)
    return corpus


def get_corpus(data='train'):
    dataset = get_data(data)
    corpus = []
    for doc in dataset:
        corpus.append(doc['data']['text'])
    return corpus

def get_entity_sentence(sentence):
    doc = nlp(sentence)
    entity = []
    tag = []
    for ent in doc.ents:
        entity.append(str(ent))
        tag.append(str(ent.label_))
    return entity, tag

def entity_to_dict(entities, tags):
    tag_dict = {}
    for e, t in zip(entities, tags):
        if t not in tag_dict:
            tag_dict[t] = []
        if e not in tag_dict[t]:
            tag_dict[t].append(e)
    for t in tag_dict.keys():
        tag_dict[t] = sorted(tag_dict[t], key= lambda t: len(t))[::-1]
    return tag_dict

def get_entity(data='train'):
    # corpus  = get_sentence_corpus(data)
    corpus = get_corpus(data)
    all_entity_map = []
    for doc in corpus:
        doc_entity, doc_tag = get_entity_sentence(doc)
        all_entity_map.append(entity_to_dict(doc_entity, doc_tag))
    os.makedirs('datasets/entity/', exist_ok=True)
    pkl.dump(all_entity_map, open(f'datasets/entity/{data}.pkl', 'wb'))
    return all_entity_map


def get_span(subword_ids):
    span = []
    prev = 0
    start = 1
    end = 0
    for i, id in enumerate(subword_ids):
        if i == 0:
            continue
        if id == None and i != len(subword_ids)-1:
            print('error: ', subword_ids)
        if id == None:
            end = i
            span.append((start, end))
            continue
        if id != prev and prev != None:
            end = i
            span.append((start, end))
            start = i
        prev = id
    return span


def bert_tokenize(data_file):
    data = get_data(data_file)
    entity_dict = pkl.load(open(f'datasets/entity/{data_file}.pkl', 'rb'))
    final_string = ''
    subword_list = []
    subword_span_list = []
    entity_list = []
    entity_span_list = []
    for i, doc in tqdm(enumerate(data)):
        subword = []
        subword_span = []
        doc_entity, doc_entity_span = [], []
        file_name = doc['id']
        final_string = final_string + '###' + str(file_name) + "\n"
        sentences = doc['annotations'][0]['result']
        all_tags = entity_dict[i].keys()
        if len(doc['annotations'][0]['result']) == 0:
            print(f'error doc in set: ', i)
            continue
        for s in sentences:
            s = s['value']
            sent = s['text'].replace("\r", "")
            #replace entity with special token 
            for tag, entities in entity_dict[i].items():
                for j, entity in enumerate(entities):
                    # print(entity)
                    replace_token = f'{tag}{j}'
                    sent = sent.replace(entity, replace_token)

            label = s['labels'][0]
            if sent.strip() != "":
                tokens = tokenizer(sent, add_special_tokens=True, max_length=128)
                sent_tokens = tokens['input_ids']
                words = tokenizer.decode(sent_tokens).split()[1:-1]
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + label + "\t" + sent_tokens_txt + "\n"

                subword_ids = get_span(tokens.word_ids())
                words = [tokenizer.decode(tokens['input_ids'][s:e]) for (s, e) in subword_ids]

                entities = []
                entity_spans = []
                for j, w in enumerate(words):
                    for t in all_tags:
                        span = re.search(f'{t.lower()}[0-9]+', w)
                        if span is not None:
                            span = span.span()
                            if span[0] == 0 and span[1] == len(w):
                                entities.append(w)
                                entity_spans.append(subword_ids[j])
                subword.append(words)
                subword_span.append(subword_ids)
                doc_entity.append(entities)
                doc_entity_span.append(entity_spans)
            
        final_string = final_string + "\n"
        subword_list.append(subword)
        subword_span_list.append(subword_span)
        entity_list.append(doc_entity)
        entity_span_list.append(doc_entity_span)
    save = {'words': subword_list, 'spans': subword_span_list, 'entities': entity_list, 'entity_spans': entity_span_list}
    os.makedirs('datasets/subwords/', exist_ok=True)
    with open(f'datasets/subwords/{data_file}.pkl', 'wb') as f:
        pkl.dump(save, f)
    with open(f'datasets/pubmed-20k/{data_file}_scibert.txt' , "w+") as file:
        file.write(final_string)
# data = get_data('train')
# train_entity_dict = get_entity('train')
# dev_entity_dict = get_entity('dev')
# test_entity_dict = get_entity('test')
if __name__ == '__main__':
    train_entity_dict = get_entity('train')
    dev_entity_dict = get_entity('dev')
    test_entity_dict = get_entity('test')
    bert_tokenize('train')
    bert_tokenize('dev')
    bert_tokenize('test')
