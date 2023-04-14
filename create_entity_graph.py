from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
from transformers import AutoTokenizer
import pickle as pkl 
import dgl
import json
import torch
import spacy
# import en_legal_ner_sm
from tqdm import tqdm
import nltk 
import string

punctuation = list(string.punctuation)
# nlp = en_legal_ner_sm.load()

def get_data(data='train'):
    dataset = json.loads(open(f'data/{data}.json', 'r').read())
    filtered_dataset = []
    for i, doc in tqdm(enumerate(dataset)):
        if len(doc['annotations'][0]['result']) == 0 or len(doc['annotations'][0]['result']) > 350:
            print(f'error doc in set: ', i)
            continue
        filtered_dataset.append(doc)
    return filtered_dataset
    
def get_tokenized_data(data='train'):
    return pkl.load(open(f'datasets/subwords/{data}.pkl', 'rb'))

def get_corpus(data='train'):
    with open(f'datasets/subwords/{data}.pkl', 'rb') as f:
        words = pkl.load(f)
    words = words['words']
    corpus = []
    for doc in words:
        d = ''
        for sent in doc:
            d += ' '.join(sent) + ' '
        d = d.strip()
        corpus.append(d)
    return corpus

def get_subword_span(data='train'):
    with open(f'datasets/subwords/{data}.pkl', 'rb') as f:
        subwords = pkl.load(f)
    words = subwords['words']
    span = subwords['spans']
    return subwords, spans
    
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

def flatten(doc):
    flat = []
    for sent in doc:
        flat.extend(sent)
    return flat

def get_content_words(rate=0.25):
    #train tfidf with train dataset
    corpus = get_corpus('train')
    tokenized_data = get_tokenized_data('train')

    entity = tokenized_data['entities']
    entity_span = tokenized_data['entity_spans']
    words = tokenized_data['words']
    word_spans = tokenized_data['spans']
    # print('word_spans: ', word_spans[0][1])
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(corpus)
    vocab = vectorizer.get_feature_names_out()
    num_cword = int(len(vocab)*rate)

    #get cword in each sentence
    train_cwords = []
    train_cword_spans = []
    for i, doc in enumerate(words):
        cword = []
        cword_span = []
        for j, sent in enumerate(doc):
            # sent_cword = []
            # sent_cword_span = []
            sent_cword = {}
            for k, w in enumerate(sent):
                if w in sent_cword.keys():
                    sent_cword[w].append(word_spans[i][j][k])
                    continue
                # for m in range(len(doc_cword[i])):
                #     if w == doc_cword[i][m]:
                #         if w not in sent_cword:
                sent_cword[w] = []
                sent_cword[w].append(word_spans[i][j][k])
            cw = []
            span = []
            for k, v in sent_cword.items():
                cw.append(k)
                span.append(v)
            cword.append(cw)
            cword_span.append(span)
        train_cwords.append(cword)
        train_cword_spans.append(cword_span)
    
    train_entity_masks = get_entity_masks(train_cwords, entity)

    #get cword in dev dataset
    corpus = get_corpus('dev')
    tokenized_data = get_tokenized_data('dev')

    entity = tokenized_data['entities']
    entity_span = tokenized_data['entity_spans']
    words = tokenized_data['words']
    word_spans = tokenized_data['spans']

    dev_cwords = []
    dev_cword_spans = []
    for i, doc in enumerate(words):
        cword = []
        cword_span = []
        for j, sent in enumerate(doc):
            # sent_cword = []
            # sent_cword_span = []
            sent_cword = {}
            for k, w in enumerate(sent):
                if w in sent_cword.keys():
                    sent_cword[w].append(word_spans[i][j][k])
                    continue
                # for m in range(len(doc_cword[i])):
                #     if w == doc_cword[i][m]:
                #         if w not in sent_cword:
                sent_cword[w] = []
                sent_cword[w].append(word_spans[i][j][k])
            cw = []
            span = []
            for k, v in sent_cword.items():
                cw.append(k)
                span.append(v)
            cword.append(cw)
            cword_span.append(span)
        dev_cwords.append(cword)
        dev_cword_spans.append(cword_span)

    dev_entity_masks = get_entity_masks(dev_cwords, entity)
    
    # print('dev_entity_masks: ', dev_entity_masks)

    os.makedirs(f'cword/{rate}', exist_ok=True)
    save_train = {'cwords': train_cwords, 'spans': train_cword_spans, 'entity_masks': train_entity_masks}
    save_dev = {'cwords': dev_cwords, 'spans': dev_cword_spans, 'entity_masks': dev_entity_masks}
    with open(f'cword/{rate}/train_cword.pkl', 'wb') as f:
        pkl.dump(save_train, f)
    with open(f'cword/{rate}/dev_cword.pkl', 'wb') as f:
        pkl.dump(save_dev, f)
    with open(f'cword/{rate}/test_cword.pkl', 'wb') as f:
        pkl.dump(save_dev, f)
    with open(f'cword/{rate}/tfidf.pkl', 'wb') as f:
        pkl.dump(vectorizer, f)
    return save_train, save_dev
    

def get_content_words_file(data='dev', rate=0.25):
    #get cword in dev dataset
    corpus = get_corpus(data)
    tokenized_data = get_tokenized_data(data)

    entity = tokenized_data['entities']
    entity_span = tokenized_data['entity_spans']
    words = tokenized_data['words']
    word_spans = tokenized_data['spans']

    dev_cwords = []
    dev_cword_spans = []
    for i, doc in enumerate(words):
        cword = []
        cword_span = []
        for j, sent in enumerate(doc):
            # sent_cword = []
            # sent_cword_span = []
            sent_cword = {}
            for k, w in enumerate(sent):
                if w in sent_cword.keys():
                    sent_cword[w].append(word_spans[i][j][k])
                    continue
                # for m in range(len(doc_cword[i])):
                #     if w == doc_cword[i][m]:
                #         if w not in sent_cword:
                sent_cword[w] = []
                sent_cword[w].append(word_spans[i][j][k])
            cw = []
            span = []
            for k, v in sent_cword.items():
                cw.append(k)
                span.append(v)
            cword.append(cw)
            cword_span.append(span)
        dev_cwords.append(cword)
        dev_cword_spans.append(cword_span)

    dev_entity_masks = get_entity_masks(dev_cwords, entity)
    
    # print('dev_entity_masks: ', dev_entity_masks)

    os.makedirs(f'cword/{rate}', exist_ok=True)
    save_dev = {'cwords': dev_cwords, 'spans': dev_cword_spans, 'entity_masks': dev_entity_masks}
    with open(f'cword/{rate}/{data}_cword.pkl', 'wb') as f:
        pkl.dump(save_dev, f)
    print(f'cword/{rate}/{data}_cword.pkl')
    return save_dev

def get_entity_masks(cwords, entities):
    entity_masks = []
    for i, doc in enumerate(cwords):
        doc_entity_masks = []
        for j, sent in enumerate(doc):
            sent_entity_masks = []
            for w in sent:
                if w in entities[i][j]:
                    sent_entity_masks.append(1)
                else:
                    sent_entity_masks.append(0)
            doc_entity_masks.append(sent_entity_masks)
        entity_masks.append(doc_entity_masks)
    return entity_masks


def get_duplicated_word_connection(flatten_cwords, flatten_entity_masks):
    index = list(range(len(flatten_cwords)))
    u, v = [], []
    for i, cword in enumerate(flatten_cwords):
        count = 0
        if i == len(flatten_cwords)-1:
            continue
        if flatten_entity_masks[i] == 0:
            continue
        for j, other in enumerate(flatten_cwords[i+1:]):
            if cword == other:
                u.extend([i,j])
                v.extend([j, i])
                count += 1
    return u, v

def get_ngrams(sentences, n=3):
    ngrams = []
    for i, sent in enumerate(sentences):
        ngrams.append([t for t in nltk.ngrams(sent.split(), n)])
    return ngrams


def dup_ngram(ngrams1, ngrams2):
    for n1 in ngrams1:
        for n2 in ngrams2:
            if n1 == n2:
                return True
    return False

def get_sentence_connections(sentences, ngram=3):
    ngrams = get_ngrams(sentences=sentences, n=ngram)
    u = []
    v = []
    for i, sent in enumerate(sentences):
        if i == len(sentences) - 1:
            continue
        for j, o_sent in enumerate(sentences[i+1:]):
            if j == 15:
                break
            if dup_ngram(ngrams[i], ngrams[j+i+1]):
                u.extend([i, i+j+1])
                v.extend([i+j+1, i])
    return u, v

def dup_ngram_with_window(ngrams1, ngrams2):
    for n1 in ngrams1:
        # if not is_entity(n1):
        #     continue
        flag = False
        for s in n1:
            if s in punctuation:
                flag = True
                break
        if flag:
            continue
        for n2 in ngrams2:
            if n1 == n2:
                return True
    return False

def get_sentence_connections_with_window(sentences, ngram=3, window=5):
    ngrams = get_ngrams(sentences=sentences, n=ngram)
    u = []
    v = []
    for i, sent in enumerate(sentences):
        if i == len(sentences) - 1:
            continue
        for j, o_sent in enumerate(sentences[i+1:]):
            if j == window:
                break
            if dup_ngram_with_window(ngrams[i], ngrams[j+i+1]):
                u.extend([i, i+j+1])
                v.extend([i+j+1, i])
    return u, v
    
def create_graph(data='train', rate=0.25, ngram=3, window=5):
    content_word = pkl.load(open(f'cword/{rate}/{data}_cword.pkl', 'rb'))
    cwords = content_word['cwords'] 
    # print(cwords[0])
    cword_spans = content_word['spans']
    entity_masks = content_word['entity_masks']
    words = pkl.load(open(f'datasets/subwords/{data}.pkl', 'rb'))['words']
    dataset = get_data(data)

    graph_list = []
    all_cluster = []
    cluster = []
    paragraph = []
    for i, doc in tqdm(enumerate(dataset)):
        flatten_cwords = flatten(cwords[i])
        flatten_entity_masks = flatten(entity_masks[i])
        all_cluster = []
        cluster = []
        num_sentences = len(doc['annotations'][0]['result'])
        for j, s in enumerate(doc['annotations'][0]['result']):
            value = s['value']
            if value['text'].startswith('\n'):
                all_cluster.append(cluster)
                cluster = [j]
            else:
                cluster.append(j)
        all_cluster.append(cluster)
        paragraph.append(all_cluster)
        
        num_para = len(all_cluster)
        if len(cwords[i]) != num_sentences:
            print('error')
        G = dgl.DGLGraph()
        G.add_nodes(num_sentences)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.ones(num_sentences)
        G.ndata["id"] = torch.LongTensor(list(range(num_sentences)))
        G.ndata["dtype"] = torch.ones(num_sentences)
        
        #add sentence edge
        sentences = [' '.join(s) for s in words[i]]
        u_s, v_s = get_sentence_connections_with_window(sentences, ngram=ngram, window=window)
        G = dgl.add_edges(G, u_s, v_s, data={"dtype": 1*torch.ones(len(u_s))})    #sent2sent
        #add word node
        u_ws, v_ws = [], []
        u_ww, v_ww = [], []
        u_dup, v_dup = [], []
        # print('cword: ', cwords[i])

        for j in range(len(cwords[i])):
            n_nodes = G.num_nodes()
            n_sent_cword = len(cwords[i][j])
            G.add_nodes(n_sent_cword)
            G.ndata["unit"][n_nodes:] = 2*torch.ones(n_sent_cword)
            G.ndata["id"][n_nodes:] = 2*torch.LongTensor(list(range(n_sent_cword)))
            G.ndata["dtype"][n_nodes:] = 2*torch.ones(n_sent_cword)
            
            #add w2s edge
            for k in range(n_sent_cword):
                u_ws.append(n_nodes+k)
                v_ws.append(j)

            #add w2w edge
            for k in range(n_sent_cword-1):
                u_ww.append(n_nodes+k)
                v_ww.append(n_nodes+k+1)
        all_nodes = G.num_nodes()
        u, v = get_duplicated_word_connection(flatten_cwords, flatten_entity_masks)
        # print(f'num of dup connection in {data} set: ', len(u)/2/num_sentences)
        u_dup = [(index+num_sentences) for index in u]
        v_dup = [(index+num_sentences) for index in v]

        G = dgl.add_edges(G, u_ww, v_ww, data={"dtype": 2*torch.ones(len(u_ww))})    #word2word
        # G = dgl.add_edges(G, v_ww, u_ww, data={"dtype": 2*torch.ones(len(u_ww))})    #word2word
        G = dgl.add_edges(G, u_dup, v_dup, data={"dtype": 2*torch.ones(len(u_dup))})    #duplicate
        G = dgl.add_edges(G, u_ws, v_ws, data={"dtype": 3*torch.ones(len(u_ws))})     #word2sent
        max_node = max(max(u_ww), max(v_ww), max(u_dup), max(v_dup), max(u_ws), max(v_ws), max(u_s), max(v_s))
        if max_node > all_nodes:
            print('error')
        graph_list.append(G)

    os.makedirs('graph/entity_graph/', exist_ok=True)
    with open(f'graph/entity_graph/{data}_graph_{rate}.pkl', 'wb') as f:
        pkl.dump(graph_list, f)
    return graph_list

if __name__ == '__main__':
    rate = 1
    ngram = 3
    window = 5
    get_content_words(rate=rate)
    train_graph_list = create_graph('train', rate=rate, ngram=ngram, window=window)
    dev_graph_list = create_graph('dev', rate=rate, ngram=ngram, window=window)
    with open(f'graph/entity_graph/dev_graph_{rate}.pkl', 'rb') as f:
        dev_graph_list = pkl.load(f)
    with open(f'graph/entity_graph/test_graph_{rate}.pkl', 'wb') as f:
        pkl.dump(dev_graph_list, f)
    import random
#     id = random.randint(0, len(dev_graph_list))
#     print(id)
#     print(dev_graph_list[id])