from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
from transformers import AutoTokenizer
import pickle as pkl 
import dgl
import json
import torch


tokenizer = AutoTokenizer.from_pretrained('../../Huggingface/bert-base-uncased')
special_token = tokenizer.special_tokens_map
CLS_TOKEN = tokenizer.encode(special_token['cls_token'])[1]
SEP_TOKEN = tokenizer.encode(special_token['sep_token'])[1]


def load_glove_model(vector_size = 300):
    print("Loading Glove Model")
    glove_file = f'glove/glove.6B.{vector_size}d.txt'
    f = open(glove_file, 'r')
    # model = {}
    vocab = []
    embedding = []
    for line in f:
        split_line = line.split()
        word = " ".join(split_line[0:len(split_line) - vector_size])
        vocab.append(word)
        embedding.append([float(val) for val in split_line[-vector_size:]])
        # model[word] = embedding
    embedding = np.array(embedding)
    return vocab, embedding

def get_content_words(glove_vocab, rate=0.25):
    path = '../samples/train_samples/doc/'
    file_names = os.listdir(path)
    train_cword = []
    train_cword_id = []
    train_corpus = []
    for i, file_name in enumerate(file_names):
        if i == 69 or i ==225:
            continue
        train_corpus.append(open(path+file_name,'r').read().lower())
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(train_corpus)
    vocab = vectorizer.get_feature_names_out()
    num_cword = int(len(vocab)*rate)
    file_names.remove(file_names[69])
    file_names.remove(file_names[225])
    for i, file_name in enumerate(file_names):

        score = vectorizer.transform([train_corpus[i]]).toarray()
        num_words = int(len(set(train_corpus[i].split()))*rate)
        score = score[0].tolist()
        sorted_index = np.argsort(score)[::-1]
        train_cword.append([])
        train_cword_id.append([])
        for j in sorted_index:
            if vocab[j] in glove_vocab:
                train_cword[-1].append(vocab[j])
                train_cword_id[-1].append(glove_vocab.index(vocab[j]))
#             flag = False
#             for k in range(len(glove_vocab)):
#                 if glove_vocab[k] == vocab[j]:
#                     flag = True
#                     break
#             if flag:
#                 train_cword[-1].append(vocab[j]) 
#                 train_cword_id[-1].append(k)
            if j == num_cword or score[j]==0:
                break
    dev_cword = []
    dev_cword_id = []
    path = '../samples/dev_samples/doc/'
    file_names = os.listdir(path)
    for i, file_name in enumerate(file_names):
        doc = open(path+file_name, 'r').read().lower()
        score = vectorizer.transform([doc]).toarray()
        num_words = int(len(set(doc.split()))*rate)
        score = score[0].tolist()
        sorted_index = np.argsort(score)[::-1][:num_words]
        dev_cword.append([])
        dev_cword_id.append([])
        for j in sorted_index:
            if vocab[j] in glove_vocab:
                dev_cword[-1].append(vocab[j])
                dev_cword_id[-1].append(glove_vocab.index(vocab[j]))
#             flag = False
#             for k in range(glove_vocab):
#                 if glove_vocab[k] == vocab[j]:
#                     flag = True
#                     break
#             if flag:
#                 dev_cword[-1].append(vocab[j]) 
#                 dev_cword_id[-1].append(k)
            if j == num_cword or score[j]==0:
                break    
    return train_cword, dev_cword, train_cword_id, dev_cword_id, vocab

def encode_sentence(sentence):
    sentence = sentence.lower()
    encoded_seq = []
    words = tokenizer.tokenize(sentence)
    index = 1
    pos = {}
    for word in words:
        if word not in pos.keys():
            pos[word] = []
        encoded_token = tokenizer.encode(word)[1:-1]
        encoded_seq.extend(encoded_token)
        pos[word].append((index, index+len(encoded_token)))
        index += len(encoded_token)
    encoded_seq = [CLS_TOKEN] + encoded_seq + [SEP_TOKEN]
    return words, pos, encoded_seq


def create_graph(data='train', add_wnode=True):
    if add_wnode:
        cword = pkl.load(open(f'cword/{data}_cword.pkl', 'rb'))
        
    dataset = json.loads(open(f'data/{data}.json', 'r').read())
    graph_list = []
    all_cluster = []
    cluster = []
    paragraph = []
    i=0
    for doc in dataset:
        all_cluster = []
        cluster = []
        num_sentences = len(doc['annotations'][0]['result'])
        if len(doc['annotations'][0]['result']) == 0:
            print(f'error doc in {data} set: ', i)
            continue
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
        
        G = dgl.DGLGraph()
        G.add_nodes(num_sentences)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.ones(num_sentences)
        G.ndata["id"] = torch.LongTensor(list(range(num_sentences)))
        G.ndata["dtype"] = torch.ones(num_sentences)
        
        n_para = G.num_nodes()
        G.add_nodes(num_para)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"][n_para:] = 3*torch.ones(num_para)
        G.ndata["id"][n_para:] = 3*torch.LongTensor(list(range(num_para)))
        G.ndata["dtype"][n_para:] = 3*torch.ones(num_para)
        
        u = []
        v = []
        for k in range(num_sentences):
            for para, c in enumerate(all_cluster):
                if k in c:
                    for c_i in c:
                        if k == c_i:
                            continue
                        u.append(k)
                        v.append(c_i)
                
        u = torch.tensor(u)
        v = torch.tensor(v)
        G = dgl.add_edges(G, u, v, data={"dtype": torch.ones(len(u))})
        
        #add word2para
        u = []
        v = []
        for k in range(num_sentences):
            for para, c in enumerate(all_cluster):
                if k in c:
                    u.append(k)
                    v.append(para+n_para)
                
        u = torch.tensor(u)
        v = torch.tensor(v)
        G = dgl.add_edges(G, u, v, data={"dtype": 3*torch.ones(len(u))})   #sent2para
        G = dgl.add_edges(G, u, v, data={"dtype": -3*torch.ones(len(u))})   #para2sent
        
        
        #add word node
        if add_wnode:
            u = []
            v = []
            num_nodes = G.num_nodes()
            n_wnodes = len(cword[i])
            G.add_nodes(n_wnodes)
            G.ndata["unit"][num_nodes:] = 2*torch.ones(n_wnodes)
            G.ndata["id"][num_nodes:] = 2*torch.LongTensor(list(range(n_wnodes)))
            G.ndata["dtype"][num_nodes:] = 2*torch.ones(n_wnodes)
            for j, s in enumerate(doc['annotations'][0]['result']):
                text = s['value']['text'].lower()
                text = text.split()
                for k, cw in enumerate(cword[i]):
                    if cw in text:
                        u.append(j)
                        v.append(k+num_nodes)
            G = dgl.add_edges(G, u, v, data={"dtype": 2*torch.ones(len(u))})     #sent2word
            G = dgl.add_edges(G, v, u, data={"dtype": -2*torch.ones(len(u))})    #word2sent
        graph_list.append(G)
        i+=1

    return graph_list, paragraph

def convert_paragraph_to_array(paragraph):
    onehot_list = []
    for para in paragraph:
        n_sentence = sum([len(p) for p in para])
        onehot = np.zeros([len(para), n_sentence])
        for i, p in enumerate(para):
            for j in p:
                onehot[i][j] = 1
        onehot_list.append(onehot)
    return onehot_list
    

glove = pkl.load(open('glove/glove_300.pkl', 'rb'))
glove_vocab = glove['vocab']

train_cword, dev_cword, train_cword_id, dev_cword_id, vocab = get_content_words(glove_vocab)
with open('cword/train_cword.pkl', 'wb') as f:
    pkl.dump(train_cword, f)
with open('cword/dev_cword.pkl', 'wb') as f:
    pkl.dump(dev_cword, f)
with open('cword/train_cword_id.pkl', 'wb') as f:
    pkl.dump(train_cword_id, f)
with open('cword/dev_cword_id.pkl', 'wb') as f:
    pkl.dump(dev_cword_id, f)
    
with open('cword/test_cword.pkl', 'wb') as f:
    pkl.dump(dev_cword, f)
with open('cword/test_cword_id.pkl', 'wb') as f:
    pkl.dump(dev_cword_id, f)
        
train_graph, train_paragraph = create_graph(data='train', add_wnode=True)
dev_graph, dev_paragraph = create_graph(data='dev', add_wnode=True)

# train_paragraph_onehot = convert_paragraph_to_array(train_paragraph)
# dev_paragraph_onehot = convert_paragraph_to_array(dev_paragraph)

print(dev_graph[0])
with open('graph/sentence-word/train_graph.pkl', 'wb') as f:
    pkl.dump(train_graph, f)
with open('graph/sentence-word/dev_graph.pkl', 'wb') as f:
    pkl.dump(dev_graph, f)
with open('graph/sentence-word/test_graph.pkl', 'wb') as f:
    pkl.dump(dev_graph, f)

os.makedirs('graph/paragraph/', exist_ok=True)
with open('graph/paragraph/train_paragraph.pkl', 'wb') as f:
    pkl.dump(train_paragraph, f)
with open('graph/paragraph/dev_paragraph.pkl', 'wb') as f:
    pkl.dump(dev_paragraph, f)
with open('graph/paragraph/test_paragraph.pkl', 'wb') as f:
    pkl.dump(dev_paragraph, f)


# print(train_cword)

# train_cword = pkl.load(open('cword/train_cword.pkl', 'rb'))
# train_cword_id = pkl.load(open('cword/train_cword_id.pkl', 'rb'))
# for i in range(len(train_cword)):
#     if len(train_cword[i]) != len(train_cword_id[i]):
#         print(i)
# train_cword = pkl.load(open('cword/train_cword.pkl', 'rb'))
# train_graph = pkl.load(open('graph/sentence-word/train_graph.pkl', 'rb'))
# if len(train_cword) == len(train_graph):
#     print('ok')
# for i in range(len(train_cword)):
#     if len(train_cword) != train_graph[i]
