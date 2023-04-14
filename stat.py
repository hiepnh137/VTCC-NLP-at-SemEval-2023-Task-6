import matplotlib.pyplot as plt
import os
import json


def get_num_sentence(data='train'):
    dataset = json.loads(open(f'data/{data}.json', 'r').read())
    count = []
    n_words = []
    n_sent = []
    print('num_doc: ', len(dataset))
    for doc in dataset:
        count.append(len(doc['annotations'][0]['result']))  
        nw = [len(t['value']['text'].split()) for t in doc['annotations'][0]['result']]
        if nw == []:
            continue
        n_words.append(sum(nw))
        n_sent.append(len(nw))
    return count, n_words, n_sent

count, n_words, n_sent = get_num_sentence(data='train')
print('avg_word: ', sum(n_words)/len(n_words))
avg_w_sent = [w/s for w,s in zip(n_words, n_sent)]
print('avg_word_per_sent: ', sum(avg_w_sent)/len(avg_w_sent))
plt.hist(count)
plt.show()