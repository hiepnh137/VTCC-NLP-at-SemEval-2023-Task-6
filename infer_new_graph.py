import os
import models
import json
import sys
import pickle as pkl
import torch
from transformers import BertTokenizer

import models
from eval import eval_model
from models import BertHSLN
from task_infer import pubmed_task
from utils import get_device
from tokenize_files_with_span import bert_tokenize, get_entity
from infer_BertSC import write_embedding
from create_entity_graph import get_content_words_file, create_graph
import warnings
warnings.filterwarnings('ignore')
def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS)


def infer(model_path, max_docs, prediction_output_json_path, device):
    ######### This function loads the model from given model path and predefined data. It then predicts the rhetorical roles and returns
    task = create_task(pubmed_task)
    model = getattr(models, config["model"])(config, [task]).to(device)
    model.load_state_dict(torch.load(model_path))
    model = model
    model.eval()
    folds = task.get_folds()
    test_batches = folds[0].test
    metrics, confusion,labels_dict, class_report, lsp_metrics, lsp_confusion, lsp_class_report = eval_model(model, test_batches, device, task)
    return labels_dict

def write_in_hsln_format(input_json,hsln_format_txt_dirpath,tokenizer):

    #tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    json_format = json.load(open(input_json))
    final_string = ''
    filename_sent_boundries = {}
    for file in json_format:
        file_name=file['id']
        final_string = final_string + '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}
        for annotation in file['annotations'][0]['result']:
            filename_sent_boundries[file_name]['sentence_span'].append([annotation['value']['start'],annotation['value']['end']])

            sentence_txt=annotation['value']['text']
            sentence_txt = sentence_txt.replace("\r", "")
            if sentence_txt.strip() != "":
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=128)
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + "NONE" + "\t" + sent_tokens_txt + "\n"
        final_string = final_string + "\n"

    with open(hsln_format_txt_dirpath + '/test_scibert.txt', "w") as file:
        file.write(final_string)

#     with open(hsln_format_txt_dirpath + '/train_scibert.txt', "w") as file:
#         file.write(final_string)
#     with open(hsln_format_txt_dirpath + '/dev_scibert.txt', "w") as file:
#         file.write(final_string)
    with open(hsln_format_txt_dirpath + '/sentece_boundries.json', 'w') as json_file:
        json.dump(filename_sent_boundries, json_file)

    return filename_sent_boundries

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
#     open(f'datasets/binary/train.txt', 'w').write('\n'.join(result))
#     open(f'datasets/binary/dev.txt', 'w').write('\n'.join(result))
    open(f'datasets/binary/test.txt', 'w').write('\n'.join(result))

if __name__=="__main__":
    [_,input_dir, prediction_output_json_path, model_path] = sys.argv
    
    config = json.loads(open(f'{model_path[:-12]}config.json', 'r').read())
    BERT_VOCAB = config['bert_model']
    
#     BERT_VOCAB = "../../Huggingface/bert-base-cased/"
    BERT_MODEL = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

#     config = {
#         "bert_model": BERT_MODEL,
#         "bert_trainable": False,
#         "model": BertHSLN.__name__,
#         "cacheable_tasks": [],

#         "dropout": 0.5,
#         "word_lstm_hs": 758,
#         "att_pooling_dim_ctx": 200,
#         "att_pooling_num_ctx": 15,

#         "lr": 3e-05,
#         "lr_epoch_decay": 0.9,
#         "batch_size": 32,
#         "max_seq_length": 128,
#         "max_epochs": 40,
#         "early_stopping": 5,

#     }

    
    MAX_DOCS = -1
    device = get_device(0)
    
    hsln_format_txt_dirpath ='datasets/pubmed-20k-lsp'
    write_in_hsln_format(input_dir,hsln_format_txt_dirpath,tokenizer)
    write_binary_data('test')

    filename_sent_boundries = json.load(open(hsln_format_txt_dirpath + '/sentece_boundries.json'))
    print('get_entity')
    get_entity('test')

    bert_tokenize('test')
    
    print('write embedding lsp')
    write_embedding('test')

    from infer_Bert import write_embedding
    print('write embedding bert')
    write_embedding('test')

    test_cwords = get_content_words_file('test', rate=1)
    test_graph_list = create_graph('test', rate=1, ngram=3, window=5)


    predictions = infer(model_path, MAX_DOCS, prediction_output_json_path, device)
    
    ##### write the output in format needed by revision script
    for doc_name,predicted_labels in zip(predictions['doc_names'],predictions['docwise_y_predicted']):
        filename_sent_boundries[doc_name]['pred_labels'] = predicted_labels
    with open(input_dir,'r') as f:
        input=json.load(f)
    for file in input:
        id=str(file['id'])
        pred_id=predictions['doc_names'].index(id)
        pred_labels=predictions['docwise_y_predicted']
        annotations=file['annotations']
        for i,label in enumerate(annotations[0]['result']):

            label['value']['labels']=[pred_labels[pred_id][i]]

    with open(prediction_output_json_path,'w') as file:
        json.dump(input,file)
