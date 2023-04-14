from allennlp.common.util import pad_sequence_to_length
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import masked_mean, masked_softmax
import copy

from transformers import BertModel

from allennlp.modules import ConditionalRandomField

import torch
import torch.nn as nn
import math
import pickle as pkl
from lsp_model import LSTM_Emitter_Binary
from module.GAT import WSWGAT
import random
from task import GEN_LABELS

class CRFOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, num_labels):
        super(CRFOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)
        self.crf = ConditionalRandomField(self.num_labels)

    def forward(self, x, mask, labels=None):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''

        batch_size, max_sequence, in_dim = x.shape

        logits = self.classifier(x)
        outputs = {}
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask)
            loss = -log_likelihood
            outputs["loss"] = loss
        else:
            best_paths = self.crf.viterbi_tags(logits, mask)
            predicted_label = [x for x, y in best_paths]
            predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predicted_label]
            predicted_label = torch.tensor(predicted_label)
            outputs["predicted_label"] = predicted_label

            #log_denominator = self.crf._input_likelihood(logits, mask)
            #log_numerator = self.crf._joint_likelihood(logits, predicted_label, mask)
            #log_likelihood = log_numerator - log_denominator
            #outputs["log_likelihood"] = log_likelihood

        return outputs

class CRFPerTaskOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, tasks):
        super(CRFPerTaskOutputLayer, self).__init__()

        self.per_task_output = torch.nn.ModuleDict()
        for task in tasks:
            self.per_task_output[task.task_name] = CRFOutputLayer(in_dim=in_dim, num_labels=len(task.labels))


    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.per_task_output[task](x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for t, task_output in self.per_task_output.items():
                task_result = task_output(x, mask)
                task_result["task"] = t
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, task in enumerate(self.per_task_output.keys()):
            if index % 2 == 0:
                self.task_to_device[task] = device1
                self.per_task_output[task].to(device1)
            else:
                self.task_to_device[task] = device2
                self.per_task_output[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]



class AttentionPooling(torch.nn.Module):
    def __init__(self, in_features, dimension_context_vector_u=200, number_context_vectors=5):
        super(AttentionPooling, self).__init__()
        self.dimension_context_vector_u = dimension_context_vector_u
        self.number_context_vectors = number_context_vectors
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=self.dimension_context_vector_u, bias=True)
        self.linear2 = torch.nn.Linear(in_features=self.dimension_context_vector_u,
                                       out_features=self.number_context_vectors, bias=False)

        self.output_dim = self.number_context_vectors * in_features

    def forward(self, tokens, mask):
        #shape tokens: (batch_size, tokens, in_features)

        # compute the weights
        # shape tokens: (batch_size, tokens, dimension_context_vector_u)
        a = self.linear1(tokens)
        a = torch.tanh(a)
        # shape (batch_size, tokens, number_context_vectors)
        a = self.linear2(a)
        # shape (batch_size, number_context_vectors, tokens)
        a = a.transpose(1, 2)
        a = masked_softmax(a, mask)

        # calculate weighted sum
        s = torch.bmm(a, tokens)
        s = s.view(tokens.shape[0], -1)
        return s



class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        # state_dict_1 = self.bert.state_dict()
        # state_dict_2 = torch.load('/home/astha_agarwal/model/pytorch_model.bin')
        # for name2 in state_dict_2.keys():
        #    for name1 in state_dict_1.keys():
        #        temp_name = copy.deepcopy(name2)
        #       if temp_name.replace("bert.", '') == name1:
        #            state_dict_1[name1] = state_dict_2[name2]

        #self.bert.load_state_dict(state_dict_1,strict=False)

        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]
        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape

        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]

        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = outputs[0]

        #### break the large judgements into sentences chunk of given size. Do this while inference
        # chunk_size = 1024
        # input_ids = batch["input_ids"].view(-1, tokens)
        # chunk_cnt = int(math.ceil(input_ids.shape[0]/chunk_size))
        # input_ids_chunk_list = torch.chunk(input_ids,chunk_cnt)
        #
        # attention_mask_chunk_list = torch.chunk(attention_mask,chunk_cnt)
        # outputs = []
        # for input_ids,attention_mask in zip(input_ids_chunk_list,attention_mask_chunk_list):
        #     with torch.no_grad():
        #         output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #         output = output[0]
        #         #output = output[0].to('cpu')
        #     outputs.append(copy.deepcopy(output))
        #     torch.cuda.empty_cache()
        #
        # bert_embeddings = torch.cat(tuple(outputs))  #.to('cuda')

        if not self.bert_trainable and batch["task"] in self.cacheable_tasks:
            # cache the embeddings of BERT if it is not fine-tuned
            # to save GPU memory put the values on CPU
            batch["bert_embeddings"] = bert_embeddings.to("cpu")

        return bert_embeddings

class BertHSLN(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''
    def __init__(self, config, tasks):
        super(BertHSLN, self).__init__()
        self.num_lsp_label = 4
        self.config = config
        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.generic_output_layer = config.get("generic_output_layer")
#         print('generic_output_layer: ', self.generic_output_layer)
        self.lstm_hidden_size = config["word_lstm_hs"]
        self.n_WWlayer = config["n_WWlayer"]
        self.n_WSlayer = config["n_WSlayer"]
        self.n_SSlayer = config["n_SSlayer"]
        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True, dropout=self.config['dropout']))

        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])

        self.init_sentence_enriching(config, tasks)

        self.reinit_output_layer(tasks, config)

#        self.sent2sent = WSWGAT(in_dim=self.lstm_hidden_size*2,
#                                out_dim=self.lstm_hidden_size*2,
#                                num_heads=4,
#                                attn_drop_out=0.1,
#                                ffn_inner_hidden_size=512,
#                                ffn_drop_out=0.1,
#                                feat_embed_size=50,
#                                layerType="S2S"
#                                )
        self.word2sent = WSWGAT(in_dim=self.lstm_hidden_size*2,
                                out_dim=self.lstm_hidden_size*2,
                                num_heads=2,
                                attn_drop_out=0.1,
                                ffn_inner_hidden_size=512,
                                ffn_drop_out=0.1,
                                feat_embed_size=50,
                                layerType="W2S"
                                )
        self.word2word = WSWGAT(in_dim=self.lstm_hidden_size*2,
                                out_dim=self.lstm_hidden_size*2,
                                num_heads=4,
                                attn_drop_out=0.1,
                                ffn_inner_hidden_size=512,
                                ffn_drop_out=0.1,
                                feat_embed_size=50,
                                layerType="W2W"
                                )
#        self.load_glove_emb()
#        self.word_emb_map = torch.nn.Linear(self.config['glove_emb_size'], self.lstm_hidden_size*2)
        self.lsp_model = LSTM_Emitter_Binary(self.num_lsp_label, 3*768, self.lstm_hidden_size*2, 0.5)
        self.supcon_loss = SupConLoss(config["temperature"], contrast_mode='all')
        self.supcon_loss_one = SupConLoss(config["temperature"], contrast_mode='one')
        
    def load_glove_emb(self):
        path = f"glove/glove_{self.config['glove_emb_size']}.pkl"
        glove = pkl.load(open(path, 'rb'))
        word_embedding = torch.from_numpy(glove['embedding']).to(dtype=torch.float32)
        self.word_embedding = torch.nn.parameter.Parameter(data=word_embedding, requires_grad=False)
        
        
    def init_sentence_enriching(self, config, tasks):
        input_dim = self.attention_pooling.output_dim
        print(f"Attention pooling dim: {input_dim}")
        self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True, dropout=0.4))

    def reinit_output_layer(self, tasks, config):
        if config.get("without_context_enriching_transfer"):
            self.init_sentence_enriching(config, tasks)
        input_dim = self.lstm_hidden_size * 2

        if self.generic_output_layer:
            self.crf = CRFOutputLayer(in_dim=input_dim*3, num_labels=len(tasks[0].labels))
            
        else:
            self.crf = CRFPerTaskOutputLayer(input_dim*3, tasks)
        self.lsp_crf = CRFOutputLayer(in_dim=input_dim, num_labels=self.num_lsp_label)
        self.contrastive_mapping = torch.nn.Linear(input_dim*3, len(tasks[0].labels))

    def get_word_embedding(self, subword_embeddings, span):
        word_embedding = []
        for i, sent in enumerate(subword_embeddings):
            for j, span_list in enumerate(span[i]):
                pooled_embedding = []
                for sp in span_list:
#                     print("torch.mean(sent[sp[0]:sp[1]], dim=0): ", torch.mean(sent[sp[0]:sp[1]], dim=0).shape)
                    pooled_embedding.append(torch.mean(sent[sp[0]:sp[1]], dim=0).unsqueeze(0))
                if len(pooled_embedding) == 1:
                    pooled_embedding = pooled_embedding[0]
#                     print('pooled_embedding_1: ', pooled_embedding.shape)
                else:
                    pooled_embedding = torch.mean(torch.cat(pooled_embedding, dim=0), dim=0).unsqueeze(0)
#                     print('pooled_embedding_2: ', pooled_embedding.shape)
                word_embedding.append(pooled_embedding)
        if len(word_embedding) == 1:
            word_embedding = word_embedding[0]
#         print('word_embedding: ', len(word_embedding))
#         print('word_embedding: ', word_embedding[0].shape)
        else:
#             print('word_embedding: ', word_embedding[0].shape)
            word_embedding = torch.cat(word_embedding, dim=0)
        return word_embedding
    
    def forward(self, batch, labels=None, output_all_tasks=False):
        graph = copy.deepcopy(batch['graph'][0])  
        documents, sentences, tokens = batch["input_ids"].shape
        device = batch["input_ids"].get_device()
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)
        
        #get sentence embedding
#         print('bert_embeddings_encoded: ', bert_embeddings_encoded.shape)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
#         print('sentence_embeddings1: ', sentence_embeddings.shape)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
#         print('sentence_embeddings1: ', sentence_embeddings.shape)
        sentence_embeddings = self.dropout(sentence_embeddings)
        sentence_mask = batch["sentence_mask"]
        sentence_embeddings_encoded = self.sentence_lstm(sentence_embeddings, sentence_mask)
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)
        
        #get word embedding
        word_embedding = self.get_word_embedding(bert_embeddings_encoded, batch['cword_spans'][0])

        old_sentence_embeddings_encoded = sentence_embeddings_encoded
        d, s, h = sentence_embeddings_encoded.shape
        sentence_embeddings_encoded = sentence_embeddings_encoded.view(d*s, h)
#         print('word_embedding: ', word_embedding.shape)
        word_embedding = word_embedding.squeeze()
        for i in range(self.n_WWlayer):
            word_embedding = self.word2word(graph, word_embedding, word_embedding)
        for i in range(self.n_WSlayer):
            sentence_embeddings_encoded = self.word2sent(graph, word_embedding, sentence_embeddings_encoded)
#        for i in range(self.n_SSlayer):
#            sentence_embeddings_encoded = self.sent2sent(graph, sentence_embeddings_encoded, sentence_embeddings_encoded)
        sentence_embeddings_encoded = sentence_embeddings_encoded.view(d, s, h)
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)

        lsp_sent_embedding = batch['lsp_embedding']
        lsp_emission, lsp_hidden = self.lsp_model(lsp_sent_embedding)
        sentence_embeddings_encoded = torch.cat([sentence_embeddings_encoded, lsp_hidden, old_sentence_embeddings_encoded], axis=-1)
        
        if self.generic_output_layer:
            output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
        else:
            output = self.crf(batch["task"], sentence_embeddings_encoded, sentence_mask, labels, output_all_tasks)
           
        lsp_mask = torch.ones(lsp_sent_embedding.shape[0], lsp_sent_embedding.shape[1]).to(lsp_sent_embedding.device)

        lsp_output = self.lsp_crf(lsp_hidden, lsp_mask, batch['lsp_label'])
        
        sentence_embeddings_encoded = self.contrastive_mapping(sentence_embeddings_encoded)
        #sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)
        # constrastive learning
        constrastive_output = {}
        if labels is not None:
            # create batch
            constrastive_losses = []
            clusters = batch['clusters'][0]
            cluster_labels = list(clusters.keys())
            for _ in range(self.config["constrastive_batch_size"]):
                # sents : (batch, 1, d)
                # sents = []
                # sent_labels = []
                # for c in cluster_labels:
                #     if len(clusters[c]) > 0:
                #         if len(clusters[c]) < self.config["constrastive_example_per_label"]:
                #             s = self.config['constrastive_example_per_label'] // len(clusters[c])
                #             r = self.config['constrastive_example_per_label'] % len(clusters[c])
                #             for _ in range(s):
                #                 for idx in clusters[c]:
                #                     sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
                #                     sent_labels.append(c)
                #             for _ in range(r):
                #                 idx = random.choice(clusters[c])
                #                 sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
                #                 sent_labels.append(c)
                #         else:
                #             idxes = random.sample(clusters[c], self.config["constrastive_example_per_label"])
                #             for idx in idxes:
                #                 sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
                #                 sent_labels.append(c)
                sents = []
                sent_labels = []
                for _ in range(16):
                    c = random.choice(cluster_labels)
                    if len(clusters[c]) < self.config["constrastive_example_per_label"]:
                        s = self.config['constrastive_example_per_label'] // len(clusters[c])
                        r = self.config['constrastive_example_per_label'] % len(clusters[c])
                        for _ in range(s):
                            for idx in clusters[c]:
                                sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
                                sent_labels.append(c)
                        for _ in range(r):
                            idx = random.choice(clusters[c])
                            sents.append(sentence_embeddings_encoded[0][c].unsqueeze(0))
                            sent_labels.append(c)
                    else:
                        idxes = random.sample(clusters[c], self.config["constrastive_example_per_label"])
                        for idx in idxes:
                            sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
                            sent_labels.append(c)
                            
                # feature : (batch, 1, d) = (3, 1, d)
                feature = torch.cat(sents, dim=0).unsqueeze(1)
                feature = torch.nn.functional.normalize(feature, p=2, dim=2)
                # cosntrastive_label : (3)
                constrastive_label = torch.tensor(sent_labels, device=feature.device)
                loss = self.supcon_loss(feature, constrastive_label)
                # print("feature", feature.shape, "label", constrastive_label.shape, "constrastive loss", loss.item())
                constrastive_losses.append(loss)

            # # constraste analysis
            # for _ in range(self.config["constrastive_batch_size"] // 2):
            #     # sents : (batch, 1, d)
            #     analysis_id = GEN_LABELS.index('ANALYSIS')
            #     for c in cluster_labels:
            #         sents = []
            #         sent_labels = []
            #         if c != analysis_id and len(clusters[c]) > 0 and len(clusters[analysis_id]) > 0:
            #             if len(clusters[c]) < self.config["constrastive_example_per_label"]:
            #                 s = self.config['constrastive_example_per_label'] // len(clusters[c])
            #                 r = self.config['constrastive_example_per_label'] % len(clusters[c])
            #                 for _ in range(s):
            #                     for idx in clusters[c]:
            #                         sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                         sent_labels.append(c)
            #                 for _ in range(r):
            #                     idx = random.choice(clusters[c])
            #                     sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                     sent_labels.append(c)
            #             else:
            #                 idxes = random.sample(clusters[c], self.config["constrastive_example_per_label"])
            #                 for idx in idxes:
            #                     sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                     sent_labels.append(c)
                        
            #             n_analysis = 5 * self.config["constrastive_example_per_label"]
            #             if len(clusters[analysis_id]) < n_analysis:
            #                 s = n_analysis // len(clusters[analysis_id])
            #                 r = n_analysis % len(clusters[analysis_id])
            #                 for _ in range(s):
            #                     for idx in clusters[analysis_id]:
            #                         sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                         sent_labels.append(analysis_id)
            #                 for _ in range(r):
            #                     idx = random.choice(clusters[analysis_id])
            #                     sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                     sent_labels.append(analysis_id)
            #             else:
            #                 idxes = random.sample(clusters[analysis_id], n_analysis)
            #                 for idx in idxes:
            #                     sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                     sent_labels.append(analysis_id)

            #             # feature : (batch, 1, d) = (3, 1, d)
            #             feature = torch.cat(sents, dim=0).unsqueeze(1)
            #             feature = torch.nn.functional.normalize(feature, p=2, dim=2)
            #             # cosntrastive_label : (3)
            #             constrastive_label = torch.tensor(sent_labels, device=feature.device)
            #             loss = self.supcon_loss_one(feature, constrastive_label)
            #             # print("feature", feature.shape, "label", constrastive_label.shape, "constrastive loss", loss.item())
            #             constrastive_losses.append(loss)
            
            # # constraste ARG PETITIONER  ARG RESPONDENT
            # for _ in range(self.config["constrastive_batch_size"] // 2):
            #     # sents : (batch, 1, d)
            #     pet_id = GEN_LABELS.index('ARG_PETITIONER')
            #     res_id = GEN_LABELS.index('ARG_RESPONDENT')
            #     if pet_id in cluster_labels and len(clusters[pet_id]) > 0 and res_id in cluster_labels and len(clusters[res_id]) > 0:
            #         sents = []
            #         sent_labels = []
            #         for c in [pet_id, res_id]:
            #             if len(clusters[c]) < self.config["constrastive_example_per_label"]:
            #                 s = self.config['constrastive_example_per_label'] // len(clusters[c])
            #                 r = self.config['constrastive_example_per_label'] % len(clusters[c])
            #                 for _ in range(s):
            #                     for idx in clusters[c]:
            #                         sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                         sent_labels.append(c)
            #                 for _ in range(r):
            #                     idx = random.choice(clusters[c])
            #                     sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                     sent_labels.append(c)
            #             else:
            #                 idxes = random.sample(clusters[c], self.config["constrastive_example_per_label"])
            #                 for idx in idxes:
            #                     sents.append(sentence_embeddings_encoded[0][idx].unsqueeze(0))
            #                     sent_labels.append(c)
            #         # feature : (batch, 1, d) = (3, 1, d)
            #         feature = torch.cat(sents, dim=0).unsqueeze(1)
            #         feature = torch.nn.functional.normalize(feature, p=2, dim=2)
            #         # cosntrastive_label : (3)
            #         constrastive_label = torch.tensor(sent_labels, device=feature.device)
            #         loss = self.supcon_loss(feature, constrastive_label)
            #         # print("feature", feature.shape, "label", constrastive_label.shape, "constrastive loss", loss.item())
            #         constrastive_losses.append(loss)
                                                        
            constrastive_loss = torch.mean(torch.tensor(constrastive_losses, device=output['loss'].device))
            constrastive_output['loss'] = constrastive_loss
            
        if labels is not None:
            return output, lsp_output, constrastive_output
        return output, lsp_output


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class BertHSLNMultiSeparateLayers(torch.nn.Module):
    '''
    Model Multi-Task Learning, where only certail layers are shared.
    This class is necessary to separate the model on two GPUs.
    '''
    def __init__(self, config, tasks):
        super(BertHSLNMultiSeparateLayers, self).__init__()


        self.bert = BertTokenEmbedder(config)


        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                                             hidden_size=self.lstm_hidden_size,
                                                             num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = PerTaskGroupWrapper(
                                        task_groups=config["attention_groups"],
                                        create_module_func=lambda g:
                                            AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])
                                )

        attention_pooling_output_dim = next(iter(self.attention_pooling.per_task_mod.values())).output_dim
        self.sentence_lstm = PerTaskGroupWrapper(
                                    task_groups=config["context_enriching_groups"],
                                    create_module_func=lambda g:
                                    PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=attention_pooling_output_dim,
                                        hidden_size=self.lstm_hidden_size,
                                        num_layers=1, batch_first=True, bidirectional=True))
                                    )

        self.crf = CRFPerTaskGroupOutputLayer(self.lstm_hidden_size * 2, tasks, config["output_groups"])



    def to_device(self, device1, device2):
        self.bert.to(device1)
        self.word_lstm.to(device1)
        self.attention_pooling.to_device(device1, device2)
        self.sentence_lstm.to_device(device1, device2)
        self.crf.to_device(device1, device2)
        self.device1 = device1
        self.device2 = device2

    def forward(self, batch, labels=None, output_all_tasks=False):
        task_name = batch["task"]
        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)

        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        device = self.attention_pooling.get_device(task_name)
        sentence_embeddings = self.attention_pooling(task_name, bert_embeddings_encoded.to(device), tokens_mask.to(device))
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)

        sentence_mask = batch["sentence_mask"]
        # shape: (documents, sentence, 2*lstm_hidden_size)
        device = self.sentence_lstm.get_device(task_name)
        sentence_embeddings_encoded = self.sentence_lstm(task_name, sentence_embeddings.to(device), sentence_mask.to(device))
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)

        device = self.crf.get_device(task_name)
        if labels is not None:
            labels = labels.to(device)

        output = self.crf(task_name, sentence_embeddings_encoded.to(device), sentence_mask.to(device), labels, output_all_tasks)

        return output

class CRFPerTaskGroupOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, tasks, task_groups):
        super(CRFPerTaskGroupOutputLayer, self).__init__()

        def get_task(name):
            for t in tasks:
                if t.task_name == name:
                    return t

        self.crf = PerTaskGroupWrapper(
                                        task_groups=task_groups,
                                        create_module_func=lambda g:
                                            # we assume same labels per group
                                            CRFOutputLayer(in_dim=in_dim, num_labels=len(get_task(g[0]).labels))
        )
        self.all_tasks = [t for t in [g for g in task_groups]]


    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.crf(task, x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for task in self.self.all_tasks:
                task_result = self.crf(task, x, mask, labels)
                task_result["task"] = task
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.crf.to_device(device1, device2)

    def get_device(self, task):
        return self.crf.get_device(task)


class PerTaskGroupWrapper(torch.nn.Module):
    def __init__(self, task_groups, create_module_func):
        super(PerTaskGroupWrapper, self).__init__()

        self.per_task_mod = torch.nn.ModuleDict()
        for g in task_groups:
            mod = create_module_func(g)
            for t in g:
                self.per_task_mod[t] = mod

        self.task_groups = task_groups

    def forward(self, task_name, *args):
        mod = self.per_task_mod[task_name]
        return mod(*args)

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, tasks in enumerate(self.task_groups):
            for task in tasks:
                if index % 2 == 0:
                    self.task_to_device[task] = device1
                    self.per_task_mod[task].to(device1)
                else:
                    self.task_to_device[task] = device2
                    self.per_task_mod[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]


