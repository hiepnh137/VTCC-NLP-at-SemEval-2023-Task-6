## Imports

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import json
import random
import numpy as np
import string
from collections import defaultdict
import os

'''
Shift Module:
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings. 
    The feature vectors are actually context-aware sentence embeddings. 
    These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''
class LSTM_Emitter_Binary(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop = 0.5, device = 'cuda'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.hidden = None
        self.device = device
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))
    
    def forward(self, sequences):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]
        
        # initialize hidden state
        self.hidden = self.init_hidden(sequences.shape[0])
        
        # generate context-aware sentence embeddings (feature vectors)
        ## tensor[batch_size, max_seq_len, emb_dim] --> tensor[batch_size, max_seq_len, hidden_dim]
        x, self.hidden = self.lstm(sequences, self.hidden)
        x_new = self.dropout(x)
        
        # generate emission scores for each class at each sentence
        # tensor[batch_size, max_seq_len, hidden_dim] --> tensor[batch_size, max_seq_len, n_tags]
        x_new = self.hidden2tag(x_new)
        return x_new, x
    
'''
RR Module:
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings. 
    The feature vectors are actually context-aware sentence embeddings. 
    These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''
class LSTM_Emitter(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop = 0.5, device = 'cuda'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(2*hidden_dim, n_tags)
        self.hidden = None
        self.device = device
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))
    
    def forward(self, sequences, hidden_binary):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]
        
        # initialize hidden state
        self.hidden = self.init_hidden(sequences.shape[0])
        
        # generate context-aware sentence embeddings (feature vectors)
        ## tensor[batch_size, max_seq_len, emb_dim] --> tensor[batch_size, max_seq_len, hidden_dim]
        x, self.hidden = self.lstm(sequences, self.hidden)
        final = torch.zeros((x.shape[0], x.shape[1], 2*x.shape[2])).to(self.device)
        ## Concat the hidden states of both Shift and RR Module LSTM's and then pass through a linear layer to get emission scores for RR Module
        for batch_name, doc in enumerate(x):
            for i, sent in enumerate(doc):
                final[batch_name][i] = torch.cat((x[batch_name][i], hidden_binary[batch_name][i]),0)
        final = self.dropout(final)
        
        # generate emission scores for each class at each sentence
        # tensor[batch_size, max_seq_len, hidden_dim] --> tensor[batch_size, max_seq_len, n_tags]
        final = self.hidden2tag(final)
        return final

'''
    A linear-chain CRF is fed with the emission scores at each sentence, 
    and it finds out the optimal sequence of tags by learning the transition scores.
'''
class CRF(nn.Module):    
    def __init__(self, n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx = None):
        super().__init__()
        
        self.n_tags = n_tags
        self.SOS_TAG_IDX = sos_tag_idx
        self.EOS_TAG_IDX = eos_tag_idx
        self.PAD_TAG_IDX = pad_tag_idx
        
        self.transitions = nn.Parameter(torch.empty(self.n_tags, self.n_tags))
        self.init_weights()
        
    def init_weights(self):
        # initialize transitions from random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        
        # enforce constraints (rows = from, cols = to) with a big negative number.
        # exp(-1000000) ~ 0
        
        # no transitions to SOS
        self.transitions.data[:, self.SOS_TAG_IDX] = -1000000.0
        # no transition from EOS
        self.transitions.data[self.EOS_TAG_IDX, :] = -1000000.0
        
        if self.PAD_TAG_IDX is not None:
            # no transitions from pad except to pad
            self.transitions.data[self.PAD_TAG_IDX, :] = -1000000.0
            self.transitions.data[:, self.PAD_TAG_IDX] = -1000000.0
            # transitions allowed from end and pad to pad
            self.transitions.data[self.PAD_TAG_IDX, self.EOS_TAG_IDX] = 0.0
            self.transitions.data[self.PAD_TAG_IDX, self.PAD_TAG_IDX] = 0.0
            
    def forward(self, emissions, tags, mask = None):
        ## emissions: tensor[batch_size, seq_len, n_tags]
        ## tags: tensor[batch_size, seq_len]
        ## mask: tensor[batch_size, seq_len], indicates valid positions (0 for pad)
        return -self.log_likelihood(emissions, tags, mask = mask)
    
    def log_likelihood(self, emissions, tags, mask = None):                   
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
            
        scores = self._compute_scores(emissions, tags, mask = mask)
        partition = self._compute_log_partition(emissions, mask = mask)
        return torch.sum(scores - partition)
    
    # find out the optimal tag sequence using Viterbi Decoding Algorithm
    def decode(self, emissions, mask = None):      
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
            
        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences
    
    def _compute_scores(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        if(torch.cuda.is_available()):
            scores = torch.zeros(batch_size).cuda()
        else:
            scores = torch.zeros(batch_size)
        
        # save first and last tags for later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
        
        # add transition from SOS to first tags for each sample in batch
        t_scores = self.transitions[self.SOS_TAG_IDX, first_tags]
        
        # add emission scores of the first tag for each sample in batch
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores
        
        # repeat for every remaining word
        for i in range(1, seq_len):
            
            is_valid = mask[:, i]
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]
            
            e_scores = emissions[:, i].gather(1, curr_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[prev_tags, curr_tags]
                        
            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid
            
            scores += e_scores + t_scores
            
        # add transition from last tag to EOS for each sample in batch
        scores += self.transitions[last_tags, self.EOS_TAG_IDX]
        return scores
    
    # compute the partition function in log-space using forward algorithm
    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape
        
        # in the first step, SOS has all the scores
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        
        for i in range(1, seq_len):
            ## tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1) 
            
            ## tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)
            
            ## tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)
            
            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim = 1)
            
            # set alphas if the mask is valid, else keep current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas
            
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)
        
        # return log_sum_exp
        return torch.logsumexp(end_scores, dim = 1)
    
    # return a list of optimal tag sequence for each example in the batch
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape
        
        # in the first iteration, SOS will have all the scores and then, the max
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        
        backpointers = []
        
        for i in range(1, seq_len):
            ## tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1) 
            
            ## tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)
            
            ## tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)
            
            scores = e_scores + t_scores + a_scores
            
            # find the highest score and tag, instead of log_sum_exp
            max_scores, max_score_tags = torch.max(scores, dim = 1)
            
            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas
            
            backpointers.append(max_score_tags.t())
            
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences
    
    # auxiliary function to find the best path sequence for a specific example
    def _find_best_path(self, sample_id, best_tag, backpointers):
        ## backpointers: list[tensor[seq_len_i - 1, n_tags, batch_size]], seq_len_i is the length of the i-th sample of the batch
        
        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path

'''
    MTL Model to classify. Our Architecture which used the RR component and 
    Shift component parallely to get the emission scores and then they are 
    fed into the CRF to get the appropriate probabilities for each label.
'''
class MTL_Classifier(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size = 0, pad_word_idx = 0, pretrained = False, device = 'cuda'):
        super().__init__()
        
        self.emb_dim = sent_emb_dim
        self.pretrained = pretrained
        self.device = device
        self.pad_tag_idx = pad_tag_idx
        self.pad_word_idx = pad_word_idx
        
        ## RR Modele    
        self.emitter = LSTM_Emitter(n_tags, sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)
        
        ## Shift or Binary Module
        self.emitter_binary = LSTM_Emitter_Binary(5, 3*sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
        self.crf_binary = CRF(5, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)
        
    
    def forward(self, x, x_binary):
        batch_size = len(x)
        seq_lengths = [len(doc) for doc in x]
        max_seq_len = max(seq_lengths)
        
            
        ## x: list[batch_size, sents_per_doc, sent_emb_dim]
        tensor_x = [torch.tensor(doc, dtype = torch.float, requires_grad = True) for doc in x]
        tensor_x_binary = [torch.tensor(doc, dtype = torch.float, requires_grad = True) for doc in x_binary]
        
        ## list[batch_size, sents_per_doc, sent_emb_dim] --> tensor[batch_size, max_seq_len, sent_emb_dim]
        tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first = True).to(self.device)    
        tensor_x_binary = nn.utils.rnn.pad_sequence(tensor_x_binary, batch_first = True).to(self.device)  
        
        self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
        for i, sl in enumerate(seq_lengths):
            self.mask[i, :sl] = 1
        
        ## Get hidden states of Shift Module and pass them to the RR Module for emission score calculation for RR Module
        self.emissions_binary, self.hidden_binary = self.emitter_binary(tensor_x_binary)
        self.emissions = self.emitter(tensor_x, self.hidden_binary)
        
        ## Passing the emission scores to the CRF to get the final sequence of tags
        _, path = self.crf.decode(self.emissions, mask = self.mask)
        _, path_binary = self.crf_binary.decode(self.emissions_binary, mask = self.mask)
        return path, path_binary
    
    def _loss(self, y):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y = [torch.tensor(doc, dtype = torch.long) for doc in y]
        tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first = True, padding_value = self.pad_tag_idx).to(self.device)
        
        nll = self.crf(self.emissions, tensor_y, mask = self.mask)
        return nll    
    
    def _loss_binary(self, y_binary):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y_binary = [torch.tensor(doc, dtype = torch.long) for doc in y_binary]
        tensor_y_binary = nn.utils.rnn.pad_sequence(tensor_y_binary, batch_first = True, padding_value = self.pad_tag_idx).to(self.device)
        
        nll_binary = self.crf_binary(self.emissions_binary, tensor_y_binary, mask = self.mask)
        return nll_binary    