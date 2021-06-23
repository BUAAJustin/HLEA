# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.crf import CRF
from model.layers import NERmodel

class HLEA(nn.Module):
    def __init__(self, data):
        super(HLEA, self).__init__()

        self.gpu = data.HP_gpu
        self.hidden_dim = data.HP_hidden_dim
        self.word_alphabet = data.word_alphabet
        self.word_emb_dim = data.word_emb_dim
        self.char_emb_dim = data.char_emb_dim
        self.pos_emb_dim = data.pos_emb_dim
        self.ptn_emb_dim = data.ptn_emb_dim
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.model_type = data.model_type
        scale = np.sqrt(3.0 / self.word_emb_dim)
        data.pretrain_word_embedding[0,:] = np.random.uniform(-scale, scale, [1, self.word_emb_dim])

        self.char_embedding = nn.Embedding(data.char_alphabet.size(), self.word_emb_dim)
        self.pos_embedding = nn.Embedding(data.pos_alphabet.size(), self.pos_emb_dim)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        self.ptn_embedding = nn.Embedding(5, 50)



        self.char_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
        self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        self.pos_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.pos_alphabet.size(), 50)))
        self.ptn_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(5, 50)))
        char_feature_dim = self.char_emb_dim + self.pos_emb_dim + self.word_emb_dim*3 + self.ptn_emb_dim
        if self.model_type == 'merged':
            char_feature_dim = self.char_emb_dim + self.pos_emb_dim + self.word_emb_dim * 4 + self.ptn_emb_dim*2
        ## lstm model
        lstm_hidden = self.hidden_dim
        self.NERmodel = NERmodel(model_type='lstm', input_dim=char_feature_dim, hidden_dim=lstm_hidden, num_layer=self.lstm_layer, biflag=self.bilstm_flag)

        self.drop = nn.Dropout(p=data.HP_dropout)
        self.hidden2tag = nn.Linear(2*self.hidden_dim, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)


        if self.gpu:
            print('use gpu')
            self.word_embedding = self.word_embedding.cuda()
            self.char_embedding = self.char_embedding.cuda()
            self.pos_embedding = self.pos_embedding.cuda()
            self.ptn_embedding = self.ptn_embedding.cuda()
            self.NERmodel = self.NERmodel.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb
    

    def get_tags(self, chars, poss, fre_words, seg_words, fre_ptns, seg_ptns, aux_words, aux_word_masks, sing_char_words):

        batch_size = chars.size()[0]
        seq_len = chars.size()[1]
        max_ci_num = aux_words.size(-1)

        # char embdding
        char_embs = self.char_embedding(chars)
        pos_embs = self.pos_embedding(poss)
        char_embs = torch.cat([char_embs, pos_embs], dim=-1)

        # main word embedding
        fre_word_embs = self.word_embedding(fre_words)
        seg_word_embs = self.word_embedding(seg_words)
        fre_ptn_embs = self.ptn_embedding(fre_ptns)
        seg_ptn_embs = self.ptn_embedding(seg_ptns)
        fre_main_word_embs = torch.cat([fre_word_embs, fre_ptn_embs], dim=-1)
        seg_main_word_embs = torch.cat([seg_word_embs, seg_ptn_embs], dim=-1)

        # auxiliary word embedding
        scw_embs = self.word_embedding(sing_char_words)

        aw_embs_init = self.word_embedding(aux_words)
        aw_embs_unmasked = torch.cat([aw_embs_init], dim=-1)
        aw_mask = aux_word_masks.unsqueeze(-1).repeat(1, 1, 1, self.word_emb_dim)
        aw_embs = aw_embs_unmasked.data.masked_fill_(aw_mask.bool(), 0)

        fre_word_emb_reshaped = fre_word_embs.reshape(batch_size, seq_len, 1, -1).repeat(1, 1, max_ci_num,1).contiguous()
        seg_word_emb_reshaped = seg_word_embs.reshape(batch_size, seq_len, 1, -1).repeat(1, 1, max_ci_num,1).contiguous()
        fre_weight = torch.sum(torch.mul(fre_word_emb_reshaped, aw_embs_init), -1)
        seg_weight = torch.sum(torch.mul(seg_word_emb_reshaped, aw_embs_init), -1)
        if self.model_type == 'frequency':
            weights = fre_weight
        elif self.model_type == 'segmenter':
            weights = seg_weight
        elif self.model_type == 'merged':
            weights = fre_weight + seg_weight
            weights_sum = torch.sum(weights, dim = -1, keepdim = True)
            weights = weights.div(weights_sum)

        aw_embs = torch.sum(torch.mul(aw_embs, weights.unsqueeze(-1)), dim = 2)
        aw_embs = torch.cat([aw_embs, scw_embs], dim=-1)

        if self.model_type == 'frequency':
            augmented_embs = torch.cat([char_embs, fre_main_word_embs, aw_embs],dim=-1)
        elif self.model_type == 'segmenter':
            augmented_embs = torch.cat([char_embs, seg_main_word_embs, aw_embs],dim=-1)
        elif self.model_type == 'merged':
            augmented_embs = torch.cat([char_embs, fre_main_word_embs,seg_main_word_embs, aw_embs],dim=-1)
        augmented_embs = self.drop(augmented_embs)
        feature_out = self.NERmodel(augmented_embs)
        tags = self.hidden2tag(feature_out)

        return tags



    def neg_log_likelihood_loss(self, chars, poss, fre_words, seg_words, fre_ptns, seg_ptns, batch_label, aux_words, aux_word_masks, mask, sing_char_words):

        tags = self.get_tags(chars, poss, fre_words, seg_words, fre_ptns, seg_ptns, aux_words, aux_word_masks, sing_char_words)
        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)

        return total_loss



    def forward(self, chars, poss, fre_words, seg_words, fre_ptns, seg_ptns, aux_words, aux_word_masks, mask, sing_char_words):

        tags = self.get_tags(chars, poss, fre_words, seg_words, fre_ptns, seg_ptns, aux_words, aux_word_masks, sing_char_words)
        score, tag_seq = self.crf._viterbi_decode(tags, mask)

        return score, tag_seq
