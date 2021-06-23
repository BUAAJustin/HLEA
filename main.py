# -*- coding: utf-8 -*-


import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import os
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.hlea import HLEA as SeqModel
from utils.data import Data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def data_initialization(data, word_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_word_file(word_file)
    data.build_word_alphabet(train_file)
    data.build_word_alphabet(dev_file)
    data.build_word_alphabet(test_file)
    data.fix_alphabet()
    return data


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(int(pred_tag[idx][idy])) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print( " Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def set_seed(seed_num=994):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)


def batchify_with_label(input_batch_list, gpu):
    batch_size = len(input_batch_list)
    chars = [sent[0] for sent in input_batch_list]
    poss = [sent[1] for sent in input_batch_list]
    fre_words = [sent[2] for sent in input_batch_list]
    seg_words = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    fre_ptns = [sent[5] for sent in input_batch_list]
    seg_ptns = [sent[6] for sent in input_batch_list]
    aux_words = [sent[7] for sent in input_batch_list]
    aux_word_masks = [sent[8] for sent in input_batch_list]
    sing_char_words = [sent[9] for sent in input_batch_list]

    char_seq_lengths = torch.LongTensor(list(map(len, chars)))
    max_seq_len = char_seq_lengths.max()
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    pos_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    fre_word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    seg_word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    sing_char_word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    fre_ptn_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    seg_ptn_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()

    aw_num = [len(aux_words[i][0]) for i in range(batch_size)]
    max_aw_num = max(aw_num)
    aux_word_tensor = torch.zeros(batch_size, max_seq_len, max_aw_num).long()
    aw_mask_tensor = torch.ones(batch_size, max_seq_len, max_aw_num).byte()

    for b, (char, pos, label, seqlen, fre_word, seg_word, fre_ptn, seg_ptn, aux_word, aux_word_mask, aw_num, sing_char_word) in \
            enumerate(zip(chars, poss, labels, char_seq_lengths, fre_words, seg_words, fre_ptns, seg_ptns, aux_words,aux_word_masks, aw_num, sing_char_words)):
        char_seq_tensor[b, :seqlen] = torch.LongTensor(char)
        pos_seq_tensor[b, :seqlen] = torch.LongTensor(pos)
        label_seq_tensor[b, :seqlen] = torch.LongTensor(label)
        fre_word_seq_tensor[b, :seqlen] = torch.LongTensor(fre_word)
        seg_word_seq_tensor[b, :seqlen] = torch.LongTensor(seg_word)
        fre_ptn_seq_tensor[b, :seqlen] = torch.LongTensor(fre_ptn)
        seg_ptn_seq_tensor[b, :seqlen] = torch.LongTensor(seg_ptn)
        mask[b, :seqlen] = torch.Tensor([1] * int(seqlen))
        sing_char_word_seq_tensor[b, :seqlen] = torch.LongTensor(sing_char_word)
        aux_word_tensor[b, :seqlen, :aw_num] = torch.LongTensor(aux_word)
        aw_mask_tensor[b, :seqlen, :aw_num] = torch.ByteTensor(aux_word_mask)

    if gpu:
        char_seq_tensor = char_seq_tensor.cuda()
        pos_seq_tensor = pos_seq_tensor.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        fre_word_seq_tensor = fre_word_seq_tensor.cuda()
        seg_word_seq_tensor = seg_word_seq_tensor.cuda()
        fre_ptn_seq_tensor = fre_ptn_seq_tensor.cuda()
        seg_ptn_seq_tensor = seg_ptn_seq_tensor.cuda()
        aux_word_tensor = aux_word_tensor.cuda()
        aw_mask_tensor = aw_mask_tensor.cuda()
        sing_char_seq_tensor = sing_char_word_seq_tensor.cuda()
        mask = mask.cuda()

    return char_seq_tensor, pos_seq_tensor, label_seq_tensor, fre_word_seq_tensor, seg_word_seq_tensor, fre_ptn_seq_tensor,\
           seg_ptn_seq_tensor, aux_word_tensor, aw_mask_tensor, mask, sing_char_word_seq_tensor


def load_model_decode(model_dir, data, name, gpu):
    data.HP_gpu = gpu
    print( "Load Model from file: ", model_dir)
    model = SeqModel(data)

    model.load_state_dict(torch.load(model_dir))

    print(("Decode %s data ..."%(name)))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f)))

    return pred_results

def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    else:
        print( "Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        with torch.no_grad():
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end =  train_num
            instance = instances[start:end]
            if not instance:
                continue

            b_char, b_pos, b_label, b_fre_word,b_seg_word, b_fre_ptn, b_seg_ptn, b_aux_word, b_aw_mask, mask, b_single_char_word = batchify_with_label(instance, data.HP_gpu)
            score, tag_seq = model(b_char, b_pos, b_fre_word, b_seg_word, b_fre_ptn, b_seg_ptn, b_aux_word, b_aw_mask, mask, b_single_char_word)
            pred_label, gold_label = recover_label(tag_seq, b_label, mask, data.label_alphabet)
            pred_results += pred_label
            gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results

def train(data, save_model_dir):
    model = SeqModel(data)
    print( "finish building model.")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters, lr=data.HP_lr)

    best_dev = -1
    best_test = -1

    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print(("Epoch: %s/%s" %(idx,data.HP_iteration)))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1

        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            words = data.train_texts[start:end]
            if not instance:
                continue

            b_char, b_pos, b_label, b_fre_word,b_seg_word, b_fre_ptn, b_seg_ptn, b_aux_word, b_aw_mask, mask, b_single_char_word = batchify_with_label(instance, data.HP_gpu)
            instance_count += 1
            loss = model.neg_log_likelihood_loss(b_char, b_pos, b_fre_word, b_seg_word, b_fre_ptn, b_seg_ptn, b_label, b_aux_word, b_aw_mask, mask, b_single_char_word)
            sample_loss += loss.data
            total_loss += loss.data
            batch_loss += loss

            if end%1000 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f "%(end, temp_cost, sample_loss))
                sys.stdout.flush()
                sample_loss = 0
            if end%data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f "%(end, temp_cost, sample_loss))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss)))

        speed, acc, p, r, f, pred_labels = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        current_score = f
        print(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f)))

        if current_score > best_dev:
            print( "Exceed previous dev f score:", best_dev)
            model_name = save_model_dir
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
            best_dev_p = p
            best_dev_r = r
            print("Best dev score: p:{}, r:{}, f:{}".format(best_dev_p, best_dev_r, best_dev))

        speed_t, acc_t, p_t, r_t, f_t, _ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        current_test_score = f_t
        print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed_t, acc_t, p_t, r_t, f_t)))
        if current_test_score > best_test:
            print("Exceed previous test f score:", best_test)
            best_test = current_test_score
            best_test_p = p_t
            best_test_r = r_t
            print("Best test score: p:{}, r:{}, f:{}".format(best_test_p, best_test_r, best_test))

        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', choices=['train', 'test'], help='update algorithm', default='train')
    parser.add_argument('--model_type', choices=['segmenter', 'frequency', 'merged'], default="merged")
    parser.add_argument('--modelpath', default="./save_model/")
    parser.add_argument('--modelname', default="weibo_model")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="./data/weibo.dset")
    parser.add_argument('--train', default="./data/weibo/train.bmeso")
    parser.add_argument('--dev', default="./data/weibo/dev.bmeso" )
    parser.add_argument('--test', default="./data/weibo/test.bmeso")
    parser.add_argument('--char_emb_path', default="./data/gigaword_chn.all.a2b.uni.ite50.vec")
    parser.add_argument('--word_emb_path', default="./data/ctb.50d.vec")
    parser.add_argument('--seed',default=2984,type=int)  # 2984  0.005  5  100 0.5
    parser.add_argument('--num_iter',default=100,type=int)
    parser.add_argument('--encoder_layer_num', default=1, type=int)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--drop', type=float, default=0.5)

    args = parser.parse_args()
    seed_num = args.seed
    set_seed(seed_num)
    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    status = args.status.lower()

    save_model_dir = args.modelpath+args.modelname
    save_data_name = args.savedset
    gpu = torch.cuda.is_available()

    char_file = args.char_emb_path
    word_file = args.word_emb_path

    sys.stdout.flush()

    if status == 'train':
        if os.path.exists(save_data_name):
            print('Loading processed data')
            with open(save_data_name, 'rb') as fp:
                data = pickle.load(fp)
            data.HP_bilstm = True
            data.model_type = args.model_type
            data.HP_batch_size = args.batch_size
            data.HP_iteration = args.num_iter
            data.HP_lr = args.lr
            data.HP_hidden_dim = args.hidden_dim
            data.HP_dropout = args.drop
            data.HP_gpu = gpu
            data.HP_lstm_layer = args.encoder_layer_num
            data.seed = args.seed

        else:
            data = Data()
            data.HP_gpu = gpu
            data.model_type = args.model_type
            data.HP_batch_size = args.batch_size
            data.HP_iteration = args.num_iter
            data.HP_dropout = args.drop
            data.HP_lr = args.lr
            data.HP_hidden_dim = args.hidden_dim
            data.HP_lstm_layer = args.encoder_layer_num
            data.seed = args.seed

            data_initialization(data, word_file, train_file, dev_file, test_file)
            data.generate_instance_with_word(train_file,'train')
            data.generate_instance_with_word(dev_file,'dev')
            data.generate_instance_with_word(test_file,'test')
            data.build_char_pretrain_emb(char_file)
            data.build_word_pretrain_emb(word_file)
            print('Dumping data')
            with open(save_data_name, 'wb') as f:
                pickle.dump(data, f)
            set_seed(seed_num)
        data.show_data_summary()
        train(data, save_model_dir)
    elif status == 'test':
        print('Loading processed data')
        with open(save_data_name, 'rb') as fp:
            data = pickle.load(fp)
        data.HP_iteration = args.num_iter
        data.HP_lr = args.lr
        data.HP_hidden_dim = args.hidden_dim
        data.generate_instance_with_word(test_file,'test')
        data.show_data_summary()
        load_model_decode(save_model_dir, data, 'test', gpu)

    else:
        print( "Invalid argument! Please use valid arguments! (train/test/decode)")


