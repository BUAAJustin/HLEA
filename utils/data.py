# -*- coding: utf-8 -*-

import sys
import numpy as np
import jieba.posseg as posseg
import jieba
from utils.alphabet import Alphabet
from utils.functions import *
from utils.lexicon import Lexicon


START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"

class Data:
    def __init__(self): 
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_char_emb = True
        self.norm_word_emb = False
        self.char_alphabet = Alphabet('char')
        self.pos_alphabet = Alphabet('pos')
        self.label_alphabet = Alphabet('label', True)
        self.word_lower = False
        self.word = Lexicon(self.word_lower)
        self.word_alphabet = Alphabet('word')
        self.word_count = {}

        self.HP_fix_word_emb = False
        self.HP_use_count = False

        self.tagScheme = "BMES"
        self.model_type = 'merged'

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.train_split_index = []
        self.dev_split_index = []

        self.char_emb_dim = 50
        self.pos_emb_dim = 50
        self.ptn_emb_dim = 50
        self.word_emb_dim = 50
        self.word_dropout = 0.5
        self.pretrain_char_embedding = None
        self.pretrain_word_embedding = None
        self.label_size = 0
        self.char_alphabet_size = 0
        self.pos_alphabet_size = 0
        self.label_alphabet_size = 0
        ### hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_hidden_dim = 128
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.HP_num_layer = 4
        self.seed = 0

        
    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Pos   alphabet size: %s" % (self.pos_alphabet_size))
        print("     Word  alphabet size: %s"%(self.word_alphabet.size()))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Norm    char    emb: %s"%(self.norm_char_emb))
        print("     Norm    word    emb: %s"%(self.norm_word_emb))
        print("     Norm  word  dropout: %s"%(self.word_dropout))
        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))
        print("     Raw   instance number: %s"%(len(self.raw_texts)))
        print("     Hyperpara   iteration: %s"%(self.HP_iteration))
        print("     Hyperpara  batch size: %s"%(self.HP_batch_size))
        print("     Hyperpara          lr: %s"%(self.HP_lr))
        print("     Hyperpara    lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s"%(self.HP_clip))
        print("     Hyperpara    momentum: %s"%(self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s"%(self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s"%(self.HP_bilstm))
        print("     Hyperpara         GPU: %s"%(self.HP_gpu))
        print("     Hyperpara fixword emb: %s"%(self.HP_fix_word_emb))
        print("     Hyperpara        seed: %s"%(self.seed))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s"%(old_size, self.label_alphabet_size))


    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                char = pairs[0]
                if self.number_normalized:
                    char = normalize_word(char)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.char_alphabet.add(char)

        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def build_word_file(self, word_file):
        ## build lexicon file,initial read word embedding file
        if word_file:
            fins = open(word_file, 'r',encoding="utf-8").readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                if fin:
                    self.word.insert(fin, "one_source")
            print ("Load word file: ", word_file, " total size:", self.word.size())
        else:
            print ("Lexicon file is None, load nothing")


    def build_word_alphabet(self, input_file):
        in_lines = open(input_file,'r',encoding="utf-8").readlines()
        char_list = []
        for line in in_lines:
            if len(line) > 2:
                char = line.strip().split()[0]
                if self.number_normalized:
                    char = normalize_word(char)
                char_list.append(char)
            else:
                sentence = ''.join(char_list)
                #seg = jieba.lcut(ws)
                pos_seg = posseg.lcut(sentence, HMM=True)
                pos = [j.flag for j in pos_seg]
                for i in range(len(pos)):
                    self.pos_alphabet.add(pos[i])
                entitys = []
                for idx in range(len(char_list)):
                    matched_entity = self.word.enumerateMatchList(char_list[idx:])
                    entitys += matched_entity
                    for entity in matched_entity:
                        self.word_alphabet.add(entity)
                        index = self.word_alphabet.get_index(entity)
                        self.word_count[index] = self.word_count.get(index,0)

                # frequency count
                entitys.sort(key=lambda x:-len(x))
                while entitys:
                    longest = entitys[0]
                    longest_index = self.word_alphabet.get_index(longest)
                    self.word_count[longest_index] = self.word_count.get(longest_index, 0) + 1

                    wordlen = len(longest)
                    for i in range(wordlen):
                        for j in range(i+1,wordlen+1):
                            covering_word = longest[i:j]
                            if covering_word in entitys:
                                entitys.remove(covering_word)
                char_list = []
        self.pos_alphabet_size = self.pos_alphabet.size()
        print("word alphabet size:", self.word_alphabet.size())
        print("pos alphabet size:", self.pos_alphabet.size())

    def fix_alphabet(self):
        self.char_alphabet.close()
        self.label_alphabet.close() 
        self.word_alphabet.close()


    def build_char_pretrain_emb(self, emb_path):
        print ("build char pretrain emb...")
        self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(emb_path, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)

    def build_word_pretrain_emb(self, emb_path):
        print ("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,  self.word_emb_dim, self.norm_word_emb)


    def generate_instance_with_word(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_word(input_file, self.word, self.char_alphabet, self.pos_alphabet,  self.word_alphabet, self.word_count, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_word(input_file, self.word,self.char_alphabet, self.pos_alphabet, self.word_alphabet, self.word_count, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_word(input_file, self.word, self.char_alphabet, self.pos_alphabet, self.word_alphabet, self.word_count, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_word(input_file, self.word, self.char_alphabet, self.pos_alphabet,  self.word_alphabet, self.word_count, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file,'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
           content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, output_file))





