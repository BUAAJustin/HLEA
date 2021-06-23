# -*- coding: utf-8 -*-

import sys
import numpy as np
import re
import jieba.posseg as posseg
import jieba
from utils.alphabet import Alphabet
import copy
NULLKEY = "-null-"

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def conflict_or_not(word_test_list):
    for i in range(len(word_test_list)):
        for j in range(len(word_test_list[i])-1):
            if len(word_test_list[i][j][0])+word_test_list[i][j][1]>word_test_list[i][j+1][1]:
                return True
    return False


def correct_once(word_test):
    word_test_copy = copy.deepcopy(word_test)
    for i in range(len(word_test) - 1):
        if len(word_test[i][0]) + word_test[i][1] > word_test[i + 1][1]:
            word_test.pop(i+1)
            word_test_copy.pop(i)
            return [word_test, word_test_copy]

def calcule_all_situation(word_list):
    iter_list = [[]]
    for idx in range(len(word_list)):
        if len(word_list[idx]) == 0:
            continue
        else:
            all_situation = [copy.deepcopy(iter_list) for _ in range(len(word_list[idx]))]
            for i in range(len(all_situation)):
                for j in range(len(all_situation[i])):
                    all_situation[i][j].append(word_list[idx][i])
            iter_list = []
            for i in all_situation:
                iter_list += i
    return iter_list

def simplify(element):
    all_situation = calcule_all_situation(element)
    #all_situation_small = [element]
    while conflict_or_not(all_situation):
        for i in all_situation:
            if conflict_or_not([i]):
                all_situation.remove(i)
                all_situation += correct_once(i)
    count_max = 0
    index_max = 0
    for i in range(len(all_situation)):
        count_each = 0
        for j in range(len(all_situation[i])):
            count_each += all_situation[i][j][2]
        if count_each > count_max:
            count_max = count_each
            index_max = i
    return all_situation[index_max]

def delete_least_word(element):
    least_index = (-1,-1)
    least_count = 1000000
    for i,ele in enumerate(element):
        for j,word in enumerate(ele):
            if word[2]<least_count:
                least_index = (i,j)
                least_count = word[2]
    element[least_index[0]].pop(least_index[1])
    if element[least_index[0]] == []:
        element.pop(least_index[0])
    return element

def slice_element(element):
    element_sliced_list = []
    element_silced = []
    max_position = 0
    for i in range(len(element)):
        max_position = max(max([j[1]+len(j[0]) for j in element[i]]), max_position)
        element_silced.append(element[i])
        if i == len(element) - 1 or max_position <= element[i + 1][0][1]:
            element_sliced_list.append(element_silced)
            element_silced = []
    return element_sliced_list

def simplify_ele_plus10(element):
    assert len(element)>10
    word_corrected = []
    element_deleted = delete_least_word(element)
    element_sliced = slice_element(element_deleted)
    for i in element_sliced:
        if len(i) > 10:
            element_corrected = simplify_ele_plus10(i)
        else:
            element_corrected = simplify(i)
        word_corrected+=element_corrected
    return word_corrected

def solve_conflict(word_list):
    word_corrected = []
    element = []
    max_position = 0
    for i in range(len(word_list)):
        max_position = max(max([j[1]+len(j[0]) for j in word_list[i]]), max_position)
        element.append(word_list[i])
        if i == len(word_list) - 1 or max_position <= word_list[i + 1][0][1]:
            if len(element) > 1000:
                element_corrected = simplify_ele_plus10(element)  # 删除一个最低频率词汇 将新element切分
            elif len(element) > 0 and len(element)<=1000:
                element_corrected = simplify(element)
            else:
                element_corrected = element[0]
            word_corrected += element_corrected
            element = []
    return word_corrected

def segment_by_frequency(chars, char, word_alphabet, word_count):
    length = len(chars)
    word_list = []
    for idx in range(length):  # [中，国，语，言，学] # id = 0
        matched_list = char.enumerateMatchList(chars[idx:])
        matched_list_long = []
        for i in matched_list:
            i_count = word_count[word_alphabet.get_index(i)]
            if len(i)>1 and is_chinese(i) and \
                    not(len(i) > 2 and i[0] == i[1] and i[1] == i[2]) and i_count>0:
                matched_list_long.append((i,idx,i_count))
        if matched_list_long != []:
            word_list.append(matched_list_long)

    final_word_list_tuple = solve_conflict(word_list)
    if final_word_list_tuple == []:
        return chars
    final_word_list = []#all_possible_situation[index_max]
    if final_word_list_tuple[0][1] > 0:
        for j in range(final_word_list_tuple[0][1]):
            final_word_list.append(chars[j])
    for i in range(len(final_word_list_tuple)-1):
        for j in range(final_word_list_tuple[i][1], final_word_list_tuple[i+1][1]):
            if j - final_word_list_tuple[i][1] < len(final_word_list_tuple[i][0]):
                final_word_list.append(final_word_list_tuple[i][0])
            else:
                final_word_list.append(chars[j])
    for j in range(final_word_list_tuple[-1][1], length):
        if j - final_word_list_tuple[-1][1] < len(final_word_list_tuple[-1][0]):
            final_word_list.append(final_word_list_tuple[-1][0])
        else:
            final_word_list.append(chars[j])
    return final_word_list

def is_chinese(string):
    """
    检查整个字符串是否为中文
        string (str): 需要检查的字符串,包含空格也是False
    """
    for chart in string:
        if chart < u'\u4e00' or chart > u'\u9fff':
            return False

    return True


def read_instance_with_word(input_file, word, char_alphabet, pos_alphabet, word_alphabet, word_count, label_alphabet, number_normalized, max_sent_length):
    in_lines = open(input_file,'r',encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    chars = []
    poss = []
    fre_words = []
    fre_ptns = []
    seg_words = []
    seg_ptns = []

    labels = []
    char_Ids = []
    sing_char_word_Ids = []
    pos_Ids = []
    fre_word_Ids = []
    seg_word_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            char = pairs[0]  # 字
            if number_normalized:
                char = normalize_word(char)
            label = pairs[-1]  # 标签

            chars.append(char)  # 列表 存放句子
            labels.append(label)  # 列表 存放标签
            char_Ids.append(char_alphabet.get_index(char))  # 列表 存放id组成的句子
            label_Ids.append(label_alphabet.get_index(label))  # 列表 存放id组成的标签
            sing_char_word_Ids.append(word_alphabet.get_index(char))

        else:  # 一个句子结束
            if ((max_sent_length < 0) or (len(chars) < max_sent_length)) and (len(chars)>0):  # 如果句子长度满足要求
                sentence = ''.join(chars)
                seg_result = posseg.lcut(sentence, HMM=True)
                seg_sentence = [i.word for i in seg_result]
                seg_pos = [j.flag for j in seg_result]
                fre_sentence = segment_by_frequency(chars, word, word_alphabet, word_count)
                sentence_length = len(chars)  # 句子长度
                seg_sequence = []
                for i in range(len(seg_sentence)):
                    for j in range(len(seg_sentence[i])):
                        seg_sequence.append(seg_sentence[i])
                        poss.append(seg_pos[i])
                        pos_Ids.append(pos_alphabet.get_index(seg_pos[i]))

                aux_word = [ [] for _ in range(sentence_length)]
                aux_word_Id = [ [] for _ in range(sentence_length)]
                
                max_aw_num = 0
                count_fre = 0
                count_seg = 0
                for idx in range(sentence_length):  # [中，国，语，言，学] # id = 0
                    matched_list = word.enumerateMatchList(chars[idx:])
                    matched_list_seg = list.copy(matched_list)
                    matched_list_fre = list.copy(matched_list)
                    if idx == count_fre:
                        matched_word = None
                        for i in matched_list_fre:  # 可修改为条件句
                            if i == fre_sentence[idx]:
                                matched_word = i
                                break
                        if matched_word:
                            matched_list_fre.remove(matched_word)
                            word_len = len(matched_word)
                            matched_Id  = word_alphabet.get_index(matched_word)
                            count_fre += word_len
                            for j in range(word_len):
                                fre_words.append(matched_word)
                                fre_word_Ids.append(matched_Id)
                                if j == 0:
                                    if word_len == 1 :
                                        fre_ptns.append(4)
                                    else:
                                        fre_ptns.append(1)
                                elif j== word_len-1:
                                    fre_ptns.append(3)
                                else:
                                    fre_ptns.append(2)
                        else:
                            single = False
                            for i in matched_list_fre:
                                if len(i) == 1:
                                    matched_list_fre.remove(i)
                                    matched_Id  = word_alphabet.get_index(i)
                                    fre_words.append(i)
                                    fre_word_Ids.append(matched_Id)
                                    fre_ptns.append(4)
                                    count_fre += 1
                                    single = True
                                    break
                            if not single:    
                                fre_ptns.append(4)
                                count_fre += 1
                                fre_words.append(word_alphabet.UNKNOWN)
                                fre_word_Ids.append(word_alphabet.get_index(word_alphabet.UNKNOWN))

                    if idx == count_seg:
                        matched_word = None
                        for i in matched_list_seg:
                            if i == seg_sequence[idx]:
                                matched_word = i
                                break
                        if matched_word:
                            matched_list_seg.remove(matched_word)
                            word_len = len(matched_word)
                            matched_Id  = word_alphabet.get_index(matched_word)
                            count_seg += word_len
                            for j in range(word_len):
                                seg_words.append(matched_word)
                                seg_word_Ids.append(matched_Id)
                                if j == 0:
                                    if word_len == 1 :
                                        seg_ptns.append(4)
                                    else:
                                        seg_ptns.append(1)
                                elif j== word_len-1:
                                    seg_ptns.append(3)
                                else:
                                    seg_ptns.append(2)
                        else:
                            single = False
                            for i in matched_list_seg:
                                if len(i) == 1:

                                    matched_list_seg.remove(i)
                                    matched_Id  = word_alphabet.get_index(i)
                                    seg_words.append(i)
                                    seg_word_Ids.append(matched_Id)
                                    seg_ptns.append(4)
                                    count_seg += 1
                                    single = True
                                    break
                            if not single:    
                                seg_ptns.append(4)
                                count_seg += 1
                                seg_words.append(word_alphabet.UNKNOWN)
                                seg_word_Ids.append(word_alphabet.get_index(word_alphabet.UNKNOWN))

                    for c in matched_list:
                        if c in matched_list_fre and c in matched_list_seg:
                            if len(c)>1:
                                for i in range(len(c)):
                                    aux_word[idx+i].append(c)
                                    aux_word_Id[idx+i].append(word_alphabet.get_index(c))
                    if not aux_word_Id[idx]:
                        aux_word_Id[idx].append(0)
                    max_aw_num = max(len(aux_word_Id[idx]), max_aw_num)
                aux_word_masks = []
                for idx in range(sentence_length):
                    aw_num = len(aux_word_Id[idx])
                    aw_mask = aw_num * [0]
                    aw_mask += (max_aw_num - aw_num) * [1]
                    aux_word_Id[idx] += (max_aw_num - aw_num) * [0]
                    aux_word_masks.append(aw_mask)

                instence_texts.append([chars, poss, fre_words, seg_words, aux_word, labels])
                instence_Ids.append([char_Ids, pos_Ids, fre_word_Ids, seg_word_Ids, label_Ids, fre_ptns, seg_ptns, aux_word_Id, aux_word_masks, sing_char_word_Ids])
            chars = []
            poss = []
            fre_words =[]
            seg_words = []
            fre_ptns = []
            seg_ptns = []
            labels = []
            char_Ids = []
            pos_Ids = []
            fre_word_Ids = []
            seg_word_Ids = []
            label_Ids = []
            sing_char_word_Ids = []

    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    pretrain_emb[0,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim