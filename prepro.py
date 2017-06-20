import numpy as np
import os
import sys
import re
import math
import json
import copy

from evaluate import normalize_answer, reset_idx

def read_data(config):
    data_types = ['train', 'dev']
    max_len_context = 0
    max_len_question = 0
    max_len_word = 0
    train_data = {}
    dev_data = {}
    data = {}
    train_q_num = 0
    dev_q_num = 0
    ###  read   ###
    for data_type in data_types:
        with open(os.path.join(config.data_dir, "{}-v1.1.json".format(data_type)), "r", encoding='utf-8', errors='ignore') as f:
            tmp = json.load(f)
            for i, i_val in enumerate(tmp['data']):
                for j, j_val in enumerate(i_val['paragraphs']):
                    context = j_val["context"]
                    for k, k_val in enumerate(j_val['qas']):
                        # normalize qustions and read data
                        if data_type == 'train':train_q_num += 1
                        else: dev_q_num += 1
                        question = k_val["question"]
                        k_val["question"] = normalize_answer(question).split()
                        question = normalize_answer(question).split()
                        k_val["quelen"] = len(question)
                        wordlen = []
                        for word in question:
                            if len(word) > max_len_word:
                                max_len_word = len(word)
                            wordlen.append(len(word))
                        k_val["wordlen"] = wordlen
                        if len(question) > max_len_question:
                            max_len_question = len(question)
                        for l, l_val in enumerate(k_val["answers"]) :
                            l_val["answer_start"] = reset_idx(context, l_val["answer_start"])
                    # normalize contexts and read data
                    j_val["context"] = normalize_answer(context).split()
                    context = normalize_answer(context).split()
                    j_val["seqlen"] = len(context)
                    wordlen = []
                    for word in context:
                        if len(word) > max_len_word:
                            max_len_word = len(word)
                        wordlen.append(len(word))
                    j_val["wordlen"] = wordlen
                    if len(context) > max_len_context:
                        max_len_context = len(context)
            if data_type == "train":
                train_data = tmp
            else:
                dev_data = tmp

    data["train"] = copy.deepcopy(train_data)
    data["dev"] = copy.deepcopy(dev_data)

    # shuffle all data #
    np.random.shuffle(data["train"]["data"])
    np.random.shuffle(data["dev"]["data"])
    
    # padding #
    #max_len_context = 100
    padding_with_limit(max_len_context, max_len_question, data)

    '''
    # padding test #
    for i, i_v in enumerate(data['train']['data']):
        for j, j_v in enumerate(i_v['paragraphs']):
            if len(j_v['context']) != 100: print(j_v['context'])
            for l, l_v in enumerate(j_v['qas']):
                if len(l_v['question']) != max_len_question: print(l_v['question'])
    '''

    print("Loading {} all data".format(len(data["train"]["data"])+len(data["dev"]["data"])))
    print("Loading {} train data".format(len(data["train"]["data"])))
    print("Loading {} dev data".format(len(data["dev"]["data"])))
    print("max length of context :  {}".format(max_len_context))
    print("max length of question :  {}".format(max_len_question))
    print("max length of word :  {}".format(max_len_word))
    print("the num of question : {}(train), {}(dev)".format(train_q_num, dev_q_num))

    return data, max_len_context, max_len_question, max_len_word


def padding_with_limit(c_limit, q_limit, data):
    ###  padding  ###
    for i in ["train","dev"]:
        for j, j_val in enumerate(data[i]["data"]):
            for k, k_val in enumerate(j_val["paragraphs"]):
                if len(k_val["context"]) < c_limit:
                    for j in range(len(k_val["context"]), c_limit):
                        k_val["context"].append("$PAD$")
                else:
                    k_val["context"] = k_val["context"][:c_limit]
                for l, l_val in enumerate(k_val["qas"]):
                    if len(l_val["question"]) < q_limit:
                        for _ in range(len(l_val["question"]), q_limit):
                            l_val["question"].append("$PAD$")
                    else:
                        l_val["question"] = l_val["question"][:q_limit]
    print("the 'padding' process has been completed")



def tokenize(context):
    context = re.sub('[=#&+_$*/,.\-;:()\\[\]?!"\']+', ' ', context)
    context = re.sub('[0-9]+', '', context)
    context = context.replace('\\n', ' ').replace('\\','')
    context = context.split()
    return context


def write_wordic(config, data):
    print("writing a dictionary . . . ")
    wordic = ['$PAD$','$UNK$']
    word_freq = {}

    for i, i_val in enumerate(data['data']):
        for j, j_val in enumerate(i_val["paragraphs"]):
            for word in j_val["context"]:
                if not word in word_freq.keys():
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
    for w,freq in word_freq.items():
        if not w in wordic and freq >= config.word_freq_limit:
            wordic.append(w)
    print("There are {} words in the dictionary".format(len(wordic)))
    return wordic





def word2index(data, wordic):
    input_x = copy.deepcopy(data)
    for i, para in enumerate(input_x):
        for j, word in enumerate(para):
            if word in wordic:
                input_x[i][j] = wordic.index(word)
            else:
                input_x[i][j] = wordic.index('$UNK$')
    return input_x
'''
def word2index(data, wordic):
    input_x = copy.deepcopy(data)
    for i, word in enumerate(input_x):
        if word in wordic:
            input_x[i] = wordic.index(word)
        else:
            input_x[i] = wordic.index('$UNK$')
    return input_x
'''




