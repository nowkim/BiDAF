import tensorflow as tf
import numpy as np
import datetime
import sys
import copy

from prepro import word2index
from evaluate import *

def run_epoch(config, model, wordic, max_len_context, step, data):
    sess = model.session
    batch_size = config.batch_size
    Data = copy.deepcopy(data)
    
    iteration = 0
    context, question, answer, context_len, question_len, y1, y2 = [],[],[],[],[],[],[]
    a_tmp, as_tmp, ae_tmp = [], [], []
    for data_idx, article in enumerate(Data['data']):
        for context_idx, paragraph in enumerate(article['paragraphs']):
            for q_idx, qa in enumerate(paragraph['qas']):
                if len(question) == batch_size:
                    iteration += 1
                    if step == 'train': keep_prob = config.keep_prob
                    else : keep_prob = 1
                    feed_dict = {
                        model.x : word2index(context, wordic),
                        model.q : word2index(question, wordic),                    
                        model.context_len : context_len,
                        model.question_len : question_len,
                        model.y1 : y1,
                        model.y2 : y2,
                        model.keep_prob : keep_prob
                    }
                    if step == 'train':
                        _ = sess.run(model.train, feed_dict=feed_dict)
                        if iteration % 10 == 0:
                            time_str = datetime.datetime.now().isoformat()
                            print("train start")
                            logit_s, logit_e, global_step, loss, grads, lr = sess.run(
                                    [model.logit_s, model.logit_e, model.global_step, model.loss, model.grads, model.lr], 
                                    feed_dict=feed_dict)
                            print("{}, {}, Loss : {:g}, lr : {:g}".format(time_str,global_step, loss, lr))
                            _, _ = get_score(logit_s, logit_e, context, context_len, answer)
                            print("train complete")
                    else:
                        time_str = datetime.datetime.now().isoformat()
                        print("validation start")
                        logit_s, logit_e, global_step, loss, grads, lr = sess.run(
                                [model.logit_s, model.logit_e, model.global_step, model.loss, model.grads, model.lr], 
                                feed_dict=feed_dict)
                        print("{}, {}, Loss : {:g}, lr : {:g}".format(time_str,global_step, loss, lr))
                        em, f1 = get_score(logit_s, logit_e, context, context_len, answer)
                        print("validation complete")
                        return em, f1
                    context, question, answer, context_len, question_len, y1, y2 = [],[],[],[],[],[],[]
                a_tmp, as_tmp, ae_tmp = [], [], []
                as_tmp = np.zeros(max_len_context)
                ae_tmp = np.zeros(max_len_context)
                if max_len_context > qa['answers'][0]['answer_start']:
                    as_tmp[qa['answers'][0]['answer_start']] = 1
                else: as_tmp[max_len_context-1] = 1
                if max_len_context > len(qa['answers'][0]['text'])+qa['answers'][0]['answer_start']:
                    ae_tmp[len(qa['answers'][0]['text'])+qa['answers'][0]['answer_start']] = 1
                else: ae_tmp[max_len_context-1] = 1
                if step == 'train':
                    a_tmp = qa['answers'][0]['text']
                else:
                    for a in qa['answers']:
                        a_tmp.append(a['text'])
                if len(question) < batch_size:
                    context.append(copy.deepcopy(paragraph['context']))
                    question.append(copy.deepcopy(qa['question']))
                    answer.append(a_tmp)
                    context_len.append(paragraph['seqlen'])
                    question_len.append(qa['quelen'])
                    y1.append(as_tmp)
                    y2.append(ae_tmp)


def get_score(logit_s, logit_e, context, context_len, answer):
    start_idx = [np.argmax(sl[:cl], 0)
            for sl, cl in zip(logit_s, context_len)]
    end_idx = [np.argmax(el[si:cl], 0) + si
            for el, si, cl in zip(logit_e, start_idx, context_len)]
    predictions = []
    for c, s_idx, e_idx in zip(context, start_idx, end_idx):
        predictions.append(' '.join([w for w in c[s_idx:e_idx+1]]))
    em = f1 = 0
    cnt = 0
    for prediction, ground_truth in zip(predictions, answer):
        single_em = metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truth)
        single_f1 = metric_max_over_ground_truths(
                f1_score, prediction, ground_truth)
        em += single_em
        f1 += single_f1
        if f1 > 0 :
            print("pred : {} , s_idx : {}, e_idx : {}".format(
                str(' '.join(prediction)), start_idx[cnt], end_idx[cnt]))
            print("real : " + str(ground_truth))
        cnt += 1
    print("em score : ", em / len(predictions))
    print("f1 score : ", f1 / len(predictions))
    return em, f1



def train(config, model, wordic, data):
    batch_size = config.batch_size
    Data = copy.deepcopy(data)
    feed_data = {}
    c_minus_q = 0
    for data_idx, article in enumerate(Data['data']):
        for c_idx, paragraph in enumerate(article['paragraphs']):
            for q_idx, qa in enumerate(paragraph['qas']):
                if (q_idx+1)%batch_size < batch_size:
                    question.append(qa['question'])
                    if c_minus_q == q_idx - c_idx:
                        context.append(paragraph['context'])
                    else: context.append('$COPY$')
                else:
                    question, context = [], []
                    question.append(qa['question'])
                    context.append(paragraph['context'])
                    
                    feed_data['question'] = question
                    feed_data['context'] = context
                    for text_type in ['context', 'question']:
                        for idx, seq in enumerate(feed_data[text_type]):
                            if seq == '$COPY$': seq = feed_data[text_type][idx-1]
                            else: seq = word2index(seq, wordic)
                    feed(config, model, feed_data)
                
                c_idx_change = q_idx - c_idx


def feed(config, model, feed_data):
    sess = model.session
     
    feed_dict = {
        model.x : feed_data['context'],
        model.q : feed_data['context'],                    
        model.context_len : feed_data['context'],
        model.question_len : feed_data['context'],
        model.y1 : feed_data['context'],
        model.y2 : feed_data['context'],
        model.keep_prob : feed_data['context']
    }
    _ = sess.run(model.train, feed_dict=feed_dict)

    if iteration % 10 == 0:
        print("iteration * 10 !!!")
        time_str = datetime.datetime.now().isoformat()
        global_step, loss, lr = sess.run(
                [model.global_step, model.loss, model.lr], feed_dict=feed_dict)
        print("{}, {}, Loss : {:g}, lr : {:g}".format(time_str,global_step,loss, lr))
        ans_s, ans_e = model.ans_s, model.ans_e
        print('answer! : ', ans_s, ans_e)



