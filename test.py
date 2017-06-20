""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import copy


def reset_idx(context, s_idx):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    s = copy.deepcopy(context[:s_idx])
    sl = len(s)
    fs = white_space_fix(remove_articles(remove_punc(lower(s))))
    fsl = len(fs)
    return s_idx - (sl - fsl) + 1


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    print(" f1 ~~~~ ")
    prediction_tokens = normalize_answer(prediction).split()
    print("prediction tokens : ", prediction_tokens)
    ground_truth_tokens = normalize_answer(ground_truth).split()
    print("ground truth tokens : ", ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    print("common : ", common)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    print("precision, recall : ", precision, recall)
    f1 = (2 * precision * recall) / (precision + recall)
    print("f1 : ", f1)
    return f1


def exact_match_score(prediction, ground_truth):
    print(" ----- em  !!! ")
    print("norm pred : ", normalize_answer(prediction))
    print("norm ground truth : ", normalize_answer(ground_truth))
    print(" em ? : ", normalize_answer(prediction) == normalize_answer(ground_truth))
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                            ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

'''
em = f1 = 0
predictions = ["by limits of parish of st philip and st michael now also includes parts of st james parish st georges parish st andrews parish and st johns parish although last two are mostly still incorporated rural parishes"]
answer = [["St. Andrew's Parish"]]
for prediction, ground_truth in zip(predictions, answer):
    single_em = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truth)
    single_f1 = metric_max_over_ground_truths(
            f1_score, prediction, ground_truth)
    print("sigle_em, f1 : ", single_em, single_f1)
    em += single_em
    f1 += single_f1
    print("pred : " + prediction)
    print("real : " + ground_truth)
print("em score : ", em / len(predictions))
print("f1 score : ", f1 / len(predictions))
'''

start_idx = [np.argmax(sl[:cl], 0)
        for sl, cl in zip(logit_s, context_len)]
end_idx = [np.argmax(el[si:cl], 0) + si
        for el, si, cl in zip(logit_e, start_idx, context_len)]
print("start, end idx : ", start_idx, end_idx)


