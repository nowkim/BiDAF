import tensorflow as tf
import numpy as np
import datetime
import sys
import copy

from prepro import word2index
from evaluate import *

def run_epoch(config, model, wordic, step, data):
	sess = model.session
	batch_size = config.batch_size
	Data = copy.deepcopy(data)
	
	context = []
	question = []
	answer = []
	for article in Data['train']['data']:
		for paragraph in article['paragraphs']:
			context.append(paragraph['context'])
			q_tmp = []
			a_tmp2 = []
			for qa in paragraph['qas']:
				q_tmp.append(qa['question'])
				a_tmp1 = []
				for a in qa['answers']:
					a_tmp1.append(a['text'])
				a_tmp2.append(a_tmp1)
			question.append(q_tmp)
			answer.append(a_tmp2)

	data_idx = 0
	acc_sum = 0
	acc = 0
	for iteration in range(len(context)//batch_size):
		if step == "train":
			keep_prob = config.keep_prob
		else:
			keep_prob = 1.0
		feed_dict = {
					model.x : batch_x,
					model.y : batch_y,
					model.seqlen : batch_x_len,
					model.keep_prob : keep_prob
			}
		
		if step == 'train':
			_ = sess.run(model.train, feed_dict=feed_dict)
			if (iteration+1) != 1 and (iteration+1) % 10 == 0:
				time_str = datetime.datetime.now().isoformat()
				global_step, loss, lr = sess.run(
								[model.global_step, model.loss, model.lr], feed_dict=feed_dict)
				print("{}, {}, Loss : {:g}, lr : {:g}".format(time_str, global_step,loss,accuracy,lr))
		else:
			time_str = datetime.datetime.now().isoformat()
			global_step, loss, lr = sess.run(
												[model.global_step, model.loss, model.lr], feed_dict=feed_dict)
			print("{}, {}, Loss : {:g}, lr : {:g}".format(time_str, global_step,loss,accuracy,lr))
			
			acc_sum += accuracy
		
			if iteration + 1 >= len(context)//batch_size:
				acc = acc_sum / iteration
				print("valid acc : {}".format(acc))
				
				return acc



