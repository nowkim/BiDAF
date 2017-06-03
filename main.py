import numpy as np
import tensorflow as tf
import sys
import time
import copy

from config import get_config
from prepro import read_data, write_wordic, word2index
from model import BiDAF
from evaluator import run_epoch


def main():
	config = get_config()

	#####  pre-process #####

	# read data
	data, max_len_context, max_len_question = read_data(config)

	
	# write a word dictionary
	wordic = write_wordic(config, data['train'])
	

	with tf.Graph().as_default():
	
		# build the model
		with tf.variable_scope("Model"):
			print("< build the whole model >")
			squad_model = BiDAF(config=config, 
													seqlen=max_len_context,
													quelen=max_len_question,
													wordic=wordic)
		
		print("= FINISH =")
		saver = tf.train.Saver()
'''
		val_acc = 0
		for i in range(config.max_epoch):
			lr_decay = config.lr_decay ** max(i + 1 - config.lr_init_epoch, 0.0)
			Model.assign_lr(config.lr * lr_decay)
			print("### {} epochs ###".format(i+1))
			now = time.localtime()
			print("current time {}:{}:{}".format(now.tm_hour, now.tm_min, now.tm_sec))
			run_epoch(config, Model, wordic, step='train', data=data['train_data'])
			val_acc += run_epoch(config, Model, wordic, step='valid', data=data['test_data'])
			
			end = time.localtime()
			epoch_time = end.tm_hour*3600 + end.tm_min*60 + end.tm_sec - (now.tm_hour*3600 + now.tm_min*60 + now.tm_sec)
			print("epoch time : {}".format(epoch_time))
			
		val_acc = val_acc / config.max_epoch
		print("validation accuracy : ", round(val_acc,3))

'''

		
if __name__ == "__main__":
	main()

