import numpy as np
import os
import sys
import re
import math
import json
import copy


def read_data(config):
	data_types = ['train', 'dev']
	max_len_context = 0
	max_len_question = 0
	train_data = {}
	valid_data = {}
	data = {}

	###  read   ###
	for data_type in data_types:
		with open(os.path.join(config.data_dir, "{}-v1.1.json".format(data_type)), "r", encoding='utf-8', errors='ignore') as f:
			tmp = json.load(f)
			for i in range(len(tmp["data"])):
				for j in range(len(tmp["data"][i]["paragraphs"])):
					context = tmp["data"][i]["paragraphs"][j]["context"]
					tmp["data"][i]["paragraphs"][j]["context"] = context.split()
					context = context.split()
					tmp["data"][i]["paragraphs"][j]["seqlen"] = len(context)
					if len(context) > max_len_context:
						max_len_context = len(context)
					for k in range(len(tmp["data"][i]["paragraphs"][j]["qas"])):
						question = tmp["data"][i]["paragraphs"][j]["qas"][k]["question"]
						tmp["data"][i]["paragraphs"][j]["qas"][k]["question"] = question.split()
						question = question.split()
						tmp["data"][i]["paragraphs"][j]["qas"][k]["quelen"] = len(question)
						if len(question) > max_len_question:
							max_len_question = len(question)
			if data_type == "train":
				train_data = tmp
			else:
				valid_data = tmp
	
	data["train"] = copy.deepcopy(train_data)
	data["valid"] = copy.deepcopy(valid_data)

	# shuffle all data #
	np.random.shuffle(data["train"]["data"])
	np.random.shuffle(data["valid"]["data"])


	# padding #
	padding(config, max_len_context, max_len_question, data)

	print("Loading {} all data".format(len(data["train"]["data"])+len(data["valid"]["data"])))
	print("Loading {} train data".format(len(data["train"]["data"])))
	print("Loading {} valid data".format(len(data["valid"]["data"])))
	print("max length of context :  {}".format(max_len_context))
	print("max length of question :  {}".format(max_len_question))
		
	return data, max_len_context, max_len_question



def padding(config, max_len_context, max_len_question, data):
	###  padding  ###
	for i in ["train","valid"]:
		for j, j_val in enumerate(data[i]["data"]):
			for k, k_val in enumerate(j_val["paragraphs"]):
				if len(k_val["context"]) < max_len_context:
					for _ in range(len(k_val["context"]), max_len_context):
						k_val["context"].append("$PAD$")
					for l, l_val in enumerate(k_val["qas"]):
						for _ in range(len(l_val["question"]), max_len_question):
							l_val["question"].append("$PAD$")
	print("the 'padding' process has been completed")


def cross_validation(config, all_data):
	num_of_folds = config.cross_valid_k
	
	print("cross-validation : {}".format(config.cross_validation))
	test_data = all_data[int(len(all_data)*config.train_data_ratio):len(all_data)]
	print('test data len', len(test_data))
	cross_data = []
	data = {}
	fold_size = int(len(all_data)*config.train_data_ratio) // config.cross_valid_k
	for fold_idx in range(config.cross_valid_k) :
		print("fold {}".format(fold_idx+1))
		train_data = []
		valid_data = []
		cross_data_tmp = {}
		train_start = fold_idx * fold_size
		valid_start = int((fold_size*(fold_idx+num_of_folds*config.train_data_ratio))%len(all_data))
		train_end = int(train_start+fold_size*num_of_folds*config.train_data_ratio)
		valid_end = int(valid_start+fold_size)
		print(train_start, train_end, valid_start, valid_end)
		
		if train_end <= len(all_data):
			train_data = all_data[train_start:train_end]
		else:
			train_tmp1 = all_data[0:train_end-len(all_data)]
			train_tmp2 = all_data[train_start:len(all_data)]
			train_data.extend(train_tmp1)
			train_data.extend(train_tmp2)
			
		valid_data = all_data[valid_start:valid_end]
		print(len(train_data), len(valid_data), len(test_data))
		cross_data_tmp['train_data'] = train_data
		cross_data_tmp['valid_data'] = valid_data
		cross_data.append(cross_data_tmp)
	
	data['cross_data'] = cross_data
	data['test_data'] = test_data

	return data


def tokenize(context):
	context = re.sub('[=#&+_$*/,.\-;:()\\[\]?!"\']+', ' ', context)
	context = re.sub('[0-9]+', '', context)
	context = context.replace('\\n', ' ').replace('\\','')
	context = context.split()
	return context


def write_wordic(config, data):
	print("writing a dictionary . . . ")
	wordic = ['$UNK$','$PAD$']
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



def one_hot(data_X, data_Y, max_len_context, wordic):
	input_X = np.array([[[]]])
	input_Y = np.array([[]])

	for loop, data in enumerate(data_X):
		one_hot = np.zeros((max_len_context, len(wordic)))
		one_hot[np.arange(max_len_context), 
				np.array([wordic.index(data[i]) if data[i] in wordic else wordic.index('$UNK$') for i in range(max_len_context)])] = 1
		one_hot = one_hot.reshape(1, max_len_context, len(wordic))

			#one_hot = np.array([[[int(i == wordic.index(contents['context'][j])) for i in range(len(wordic))] if contents['context'][j] in wordic else [int(i == wordic.index('$UNK$')) for i in range(len(wordic))] for j in range(max_len_context)]])
		if loop == 0:
			input_X = one_hot
			input_Y = np.array([[int(data_Y[loop]=="neg"), int(data_Y[loop]=="pos")]])
			continue
		else:
			input_X = np.concatenate((input_X, one_hot))
			input_Y = np.concatenate((input_Y, np.array([[int(data_Y[loop]=="neg"), int(data_Y[loop]=="pos")]])))

	print(input_X.shape, input_Y.shape)

	return input_X, input_Y


def word2index(data_X, data_Y, wordic):
	input_x = data_X
	input_y = np.array([[]])
	for i, para in enumerate(input_x):
		for j, word in enumerate(para):
			if word in wordic:
				input_x[i][j] = wordic.index(word)
			else:
				input_x[i][j] = wordic.index('$UNK$')
	for loop, senti in enumerate(data_Y):
		if loop == 0:
			input_y = np.array([[int(data_Y[loop]=="neg"), int(data_Y[loop]=="pos")]])
		else:
			input_y = np.concatenate((input_y, np.array([[int(data_Y[loop]=="neg"), int(data_Y[loop]=="pos")]])))

	return input_x, input_y


