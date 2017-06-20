import numpy as np
import tensorflow as tf
import sys
import time
import copy

from config import get_config
from prepro import read_data, write_wordic, word2index
from bidaf import BiDAF
from run import run_epoch
from evaluate import *

def main():
    config = get_config()

    #####  pre-process #####

    # read data
    data, max_len_context, max_len_question, max_len_word = read_data(config)

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
            print("- model building has been complete -")
    
            sum_em = sum_f1 = 0
            for i in range(config.max_epoch):
                now = time.localtime()
                print("### {} epochs ({}:{}:{}) ###".format(i+1, now.tm_hour, now.tm_min, now.tm_sec))
                lr_decay = config.lr_decay ** max(i + 1 - config.lr_init_epoch, 0.0)
                squad_model.assign_lr(config.lr * lr_decay)
                    
                run_epoch(config, squad_model, wordic, max_len_context, step='train', data=data['train'])
                em, f1 = run_epoch(config, squad_model, wordic, max_len_context, step='dev', data=data['dev'])
                sum_em += em
                sum_f1 += f1

                end = time.localtime()
                epoch_time = end.tm_hour*3600 + end.tm_min*60 + end.tm_sec - (now.tm_hour*3600 + now.tm_min*60 + now.tm_sec)
                print("spending time per epoch : {}".format(epoch_time))
                    
            em = sum_em / config.max_epoch
            f1 = sum_f1 / config.max_epoch
            print("EM , F1 : ", round(em, 3), round(f1, 3))

            
if __name__ == "__main__":
    main()

