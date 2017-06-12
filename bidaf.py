import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib.tensorboard.plugins import projector
import os
import time

from model import bidirectional_LSTM

class BiDAF(object):
    def __init__(self, config, seqlen, quelen, wordic):
        sessConfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sessConfig.gpu_options.allow_growth = True
        self.session = tf.Session(config=sessConfig)
        self.wordic = wordic
        self.vocab_size = len(wordic)
        self.con_time_step = seqlen
        self.que_time_step = quelen
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.Variable(0.0, trainable = False)

        # placeholders
        self.x = tf.placeholder(tf.int32, [None, self.con_time_step])
        self.q = tf.placeholder(tf.int32, [None, self.que_time_step])	
        self.context_len = tf.placeholder(tf.int32, [None])
        self.question_len = tf.placeholder(tf.int32, [None])
        self.last_batch_idx = tf.placeholder(tf.int32, [None])
        self.y1 = tf.placeholder(tf.int32, [None, None])
        self.y2 = tf.placeholder(tf.int32, [None, None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.new_lr = tf.placeholder(tf.float32, shape=[])


        # build the model layer
        #self.char_embeded_x = self.character_embedding_layer(
        #									config, self.x, self.q, self.context_len, self.question_len, self.last_batch_idx)
        self.X, self.Q = self.word_embedding_layer(config, self.wordic, self.x, self.q,
                                    self.con_time_step, self.que_time_step)
        print("X : ", tf.stack(self.X))
        print("Q : ", tf.stack(self.Q))
        self.H, self.U = self.contextual_embedding_layer(config, self.X, self.Q,  
                        self.con_time_step, self.que_time_step, self.context_len, self.question_len)
        print("H : ", tf.stack(self.H))
        print("U : ", tf.stack(self.U))
        self.G = self.attention_flow_layer(config, self.H, self.U)
        print("G : ", self.G)
        self.M = self.modeling_layer(config, self.G, self.con_time_step, self.context_len)
        print("M : ", self.M)
        self.logit_s, self.logit_e = self.output_layer(config, self.G, self.M, self.con_time_step, self.context_len)
        print("Whole layer has been built !")
        
        # optimize and evaluate	
        self.optimize(config, self.logit_s, self.logit_e, self.y1, self.y2)

        # initialize whole parameters
        self.initialize()


    def initialize(self):
        self.session.run(tf.global_variables_initializer())

    def character_embedding_layer(self, config, x, q, context_len, question_len, last_idx):
        with tf.variable_scope("Character_embeding_layer") as v_scope:
            embeddings = tf.Variable(tf.random_uniform([80, config.char_embedding_size], -1.0, 1.0),
                                                                                    name="char_embed_table")
            for switch in [("context", x, context_len), ("question", q, question_len)]:
                with tf.variable_scope(switch[0]) as v_scope:
                    print("** " + str(layer_scope) + "(" + switch[0] + ")")
                    words = switch[1]
                    print(words)

                    #embed = tf.nn.embedding_lookup(embeddings, )
                    

    def higyway_network(self):
        return

    def word_embedding_layer(self, config, wordic, x, q, con_time_step, que_time_step):
        with tf.variable_scope("Word_embeding_layer") as v_scope:
            now = time.localtime()
            print("** Word embedding layer ({}:{}:{})".format(now.tm_hour, now.tm_min, now.tm_sec))
            glove = {}
            embedding_table = []
            with open('data/glove/glove.6B.'+str(config.word_embedding_size)+'d.txt', 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    line = line.split()
                    for idx in range(len(line[1:])):
                        line[idx+1] = float(line[idx+1])
                    glove[line[0]] = line[1:]

            for _, word in enumerate(wordic):
                if not word in glove.keys():
                    embedding_table.append([0.0 for _ in range(config.word_embedding_size)])
                else:
                    embedding_table.append(glove[word])

            embeddings = tf.Variable(embedding_table, trainable=False, name='word_embedding')
    
            embed_x = tf.nn.embedding_lookup(embeddings, x)
            embed_q = tf.nn.embedding_lookup(embeddings, q)
                    # embed : shape = (batch_size, time_step_size, embedding_size)
            embed_x = tf.unstack(embed_x, con_time_step, 1)
            embed_q = tf.unstack(embed_q, que_time_step, 1)
                    # embed : shape = (batch_size, embedding_size) * time_step_size
        return embed_x, embed_q
            
    
    def contextual_embedding_layer(self, config, X, Q, con_time_step, que_time_step, context_len, question_len):
        with tf.variable_scope("Contextual_embeding_layer") as v_scope:
            for switch in [("context", X, con_time_step, context_len), ("question", Q, que_time_step, question_len)]:
                with tf.variable_scope(switch[0]) as v_scope:
                    now = time.localtime()
                    print("*** Contextual embeding layer - {} ({}:{}:{})".format(switch[0], now.tm_hour, now.tm_min, now.tm_sec))
                    if switch[0] == "context":
                        H = bidirectional_LSTM(config, self.keep_prob, switch[1], switch[2], switch[3], 2)
                        # (?, context time step, 2d)
                    else:
                        U = bidirectional_LSTM(config, self.keep_prob, switch[1], switch[2], switch[3], 2)
                        # (?, question time step, 2d)

            return H, U
            

    
    def attention_flow_layer(self, config, H, U):	
        with tf.variable_scope("Attention_flow_layer") as v_scope:
            now = time.localtime()
            print("**** Attention flow layer ({}:{}:{})".format(now.tm_hour, now.tm_min, now.tm_sec))
            H = tf.transpose(H, [0, 2, 1])
            # (?, context time step, 2d)
            print(H, U)
            
            S = tf.matmul(H, U)
            # (?, context time step, que time step)
            print('S : ',S)
            
            a = tf.nn.softmax(S, -1)
            # (?, context time step, que time step)
            print('a : ',a)

            attended_U = tf.matmul(U, tf.transpose(a, [0, 2, 1]))
            # (?, 2d, context time step)
            print('attended_U : ', attended_U)

            b = tf.nn.softmax(tf.reduce_max(S, 2), -1)
            # (?, context time step)
            print('b : ', b)

            H = tf.transpose(H, [0, 2, 1])

            attended_H = tf.multiply(H, tf.expand_dims(b, 1))
            # (?, 2d, context time step)
            print('attended_H : ', attended_H)

            h_el_mul_att_u = tf.multiply(H, attended_U)
            h_el_mul_att_h = tf.multiply(H, attended_H)
            print("hOu~ : ", h_el_mul_att_u)
            print("hOh~ : ", h_el_mul_att_h)

            G = tf.concat([H, attended_U, h_el_mul_att_u, h_el_mul_att_h], axis=1)
            
            return G


    def modeling_layer(self, config, G, con_time_step, context_len):
        with tf.variable_scope("Modeling_layer") as v_scope:
            now = time.localtime()
            print("***** Modeling layer ({}:{}:{})".format(now.tm_hour, now.tm_min, now.tm_sec))
            G = tf.unstack(G, con_time_step, 2)
            # (batch, 8d) * con_time_step
                                    
            M = bidirectional_LSTM(config, self.keep_prob, G, con_time_step, context_len, 1/4)
            # (?, question time step, 2d)

            return M

    def output_layer(self, config, G, M, con_time_step, context_len):
        with tf.variable_scope("Output_layer") as v_scope:
            now = time.localtime()
            print("****** Output layer ({}:{}:{})".format(now.tm_hour, now.tm_min, now.tm_sec))
            G_M = tf.concat([G, M], axis=1)
            print("G_M : ", G_M)
            # (?, 10d, time step)
            
            w_p1 = tf.Variable(tf.random_normal([5*M.get_shape().as_list()[1]]))
            print("w_p1 : ", w_p1)
            # (?, 10d)

            w_p1 = tf.expand_dims(w_p1, 1)
            w_p1 = tf.expand_dims(w_p1, 0)
            p1 = tf.reduce_sum(tf.multiply(w_p1, G_M), 1)
            print("p1 : ", p1)
            # (?, time step)

            M = tf.unstack(M, con_time_step, 2)
            # (batch_size, embedding_size) * time_step_size
            M2 = bidirectional_LSTM(config, self.keep_prob, M, con_time_step, context_len, 1)
            # (?, 2d, time step)

            G_M2 = tf.concat([G, M2], axis=1)
            print("G_M2 : ", G_M2)
            # (?, 10d, time step)

            M = tf.stack(M)
            print('M after stack : ', M)
            w_p2 = tf.Variable(tf.random_normal([5*M.get_shape().as_list()[-1]]))
            print("w_p2 : ", w_p2)
            # (?, 10d, time step)

            w_p2 = tf.expand_dims(w_p2, 1)
            w_p2 = tf.expand_dims(w_p2, 0)
            p2 = tf.reduce_sum(tf.multiply(w_p2, G_M2), 1)
            print("p2 : ", p2)
            # (?, time step)

            return p1, p2


    def optimize(self, config, logit_s, logit_e, y1, y2):
        now = time.localtime()
        print("optimizer building ({}:{}:{})".format(now.tm_hour, now.tm_min, now.tm_sec))
        loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_s, labels=y1))
        loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_e, labels=y2))
        self.loss = loss_1 + loss_2
        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.max_grad_norm)


        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train = optimizer.apply_gradients(zip(self.grads, tvars), global_step = self.global_step)

        self.lr_update = tf.assign(self.lr, self.new_lr)

    
    def assign_lr(self, lr_value):
        sess = self.session
        sess.run(self.lr_update, feed_dict={self.new_lr : lr_value})


            


