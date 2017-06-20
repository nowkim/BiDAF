import tensorflow as tf


def bidirectional_LSTM(config, keep_prob, X, time_step, X_len, output_dim):
    hidden_size = config.lstm_hidden_size
    print("  - call -> bidirectional_LSTM()   ")
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    fw_drop = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
    bw_drop = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)


    outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_drop, bw_drop, 
            X, sequence_length=X_len, dtype=tf.float32)

    stacked_output = tf.stack(outputs)
    # outputs : shape=(seq_size, batch_size, 2*lstm_size)
    stacked_output = tf.transpose(stacked_output, [1,0,2])  
    # outputs : shape=(batch_size, seq_size, 2*lstm_size)

    X = tf.stack(X)
    #X = tf.Print(X, [X], "bidir X : ", summarize = 200)
    # (time step, ?, d)
    W = tf.Variable(tf.random_normal([2*hidden_size, int(X.get_shape().as_list()[-1]*output_dim)]))
    #W = tf.Print(W, [W], "bidir W : ", summarize = 200)
    b = tf.Variable(tf.constant(0.0, shape = [int(X.get_shape().as_list()[-1]*output_dim)]))
    #b = tf.Print(b, [b], "bidir b : ", summarize = 200)

    reshaped_output = tf.reshape(stacked_output, [-1, 2*hidden_size])
    # (batch * time step , 2*lstm_size)
    reshaped_output = tf.matmul(reshaped_output, W) + b
    # (?, output_dim * d)

    Y = tf.reshape(reshaped_output, [-1, time_step, int(X.get_shape().as_list()[-1]*output_dim)])
    # (?, time step, output_dim * d)
    Y = tf.transpose(Y, [0, 2, 1])
    #Y = tf.Print(Y, [Y], "bidir Y : ", summarize = 200)
    # (?, output_dim * d, time step)

    return Y
