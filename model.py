import tensorflow as tf


def bidirectional_LSTM(X, time_step, X_len, hidden_size, output_dim):
	print("  - call -> bidirectional_LSTM()   ")
	lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
	lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
			
	outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, 
																				X, sequence_length=X_len, dtype=tf.float32)

	stacked_output = tf.stack(outputs)
	# outputs : shape=(seq_size, batch_size, 2*lstm_size)
	stacked_output = tf.transpose(stacked_output, [1,0,2])  
	# outputs : shape=(batch_size, seq_size, 2*lstm_size)

	X = tf.stack(X)
	# (time step, ?, d)
	W = tf.Variable(tf.random_normal([2*hidden_size, int(X.get_shape().as_list()[-1]*output_dim)]))
	b = tf.Variable(tf.constant(0.0, shape = [int(X.get_shape().as_list()[-1]*output_dim)]))

	reshaped_output = tf.reshape(stacked_output, [-1, 2*hidden_size])
	# (batch * time step , 2*lstm_size)
	reshaped_output = tf.matmul(reshaped_output, W) + b
	# (?, output_dim * d)

	Y = tf.reshape(reshaped_output, [-1, time_step, int(X.get_shape().as_list()[-1]*output_dim)])
	# (?, time step, output_dim * d)
	Y = tf.transpose(Y, [0, 2, 1])
	# (?, output_dim * d, time step)

	return Y
