import numpy as np
import tensorflow as tf

t0 = tf.Variable(tf.random_uniform([5, 2]))
t1 = tf.Variable(tf.random_uniform([5, 2]))
A = tf.TensorArray(dtype = tf.float32, size = 2)
A = A.write(0, t0)
A = A.write(1, t1)

a0 = A.read(0)
a1 = A.read(1)

b = A.stack()

C = tf.TensorArray(dtype = tf.float32, size = 5)
C = C.unstack(t0) # TensorArray with {Tensor with size 2} x 5
c0 = C.read(0)
c1 = C.read(1)
c2 = C.read(2)
c3 = C.read(3)
c4 = C.read(4)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	print("t0=\n", sess.run(t0))
	print("t1=\n", sess.run(t1))
	print("a0=\n", sess.run(a0))
	print("a1=\n", sess.run(a1))
	print("b=\n", sess.run(b))
	print("c0=\n", sess.run(c0))
	print("c1=\n", sess.run(c1))
	print("c2=\n", sess.run(c2))
	print("c3=\n", sess.run(c3))
	print("c4=\n", sess.run(c4))


