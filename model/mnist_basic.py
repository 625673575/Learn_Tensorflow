import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
with tf.name_scope('weight'):
    W = tf.Variable(tf.zeros([784, 10]))
with tf.name_scope('bias'):
    b = tf.Variable(tf.zeros([10]))

y_ = tf.placeholder(tf.float32, shape=[None, 10])#实际的y label
y = tf.nn.softmax(tf.matmul(x, W) + b)#运算过程中产生的y label
with tf.name_scope("cross_entropy"):
    # cross_entropy=tf.reduce_mean(tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    cross_entropy =tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=(tf.matmul(x, W) + b)))
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

train_writer = tf.summary.FileWriter('/tmp/MNIST/', sess.graph)

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

px = mnist.test.images[34]
plt.imshow(np.reshape(px, [28, 28]))
px = tf.expand_dims(px, 0)
predict = tf.matmul(px, W) + b
print('predict cross entroppy', sess.run(predict))
possibility = tf.squeeze(tf.nn.softmax(predict), axis=0)
print('possibility', sess.run((possibility)))
result = tf.where(tf.equal(possibility, tf.constant(1.0, tf.float32)))
print('result', sess.run(tf.squeeze(result)))

plt.show()
