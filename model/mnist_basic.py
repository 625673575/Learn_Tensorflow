import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
with tf.name_scope('weight'):
    W = tf.Variable(tf.zeros([784, 10]))
    variable_summaries(W)
with tf.name_scope('bias'):
    b = tf.Variable(tf.zeros([10]))
    variable_summaries(b)

y_ = tf.placeholder(tf.float32, shape=[None, 10])#实际的y label
y = tf.nn.softmax(tf.matmul(x, W) + b)#运算过程中产生的y label
with tf.name_scope("cross_entropy"):
    # cross_entropy=tf.reduce_mean(tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    cross_entropy =tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=(tf.matmul(x, W) + b)))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc', accuracy)

sess = tf.InteractiveSession()
tf.summary.image()
train_writer = tf.summary.FileWriter('/tmp/mnist_basic/', sess.graph)#必须要先创建FileWriter然后再初始化全局变量
tf.global_variables_initializer().run()
merged = tf.summary.merge_all()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary,_=sess.run([merged,train_step] , feed_dict={x: batch_xs, y_: batch_ys})
    #train_writer.add_summary(summary,i)

train_writer.close()
px = mnist.test.images[34]
plt.imshow(np.reshape(px, [28, 28]))
px = tf.expand_dims(px, 0)
predict = tf.matmul(px, W) + b
print('predict cross entroppy', sess.run(predict))
possibility = tf.squeeze(tf.nn.softmax(predict), axis=0)
print('possibility', sess.run((possibility)))
result = tf.arg_max(possibility,dimension=0)
print('result', sess.run(result))

plt.show()
