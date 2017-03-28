# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:33:44 2017

@author: Liu
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.datasets as datasets

import sklearn.datasets as datasets

sess = tf.Session()
iris=datasets.load_iris()

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
#fixW = tf.assign(W, [-1.])
#fixb = tf.assign(b, [1.])
#sess.run([fixW, fixb])
#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
sess.run(init)
for i in range(999):
    sess.run(train,{x:x_train,y:y_train})
    
# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    