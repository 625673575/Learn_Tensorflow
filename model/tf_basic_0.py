# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:33:44 2017

@author: Liu
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf


def threeAdd():
    input =np.array( [[0, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 1, 0],
             [1, 0, 1],
             [0, 1, 1],
             [1, 1, 1]],dtype=np.float32)
    output = np.sum(input*[1,8,4], axis=1)+2
    print(input.shape,output)

    x = tf.placeholder(tf.float32,name='x')
    y = tf.placeholder(tf.float32,name='y')
    w = tf.Variable([0.4,-0.5,2.4])
    b=tf.Variable(0.427384)
    sess = tf.InteractiveSession()

    model =tf.reduce_sum(w*x,axis=1)+b
    loss = tf.nn.l2_loss(model - y)

    train = tf.train.AdamOptimizer(0.1).minimize(loss)
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train, {x: input, y: output})
    print(sess.run([w,b, model], {x: [[1,1,1]]}))

sess = tf.Session()

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
plt.plot(x_train,y_train,'o')
plt.plot(x_train,curr_W*x_train+curr_b,'-')
plt.suptitle('Line Regression')
plt.title( 'loss='+str( curr_loss))
plt.show()
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    