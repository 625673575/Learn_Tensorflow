# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:33:44 2017

@author: Liu
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

with tf.device("/cpu:0"):
    with tf.Session() as sess:
        #max(features, 0)
        relu=tf.nn.relu(-2.0)
        print(sess.run(relu))

        relu=tf.nn.relu(5.0)
        print(sess.run(relu))

        relu=tf.nn.relu(0.)
        print(sess.run(relu))

        #min(max(features, 0), 6)
        relu6=tf.nn.relu6(8.5)
        print("relu6\n",sess.run(relu6))

        relu6=tf.nn.relu6(3.5)
        print(sess.run(relu6))

        relu6=tf.nn.relu6(-7.5)
        print(sess.run(relu6))

        #exp(features) - 1 if < 0
        elu=tf.nn.elu(8.5)
        print("elu\n",sess.run(elu))

        elu=tf.nn.elu(3.5)
        print(sess.run(elu))

        elu=tf.nn.elu(-0.5)
        print(sess.run(elu))

        elu=tf.nn.elu(-9.5)
        print(sess.run(elu))

        #log(exp(features) + 1)
        softplus=tf.nn.softplus(-2.0)
        print("softplus\n",sess.run(softplus))

        softplus=tf.nn.softplus(-20.0)
        print(sess.run(softplus))

        softplus=tf.nn.softplus(20.0)
        print(sess.run(softplus))

        #features / (abs(features) + 1)
        softsign=tf.nn.softsign(-2.5)
        print("softsign\n",sess.run(softsign))

        softsign=tf.nn.softsign(2.5)
        print(sess.run(softsign))

        #y = 1 / (1 + exp(-x))
        sigmoid=tf.nn.sigmoid(-0.5)
        print("sigmoid\n",sess.run(sigmoid))

        sigmoid=tf.nn.sigmoid(0.5)
        print(sess.run(sigmoid))

        sigmoid=tf.nn.sigmoid(3.5)
        print(sess.run(sigmoid))

        #y = 1 / (1 + exp(-x))
        tanh=tf.nn.tanh(-0.5)
        print("tanh\n",sess.run(tanh))

        tanh=tf.nn.tanh(0.5)
        print(sess.run(tanh))

        tanh=tf.nn.tanh(3.5)
        print(sess.run(tanh))

        #y = 1 / (1 + exp(-x))
        crelu=tf.nn.crelu([-2.5,55.,1.0,-2.5])
        print("crelu\n",sess.run(crelu))

        #变量初始化###############################################################
        i=tf.constant(2,tf.int32)
        f=tf.constant(3.5,tf.float32)

        logits = tf.Variable(tf.random_normal([5, 5], stddev=0.35), name="logits")
        zeroes = tf.Variable(tf.zeros([200]),tf.float32, name="zeroes")
        linear = tf.Variable(tf.linspace(-5.0,4.0,10),tf.float32,name="linear")
        placeholder=tf.placeholder(tf.float32,name="placeholder")
        init=tf.global_variables_initializer()
        sess.run(init)

        #########################################################################
        ##sum(t ** 2) / 2
        l2_loss=tf.nn.l2_loss(logits)
        print("l2_loss\n",sess.run(l2_loss))
        #1+4+9+16=30 ,30/2=15.0
        l2_loss=tf.nn.l2_loss(placeholder)
        print(sess.run(l2_loss,{placeholder:[[1.0,2.],[3.,4.]]}))

        #exp(logits) / reduce_sum(exp(logits), dim)
        softmax=tf.nn.softmax(logits)
        print(sess.run(softmax))

        #x / sqrt(max(sum(x**2), epsilon))
        l2_normalize=tf.nn.l2_normalize(placeholder,1)
        print("l2_normalize dim=1\n",sess.run(l2_normalize,{placeholder:[[1.0,2.],[3.,4.]]}))

        l2_normalize=tf.nn.l2_normalize(placeholder,0)
        print("l2_normalize dim=0\n",sess.run(l2_normalize,{placeholder:[[1.0,2.],[3.,4.]]}))

        #返回 最后一维数的相加结果
        biasadd=tf.nn.bias_add(tf.expand_dims( tf.reshape( linear,[-1,2]),axis=1),tf.linspace(1.,2.,2))
        print("bias_add",sess.run(biasadd))