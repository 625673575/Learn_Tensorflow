# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:36:33 2017

@author: Liu
"""

import tensorflow as tf
import numpy as np
def model(features,labels,mode):
    with tf.device("/cpu:0"):
        W=tf.get_variable("W",[1],dtype=tf.float64)
        b=tf.get_variable("b",[1],dtype=tf.float64)
        y=W*features['x']+b
        loss=tf.reduce_mean(tf.square(y-labels))
        global_step=tf.train.get_global_step()
        optimizer=tf.train.GradientDescentOptimizer(0.01)
        train=tf.group(optimizer.minimize(loss),tf.assign_add(global_step,1))
        return tf.contrib.learn.ModelFnOps(mode=mode,predictions=y,loss=loss,train_op=train)

features=[tf.contrib.layers.real_valued_column("x",dimension=1)]
#estimator=tf.contrib.learn.LinearRegressor(feature_columns=features)
estimator=tf.contrib.learn.Estimator(model_fn=model)

x=np.array([1.,2.,3.,4.])
y=np.array([-1.,-2.,-3.,-4.])
input_fn=tf.contrib.learn.io.numpy_input_fn({"x":x},y,batch_size=4,num_epochs=1000)
estimator.fit(input_fn=input_fn, steps=1000)
evaluation=estimator.evaluate(input_fn=input_fn)
print(evaluation)