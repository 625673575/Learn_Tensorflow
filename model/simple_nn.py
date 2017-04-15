import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


def devsigmoid(x):
    return tf.multiply(x , (1 - x))


def relu(x):
    return tf.maximum(x, 0)


def devrelu(x):
    return tf.ceil(tf.clip(x, -0.1, 0.5))


def tanh(x):
    return (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))


def devtanh(x):
    return 1 - tanh(x) * tanh(x)


def basicNN(input, output):
    sess=tf.Session()
    w0 =tf.Variable((2 * tf.random_normal([3,1])) - 1,dtype=tf.float32)
    l1 = tf.Variable(tf.random_normal([5,1]), dtype=tf.float32)
    l1_delta = tf.Variable(tf.random_normal([5,1]), dtype=tf.float32)
    sess.run(tf.global_variables_initializer())

    l1 = tf.sigmoid(tf.matmul(input, w0))
    l1_delta =tf.multiply( tf.subtract(output,l1 ), devsigmoid(l1)*2)
    train=tf.assign_add( w0 ,tf.matmul(tf.transpose(input), l1_delta))
    for i in range(1000):
        sess.run(train)

    testinput = tf.constant([[1, 0, 0],
                  [0, 1, 0],
                  [1, 1, 1],
                  [0, 0, 1],
                  [1, 0, 1]],tf.float32)
    o = tf.matmul(testinput, w0)
    print("weight multipy:", sess.run(o))
    output = sigmoid(o)
    print("after simulate:",sess.run(output) )


def oneHiddenLayer(input, output):
    w0 = (2 * np.random.rand(3, 4)) - 1  # (3,4) hiddenlayer有4个w
    w1 = (2 * np.random.rand(4, 1)) - 1  # (4,1)
    for i in range(1000):
        l0 = input  # (5,3)
        l0mulweight = np.dot(l0, w0)  # (5,4)
        l1 = sigmoid(l0mulweight)

        l1mulweight = np.dot(l1, w1)  # (5,1)
        l2 = sigmoid(l1mulweight)

        l2_error = output - l2  # (5,1)
        l2_delta = l2_error * devsigmoid(l2)

        w1 += np.dot(l1.T, l2_delta)  # (4,1)

        l1_error = l2_delta.dot(w1.T)  # (5,4)
        l1_delta = l1_error * devsigmoid(l1)

        w0 += np.dot(l0.T, l1_delta)  # (3,4)

    print("w0", w0)
    print("w1", w1)
    testinput = np.array([0, 1, 0])
    o0 = np.dot(testinput, w0)
    o1 = np.dot(o0, w1)
    print("weight multipy:", o1)
    output = sigmoid(o1)
    print("after simulate:", output)


def twoHiddenLayer(input, output):  # 第一个隐层为4，第二个为2 3为input,1为output
    w0 = (2 * np.random.rand(3, 4)) - 1  # (3,4)
    w1 = (2 * np.random.rand(4, 2)) - 1  # (4,2)
    w2 = (2 * np.random.rand(2, 1)) - 1  # (2,1)

    for i in range(1000):
        l0 = input
        l0mw = np.dot(input, w0)  # (x,4)
        l1 = sigmoid(l0mw)
        l1mw = np.dot(l1, w1)  # (x,2)
        l2 = sigmoid(l1mw)
        l2mw = np.dot(l2, w2)  # (x,1)
        l3 = sigmoid(l2mw)

        l3_error = output - l3  # (x,1)
        l3_delta = l3_error * devsigmoid(l3)
        w2 += np.dot(l2.T, l3_delta)  # dot((2,x),(x,1))=(2,1)

        l2_error = l3_delta.dot(w2.T)  # (x,2)
        l2_delta = l2_error * devsigmoid(l2)
        w1 += np.dot(l1.T, l2_delta)

        l1_error = l2_delta.dot(w1.T)  # (x,4)
        l1_delta = l1_error * devsigmoid(l1)
        w0 += np.dot(l0.T, l1_delta)

    print("w0", w0)
    print("w1", w1)
    print("w2", w2)

    testinput = np.array([1, 1, 0])
    o0 = np.dot(testinput, w0)
    o1 = np.dot(o0, w1)
    o2 = np.dot(o1, w2)
    print("weight multipy:", o2)
    output = sigmoid(o2)
    print("after simulate:", output)

input = tf.constant([[1.0, 0, 0],
                  [0, 1, 0],
                  [1, 1, 1],
                  [0, 0, 1],
                  [1, 0, 1]],dtype= tf.float32)

output = tf.transpose(tf.constant([[0, 0, 1, 1, 1]],dtype=tf.float32))
# given a random weight and remap the value to (-1,1)
# basicNN(input,output)
basicNN(input, output)