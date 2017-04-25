
import argparse
import sys

import tensorflow as tf

FLAGS = None

def main():
    # In task 0:
    cluster = tf.train.ClusterSpec({"local": ["http://192.168.0.3:2223", "http://192.168.0.4:2223"]})
    server = tf.train.Server(cluster, job_name="local", task_index=0)
    with tf.device("/job:ps/task:0"):
        weights_1 = tf.Variable(tf.constant( [1,2,3,4],tf.float32))
        biases_1 = tf.Variable(tf.constant( [1,2,3,4],tf.float32))

    with tf.device("/job:ps/task:1"):
        weights_2 = tf.Variable(tf.constant( [5,6,7,8],tf.float32))
        biases_2 = tf.Variable(tf.constant( [5,6,7,8],tf.float32))
        layer_1 = tf.multiply(biases_1, weights_1)
        logits = tf.multiply(biases_2, weights_2)

    sess = tf.Session(server.target)
    print(sess.run(logits))

main()