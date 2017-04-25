import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
FLAGS = None

def server():
    # In task 0:
    cluster = tf.train.ClusterSpec({"ps": ["192.168.0.3:2222", "192.168.0.4:2222"]})
    server = tf.train.Server(cluster, job_name="ps", task_index=1)
    server.join()

def client():

    with tf.Session("grpc://192.168.0.4:2222") as sess:
        with tf.device("/job:ps/task:0"):
            input = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [1, 1, 0],
                              [1, 0, 1],
                              [0, 1, 1],
                              [1, 1, 1]], dtype=np.float32)
            output = np.sum(input * [1, 8, 4], axis=1) + 2
            print(input.shape, output)

            x = tf.placeholder(tf.float32, name='x')
            y = tf.placeholder(tf.float32, name='y')
            w = tf.Variable([0.4, -0.5, 2.4])
            b = tf.Variable(0.427384)

            model = tf.reduce_sum(w * x, axis=1) + b
            loss = tf.nn.l2_loss(model - y)

            train = tf.train.AdamOptimizer(0.1).minimize(loss)
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                sess.run(train, {x: input, y: output})
            print(sess.run([w, b, model], {x: [[1, 1, 1]]}))

client()