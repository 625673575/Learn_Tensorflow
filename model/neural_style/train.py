import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import utils
import net_factory

import yaml
import argparse

image_dir='D:/CommonCode/Learn_Tensorflow/Images/'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/mosaic.yml', help='the path to the conf file')
    return parser.parse_args()

def main(FLAGS):
    print(FLAGS.naming)
    sess = tf.Session()

    #images = tf.placeholder(tf.float32, [None,256, 256, 3])
    variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
    print([v.name for v in variables_to_restore ])
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint("pretrained/vgg_16.ckpt", variables_to_restore)
    sess.run(init_assign_op, init_feed_dict)

    image=utils.read_image(image_dir+ '5.jpg',256,256)
    images=tf.reshape(image,[1,256,256,3])
    predictions ,end_points= nets.vgg.vgg_16(image)
    sess.run(predictions.predict())
    plt.imshow(sess.run(image))
    sess.run(tf.global_variables_initializer())
    plt.show()
    # log_predictions,end_points_dict = utils.get_vgg16_network(images=tf.expand_dims(image,dim=0) ,is_training=True)
    # print(sess.run(log_predictions) ,'\n',end_points_dict)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    main(FLAGS)