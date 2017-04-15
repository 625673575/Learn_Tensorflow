import tensorflow as tf
import yaml
import functools
import os
from  tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim


def _get_init_fn(FLAGS):
    """
    This function is copied from TF slim.

    Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    tf.logging.info('Use pretrained model %s' % FLAGS.loss_model_file)

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)


class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS

def get_vgg16_network(images,is_training,weight_decay=0.0):
    return vgg.vgg_16(images, num_classes=200, is_training=is_training, spatial_squeeze=False)

def read_image(path,width=128,height=128,channels=3,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    image=None
    if(str.endswith(path,'.jpg')):
        image=tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(path), channels=channels),
                                 dtype=tf.float32)
    if (str.endswith(path,'.png')):
        image=tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(path), channels=channels),
                                 dtype=tf.float32)
    resize= tf.image.resize_images(image, [width, height], method=method)
    return resize
if __name__ == '__main__':
    f = read_conf_file('conf/mosaic.yml')
    print(f.loss_model_file)
