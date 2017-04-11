import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

sess = tf.Session()
reader = tf.WholeFileReader()
image0 = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file('../Images/5.jpg'), channels=3),
                                      dtype=tf.float32)
image1 = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file('../Images/3.jpg'), channels=3),
                                      dtype=tf.float32)
mask0 = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file('../Images/mask_0.jpg'), channels=1),
                                      dtype=tf.float32)
resize0 = tf.image.resize_images(image0, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
resize1 = tf.image.resize_images(image1, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
resize2 = tf.image.resize_images(mask0, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# image process
contrast = tf.image.adjust_contrast(resize0, 1)
gamma = tf.image.adjust_gamma(contrast, 0.67, 0.5)
hue = tf.image.adjust_hue(resize0, 0.1)
saturate = tf.image.adjust_saturation(resize0, 0.5)
brightness = tf.image.adjust_brightness(resize0, 100)
crop = tf.image.crop_to_bounding_box(resize0, 32, 32, 64, 64)


# end process
def desaturate(image, rw, gw, bw):
    r, g, b = tf.split(image, num_or_size_splits=3, axis=2)
    rx = r * rw + g * gw + b * bw
    image = tf.concat([rx, rx, rx], 2)
    return image


def shift_pos(image, axis, stride):
    # contrast=tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
    # nparray=session.run(image)
    # nparray= np.roll(nparray,stride,axis)
    # return tf.convert_to_tensor(nparray)
    shape = tf.shape(image)
    if axis == 0:
        if stride > 0:
            slice = tf.slice(image, [stride, 0, 0], [-1, -1, -1])
            zeros = tf.zeros([stride, shape[1], shape[2]])
            stack = tf.concat([slice, zeros], axis=0)
            return stack
        else:
            stride = -stride
            slice = tf.slice(image, [0, 0, 0], [shape[0] - stride, -1, -1])
            zeros = tf.zeros([stride, shape[1], shape[2]])
            stack = tf.concat([zeros, slice], axis=0)
            return stack
    else:
        if stride > 0:
            slice = tf.slice(image, [0, stride, 0], [-1, -1, -1])
            zeros = tf.zeros([shape[0], stride, shape[2]])
            stack = tf.concat([slice, zeros], axis=1)
            return stack
        else:
            stride = -stride
            slice = tf.slice(image, [0, 0, 0], [-1, shape[1] - stride, -1])
            zeros = tf.zeros([shape[0], stride, shape[2]])
            stack = tf.concat([zeros, slice], axis=1)
            return stack


def blur_iter(image, stride, distribution):
    col = tf.Variable(tf.zeros(shape=tf.shape(image)), name='color')
    sess.run(tf.global_variables_initializer())
    l = len(distribution)
    for i in range(l):
        index = (int)(i - (l - 1) / 2)
        col += shift_pos(image, 0, stride * index) * distribution[i]  # y方向上的模糊
        col += shift_pos(image, 1, stride * index) * distribution[i]  # x方向上的模糊
    return col * 0.5


def blur(image, stride, distribution, iter=1):
    for i in range(iter):
        image = blur_iter(image, stride, distribution)
    return image


def blur_test():
    ax1 = plt.subplot(121)
    plt.imshow(sess.run(resize0))
    result1 = blur(resize0, 1, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], 1)
    result1 = blur(result1, 2, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], 2)
    result1 = blur(result1, 3, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], 1)
    result1 = blur(result1, 2, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], 2)
    result1 = blur(result1, 1, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], 2)
    result1 = sess.run(blur(result1, 1, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05], ))
    ax1 = plt.subplot(122)
    plt.imshow(result1)
    plt.show()
def desaturate_test():
    result = sess.run(desaturate(resize0, 0.3, 0.5, 0.2))
    plt.imshow(result)
    plt.show()

def blend(image0,image1 ,alpha):
    return image0*alpha+image1*(1-alpha)

def blend_test():
    result=sess.run(blend(resize0, resize1, resize2))
    ax1 = plt.subplot(121)
    plt.imshow(result)
    ax1 = plt.subplot(122)
    result = sess.run(blend(resize0, resize1, tf.constant(0.8)))
    plt.imshow(result)
    plt.show()

blend_test()
#blur_test()