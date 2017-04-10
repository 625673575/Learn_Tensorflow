import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

sess = tf.Session()
reader = tf.WholeFileReader()
image0 = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file('../Images/4.jpg'), channels=3),
                                      dtype=tf.float32)
resize = tf.image.resize_images(image0, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# image process
contrast = tf.image.adjust_contrast(resize, 1)
gamma = tf.image.adjust_gamma(contrast, 0.67, 0.5)
hue = tf.image.adjust_hue(resize, 0.1)
saturate = tf.image.adjust_saturation(resize, 0.5)
brightness = tf.image.adjust_brightness(resize, 100)
crop = tf.image.crop_to_bounding_box(resize, 32, 32, 64, 64)


# end process
# 产生黑白图片效果
def desaturate(image, rw, gw, bw):
    r, g, b = tf.split(image, num_or_size_splits=3, axis=2)
    rx = r * rw + g * gw + b * bw
    image = tf.concat([rx, rx, rx], 2)
    return image


# 高斯模糊
def shift_pos(image, axis, stride):
    # contrast=tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
    # nparray=session.run(image)
    # nparray= np.roll(nparray,stride,axis)
    # return tf.convert_to_tensor(nparray)
    shape = tf.shape(image)
    print(sess.run(shape))
    if axis == 0:
        if stride >0:
            slice=tf.slice(image, [stride, 0, 0], [-1, -1, -1])
            zeros=tf.zeros([stride,shape[1],shape[2]])
            stack =tf.concat([slice,zeros],axis=0 )

            print('stride>0',sess.run(tf.shape(stack)))
            return stack
        else:
            stride=-stride
            slice=tf.slice(image,[0,0,0],[shape[0]-stride,-1,-1])
            zeros=tf.zeros([stride,shape[1],shape[2]])
            stack =tf.concat([zeros,slice],axis=0 )
            print('stride<0',sess.run(tf.shape(stack)))
            return stack
    else:
        if stride >0:
            slice=tf.slice(image, [0, stride, 0], [-1, -1, -1])
            zeros=tf.zeros([shape[0],stride,shape[2]])
            stack =tf.concat([slice,zeros],axis=1 )

            print('stride>0',sess.run(tf.shape(stack)))
            return stack
        else:
            stride=-stride
            slice=tf.slice(image,[0,0,0],[-1,shape[1]-stride,-1])
            zeros=tf.zeros([shape[0],stride,shape[2]])
            stack =tf.concat([zeros,slice],axis=1 )
            print('stride<0',sess.run(tf.shape(stack)))
            return stack


def blur_iter(image, stride, distribution):
    col = tf.Variable(tf.zeros(shape=tf.shape(image)), name='color')
    sess.run(tf.global_variables_initializer())
    l = len(distribution)
    for i in range(l):
        index = (int)(i - (l - 1) / 2)
        col += shift_pos(image, 0, stride * index) * distribution[i]#y方向上的模糊
        col += shift_pos(image, 1, stride * index) * distribution[i]#x方向上的模糊
    return col * 0.5


def blur(image, stride, distribution, iter=1):
    for i in range(iter):
        image = blur_iter(image, stride, distribution)
    return image
shift_pos(resize,0,10)

result = sess.run(desaturate(resize, 0.4, 0.4, 0.2))
plt.imshow(sess.run(resize))
plt.show()
result1 = sess.run(blur(resize, 1, [0.03, 0.07, 0.2, 0.4, 0.2, 0.07, 0.03], 5))
plt.imshow(result1)
plt.show()
