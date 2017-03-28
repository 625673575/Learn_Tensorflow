import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

reader = tf.WholeFileReader()
image0=tf.image.convert_image_dtype(tf.image.decode_jpeg( tf.read_file('3.jpg'),channels=3),dtype=tf.float32)
resize=tf.image.resize_images(image0,[128,128],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#image process
contrast= tf.image.adjust_contrast(resize,1)
gamma=tf.image.adjust_gamma(contrast,0.67,0.5)
hue=tf.image.adjust_hue(resize,0.1)
saturate=tf.image.adjust_saturation(resize,0.5)
brightness=tf.image.adjust_brightness(resize,100)
crop=tf.image.crop_to_bounding_box(resize,32,32,64,64)
#end process

with tf.Session() as sess:
    result=sess.run(gamma)
plt.imshow(result)
plt.show()