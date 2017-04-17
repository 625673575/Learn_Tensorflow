import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image_file=tf.read_file('../Images/child.jpg')
image=tf.image.convert_image_dtype(tf.image.decode_jpeg(image_file,channels=3),tf.float32)
image=tf.image.resize_images(image,[224, 224],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
image_4d=tf.expand_dims(image,0)
filter=tf.Variable(tf.random_normal([4,4,3,64]))#[width,height,channels,filter_num]
conv_op=tf.nn.conv2d(image_4d,filter=filter,strides=[1,4,4,1],padding='VALID',use_cudnn_on_gpu=True)

sess = tf.InteractiveSession()

writer=tf.summary.FileWriter('/tmp/conv2d_summary/',sess.graph)

tf.global_variables_initializer().run()
conv_result=sess.run(conv_op[0])# we only have one image

for i in range (1,64):
    plt.subplot(8,8,i)
    plt.imshow(conv_result[:,:,i],cmap='gray')

for i in range(conv_result.shape[-1]):
    image_summary=tf.summary.image('filter/'+str(i),tf.expand_dims( conv_op[:,:,:,i],-1))
    writer.add_summary(sess.run(image_summary))
writer.close()
plt.show()