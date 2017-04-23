from PIL import Image, ImageDraw
import face_recognition
import matplotlib.pyplot as plt
import tensorflow as tf

def read_image(path,filter=None,param=None):
    image=tf.image.decode_image(tf.read_file(path), channels=3)
    if filter is None:
        pass
    if filter =='contrast':
        if param is None:
            param=0.5
        image=tf.image.adjust_contrast(image,param)
    if filter == 'saturation':
        if param is None:
            param = 0.5
        image = tf.image.adjust_saturation(image,param)
    if filter == 'hue':
        if param is None:
            param = 0.5
        image = tf.image.adjust_hue(image,param)
    if filter == 'brightness':
        if param is None:
            param = 0.5
        image = tf.image.adjust_brightness(image,param)
    if filter == 'transpose':
        tf.image.transpose_image(image)
    return image

def classify_hair(image):
    image=tf.cast(image,tf.float32)
    R,G,B=tf.split(image,3,axis=2)
    Cb =tf.squeeze( 128 - 37.797 * R / 255 - 74.203 * G / 255 + 112 * B / 255)
    Cr = tf.squeeze(128 + 112 * R / 255 - 93.768 * G / 255 - 18.214 * B / 255)
    return tf.logical_and(tf.logical_and( tf.logical_and( Cb>=105,Cb <= 141),tf.logical_and( Cr >= 105 , Cr <= 144)),  (tf.squeeze( R) < 35))

def hair_tf():
    sess = tf.InteractiveSession()
    image=read_image('../Images/me1.jpg','brightness',-0.5)
    print(classify_hair(image))

    hair=tf.cast(classify_hair(image),dtype=tf.int8)
    plt.subplot(121)
    plt.imshow(sess.run(image),cmap='Oranges')
    plt.subplot(122)
    plt.imshow(sess.run(hair))
    plt.show()

def face_recog():
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("../Images/me1.jpg")
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]

        for facial_feature in facial_features:
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)

        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], width=5)

        pil_image.save('../Images/me_face_reg.jpg')
        pil_image.show()

#face_recog()
hair_tf()