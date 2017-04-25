from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
import model.speech_data as speech_data
import librosa
import os.path as path
import tensorflow as tf
def load_feature(wav):
    if not wav.endswith(".wav"): return
    wave, sr = librosa.load(wav, mono=True)
    wav= path.basename(wav)
    label = speech_data.dense_to_one_hot(int(wav[0]), 10)
    mfcc = librosa.feature.mfcc(wave, sr)
    mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)
    print(np.array(mfcc).shape)
    return np.array(mfcc),label

predict_feature,predict_label=load_feature("./data/spoken_numbers_pcm/1_Victoria_200.wav")
learning_rate = 0.001
training_iters = 300  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits
#
batch = word_batch = speech_data.mfcc_batch_generator(batch_size)

X, Y = next(batch)
print(X[0].shape,Y[0].shape)
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now
print('trainy', trainY[20])
# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
# for i in range(30):
#   model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
#           batch_size=batch_size)
# model.save("tflearn.lstm.model")

model.load("tflearn.lstm.model")
_y=model.predict(np.expand_dims( predict_feature,0))
print( predict_label)
print (_y[0])