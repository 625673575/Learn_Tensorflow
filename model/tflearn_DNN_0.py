import numpy as np
import tflearn
import sys
# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

class MonitorCallback(tflearn.callbacks.Callback):

    def on_epoch_end(self, training_state):
        print({
            'accuracy': training_state.global_acc,
            'loss': training_state.global_loss,
        })

    def on_batch_end(self, training_state, snapshot=False):
        print({
            'step': training_state.step,
            'loss': training_state.global_loss,
        })

# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.expand_dims(np.array(data, dtype=np.float32),axis=2)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)

# Build neural network
net = tflearn.input_data(shape=[None, 6,1])
net=tflearn.conv_1d(net, nb_filter=4,filter_size=3,strides=2,padding='same')
print( net)
net = tflearn.fully_connected(net, 32,activation="relu6")
net = tflearn.fully_connected(net, 32,activation="relu6")
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
#model=tflearn.DNN.load(net,"model.ckpt")
# Start training (apply gradient descent algorithm)
callback=MonitorCallback()
model.fit(data, labels, n_epoch=10, batch_size=64, show_metric=True,callbacks=[callback])
model.save("model.ckpt")
# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])