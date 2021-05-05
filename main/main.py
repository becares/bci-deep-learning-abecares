import os.path
import numpy as np
import mne
import keras.utils

import lemb
import data
import nn


# filenames = ('CLASubjectB1510193StLRHand.mat',
#              'CLASubjectB1510203StLRHand.mat',
#              'CLASubjectB1512153StLRHand.mat')
filenames = ('CLASubjectC1511263StLRHand.mat',
             'CLASubjectC1512163StLRHand.mat',
             'CLASubjectC1512233StLRHand.mat')

# filenames = ('CLASubjectB1510193StLRHand.mat', )

# Loading files only once to minimize the memory access 
def load_epochs(filenames):
    
    # 1 epoch per file
    epochs_list = []

    for filename in filenames:
        raw = lemb.load_data(os.path.join('..', 'data', filename))
        raw.drop_channels('X5')  # Sync channel
        raw.apply_function(lambda x: x * 1e6, picks=['eeg'], channel_wise=False)
        raw = raw.set_eeg_reference(ref_channels='average')
        raw.filter(0.1, 40)
        raw.apply_function(data.standarize, picks=['eeg'], channel_wise=True)
        events = mne.find_events(raw, stim_channel='STI101', output='step')

        active_events_id = {k: lemb.event_id[k]
                            for k in ('left/hand', 'right/hand')}
        epochs = mne.Epochs(raw, events,  event_id=active_events_id, baseline=None,
                            tmin=0.0, tmax=1.0, preload=True)
        epochs_list.append(epochs)
    
    return epochs_list

# Window size pointwise, [50,150]
window_size = 100

# Step size %, [10,100]
step_size = 20

# Load files
epochs_list = load_epochs(filenames)

active_crops = []
labels = []

for epochs in epochs_list:
    crops = data.crop_epochs(epochs, window_size, step_size)
    active_crops.append(crops[0])
    labels.append(crops[1])

# Train epochs, last one for test
x_train = np.concatenate(active_crops[:-1], axis=0)
y_train = np.concatenate(labels[:-1], axis=0)

permutation = np.random.permutation(len(x_train))
x_train = x_train[permutation]
y_train = y_train[permutation]
del permutation

n_crops = len(x_train)
x_train = np.expand_dims(x_train, axis=3)
y_train = keras.utils.to_categorical(y_train - 1, 2)

# Last epochs for test
x_test = np.expand_dims(active_crops[-1], axis=3)
y_test = keras.utils.to_categorical(labels[-1] - 1, 2)

print("Training data shape: {}".format(x_train.shape))
print("Test data shape: {}".format(x_test.shape))

# Logarithmic scale, 10 samples -> [10**-2, 10**-5] // np.logspace(-2, -5, 10)
learning_rate = 1e-4#

# Kernel size -> {3,5,7,9}
kernel_size = 7              

# Pool size -> {2,3}
pool_size = 2

# Learning algorithm {SGD, Adam}
optimizer = 'adam'

# Batch normalization(learning_rate=learning_rate[9], kernel_size, pool_size) {True,False}
batch_normalization = False

# Activation function {ReLU, ELU}
activation = 'relu'

# Convolutional layers {2,3,4}
#layers = 4

# Temporal filters on the first layer, 5 samples -> [10,50] // np.linspace(10,50,5).astype('int')
temporal_filters = 50

# Dropout [0,0.6]

# FC layers {0,1,2}

# 2nd layer neurons % [20,75]

# Input shape

# Batch-size 
batch_size = 32

# Epochs
epochs = 10

network = nn.conv(temporal_filters=temporal_filters,
                  kernel_size=kernel_size,
                  pool_size=pool_size,
                  learning_rate=learning_rate,
                  optimizer=optimizer,
                  activation=activation)   

network.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=100,
            verbose=1)

score = network.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
