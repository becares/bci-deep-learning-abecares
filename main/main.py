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
active_crops = []
labels = []
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
    crops = data.crop_epochs(epochs, window_size=100, step_size=20)
    active_crops.append(crops[0])
    labels.append(crops[1])
x_train = np.concatenate(active_crops[:-1], axis=0)
y_train = np.concatenate(labels[:-1], axis=0)

# exp_events = [v for k, v in lemb.event_id.items() if k.startswith('exp/')]
# passive_crops = data.crop_passive(raw.pick('eeg'), events,
#                                   window_size=100, step_size=20,
#                                   exclude=exp_events, activity_duration=200,
#                                   passive_id=lemb.event_id['passive'],
#                                   ignore_longers=(1000, 700))

permutation = np.random.permutation(len(x_train))
x_train = x_train[permutation]
y_train = y_train[permutation]
del permutation

n_crops = len(x_train)
x_train = np.expand_dims(x_train, axis=3)
y_train = keras.utils.to_categorical(y_train - 1, 2)
x_test = np.expand_dims(active_crops[-1], axis=3)
y_test = keras.utils.to_categorical(labels[-1] - 1, 2)

print("Training data shape: {}".format(x_train.shape))
print("Test data shape: {}".format(x_test.shape))

# import pdb;pdb.set_trace()
network = nn.conv()
network.fit(x_train, y_train,
            batch_size=32,
            epochs=100,
            verbose=1)

score = network.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
