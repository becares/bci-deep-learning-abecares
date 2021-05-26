import os.path
import numpy as np
import mne
import keras.utils
import ray
from tensorflow.keras.utils import Sequence
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

import lemb
import data
import nn


##### AUXILIARY FUNCTIONS #####

# Loading files only once to minimize the memory access 
def load_epochs():
    
    filenames = ('CLASubjectC1511263StLRHand.mat',
                 'CLASubjectC1512163StLRHand.mat',
                 'CLASubjectC1512233StLRHand.mat')
    
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

# Generate the data given the epoch list and some parameters
def generate_data(epoch_list, window_size, step_size):
    
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
    
    return {'x_train': x_train, 
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
    }

######################################## PARAMETER LIST ########################################
# Learning rate in logarithmic scale, 10 samples -> [10**-2, 10**-5] // np.logspace(-2, -5, 10)
# Kernel size {3,5,7,9}
# Pool size {2,3}
# Learning algorithm {sgd, adam}
# Batch normalization(learning_rate=learning_rate[9], kernel_size, pool_size) {True,False}
# Activation function {relu, elu}
# Convolutional layers {2,3,4}
# Temporal filters on the first layer, 5 samples -> [10,50] // np.linspace(10,50,5).astype('int')
# Dropout [0,0.6]
# FC layers {0,1,2}
# 2nd layer neurons % [20,75]
# Batch-size 
# Epochs
################################################################################################

# MODEL BUILDER #
class NNModel(tune.Trainable):
    
    def setup(self, config):
        
        self.window_size = 100
        self.step_size = 20
        
        self.kernel_size = 7
        self.pool_size = 2
        
        self.optimizer = 'adam'
        self.activation = 'relu'
        
        self.learning_rate = 1e-4
        self.temporal_filters = 50
        
        self.batch_size = 32
        self.epochs = 1
        
        self.accuracy = 0.0
    
    def step(self):
        
        data = generate_data(self.config['epochs_list'], self.window_size, self.step_size)
        
        network = nn.conv(temporal_filters=self.temporal_filters,
                  kernel_size=self.kernel_size,
                  pool_size=self.pool_size,
                  learning_rate=self.config['learning_rate'],
                  optimizer=self.optimizer,
                  activation=self.activation)
        
        network.fit(data['x_train'], 
                    data['y_train'],
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=0)
        
        #Test loss: score[0]
        #Test accuracy: score[1]
        score = network.evaluate(data['x_test'], data['y_test'], verbose=0)
        
        return {'accuracy': score[1]}
    
    def save_checkpoint(self, checkpoint_dir):
        return {
            "accuracy": self.accuracy,
            "learning_rate": self.learning_rate,
        }

    def load_checkpoint(self, checkpoint):
        self.accuracy = checkpoint["accuracy"]
        
    def reset_config(self, new_config):
        self.lr = new_config["learning_rate"]
        self.config = new_config
        return True

# MAIN #
if __name__ == "__main__":
    
    ray.init()
    
    epochs_list = load_epochs()

    pbt = PopulationBasedTraining(
          perturbation_interval=2,
          hyperparam_mutations={"learning_rate": lambda: 10**np.random.randint(-5, -2),
          })

    results = tune.run(
                NNModel,
                resources_per_trial={'gpu': 1},
                name="nn_test",
                scheduler=pbt,
                metric="accuracy",
                mode="max",
                stop={"training_iteration": 2},
                num_samples=2,
                config={
                    "epochs_list": epochs_list,
                    "learning_rate": 1e-4
                }
                )

    df = results.dataframe(metric="accuracy", mode="max")
    #print(df)
    print(results.best_config)
