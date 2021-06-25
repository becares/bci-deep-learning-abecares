import os.path
import argparse
import numpy as np
import mne
import keras.utils
import ray
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
import lemb
import data
import nn


##### SOURCES #####
# Cross-validation: https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/#code-example-k-fold-cross-validation-with-tensorflow-and-keras
#                   https://scikit-learn.org/stable/modules/cross_validation.html


##### TASKS #####


##### AUXILIARY FUNCTIONS #####

# Loading files only once to minimize the memory access 
def load_epochs(subject='C'):
    
    if subject == 'B':
        filenames = ('CLASubjectB1510193StLRHand.mat',
                     'CLASubjectB1510203StLRHand.mat',
                     'CLASubjectB1512153StLRHand.mat')
    if subject == 'C':
        filenames = ('CLASubjectC1511263StLRHand.mat',
                     'CLASubjectC1512163StLRHand.mat',
                     'CLASubjectC1512233StLRHand.mat')
    if subject == 'E':
        filenames = ('CLASubjectE1512253StLRHand.mat',
                     'CLASubjectE1601193StLRHand.mat',
                     'CLASubjectE1601193StLRHand.mat')
    if subject == 'F':
        filenames = ('CLASubjectF1509163StLRHand.mat',
                     'CLASubjectF1509173StLRHand.mat',
                     'CLASubjectF1509283StLRHand.mat')
    
    # 1 epoch per file
    epochs_list = []

    for filename in filenames:
        print(f'Current file:{filename}\n')
        raw = lemb.load_data(os.path.join('..','data',filename))
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
def generate_data(epochs_list, window_size, step_size):
    
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
        
        self.batch_size = 32
        self.epochs = 100
        
        self.learning_rate = 0.0
        self.accuracy = 0.0
    
    def step(self):
        
        window_size = int(self.config['window_size'])
        step_size = int(window_size * self.config['step_size'])
        kernel_size = int(self.config['kernel_size'] + 1)
        pool_size = int(self.config['pool_size'])
        learning_rate = self.config['learning_rate']
        temporal_filters = int(self.config['temporal_filters'])
        optimizer = self.config['optimizer']
        activation = self.config['activation']
        batch_normalization = self.config['batch_normalization']
        n_conv_layers = int(self.config['n_conv_layers'])
        n_fc_layers = int(self.config['n_fc_layers'])
        n_neurons_2nd_layer = self.config['n_neurons_2nd_layer']
        dropout_rate = self.config['dropout_rate']

        print('---------------------------------------------------------------------------------')
        print('Current configuration:')
        print(f'window_size: {window_size}, step_size: {step_size},')
        print(f'kernel_size: {kernel_size}, pool_size: {pool_size},')
        print(f'learning_rate: {learning_rate}, temporal_filters: {temporal_filters}')
        print(f'optimizer: {optimizer}, activation: {activation},')
        print(f'batch_normalization: {batch_normalization}, n_conv_layers: {n_conv_layers},')
        print(f'n_fc_layers: {n_fc_layers}, n_neurons_2nd_layer: {n_neurons_2nd_layer},')
        print(f'dropout_rate: {dropout_rate}')
        print('---------------------------------------------------------------------------------')
        
        data = generate_data(epochs_list, window_size, step_size)
        
        # Ignore last session
        #inputs = np.concatenate((data['x_train'], data['x_test']), axis=0)
        #targets = np.concatenate((data['y_train'], data['y_test']), axis=0)
        
        inputs = data['x_train']
        targets = data['y_train']
        
        callback = EarlyStopping(monitor='val_accuracy', min_delta=1e-3, mode='max', patience=5)
        
        acc_per_fold = []
        loss_per_fold = []
        
        kfold = KFold(n_splits=10, shuffle=True) # 10-Fold Cross-Validation
        
        fold = 1
        for train, test in kfold.split(inputs, targets):
        
            model = nn.conv(temporal_filters=temporal_filters,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    window_size=window_size,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    activation=activation,
                    batch_normalization=batch_normalization,
                    n_conv_layers=n_conv_layers,
                    n_fc_layers=n_fc_layers,
                    n_neurons_2nd_layer=n_neurons_2nd_layer,
                    dropout_rate=dropout_rate)
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold} ...')
            
            history = model.fit(inputs[train], 
                      targets[train],
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      callbacks=[callback],
                      validation_split=0.2,
                      verbose=2)
            
            scores = model.evaluate(inputs[test], targets[test], verbose=0)
            #print(scores)
            fold_val_loss = scores[0]
            fold_val_accuracy = scores[1]
            print('--------------------------------------------------------------------------------------------------')
            print(f'Score for fold {fold}: fold_val_loss of {fold_val_loss}; fold_val_accuracy of {fold_val_accuracy}')
            acc_per_fold.append(fold_val_accuracy)
            loss_per_fold.append(fold_val_loss)
            
            fold += 1
        
        return {'val_accuracy': np.mean(acc_per_fold)}

# MAIN #
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    args = parser.parse_args()
    
    epochs_list = load_epochs(args.subject)
    
    ray.init(num_gpus=1)
    
    #byo = BayesOptSearch()
    hys = HyperOptSearch()
    
    hys = ConcurrencyLimiter(hys, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler()
    
    name = f'nn_test_{args.subject}'
    
    results = tune.run(
                NNModel,
                resources_per_trial={'gpu': 1},
                name=name,
                scheduler=scheduler,
                search_alg=hys,
                metric="val_accuracy",
                mode="max",
                stop={"training_iteration": 1},
                num_samples=250,
                config={
                    "window_size": tune.quniform(50,100,50),
                    "step_size": tune.quniform(0.1,1,0.1),
                    "kernel_size": tune.quniform(2,8,2),
                    "pool_size": tune.quniform(2,3,1),
                    "optimizer": tune.choice(["adam","sgd"]),
                    "activation": tune.choice(["relu","elu"]),
                    "learning_rate": tune.loguniform(1e-5,1e-2),
                    "temporal_filters": tune.quniform(10,50,10),
                    "batch_normalization": tune.choice([True,False]),
                    "n_conv_layers": tune.quniform(0,1,1),
                    "n_fc_layers": tune.quniform(0,2,1),
                    "n_neurons_2nd_layer": tune.quniform(0.2,0.75,0.05),
                    "dropout_rate": tune.choice([0,0.6])
                    }
              )

    df = results.dataframe(metric="val_accuracy", mode="max")
    best = results.best_config
    
    # Last model generation    
    window_size = int(best['window_size'])
    step_size = int(window_size * best['step_size'])
    kernel_size = int(best['kernel_size'] + 1)
    pool_size = int(best['pool_size'])
    learning_rate = best['learning_rate']
    temporal_filters = int(best['temporal_filters'])
    optimizer = best['optimizer']
    activation = best['activation']
    batch_normalization = best['batch_normalization']
    n_conv_layers = int(best['n_conv_layers'])
    n_fc_layers = int(best['n_fc_layers'])
    n_neurons_2nd_layer = best['n_neurons_2nd_layer']
    dropout_rate = best['dropout_rate']
    
    print('---------------------------------------------------------------------------------')
    print('BEST CONFIGURATION MODEL:')
    print(f'window_size: {window_size}, step_size: {step_size},')
    print(f'kernel_size: {kernel_size}, pool_size: {pool_size},')
    print(f'learning_rate: {learning_rate}, temporal_filters: {temporal_filters}')
    print(f'optimizer: {optimizer}, activation: {activation},')
    print(f'batch_normalization: {batch_normalization}, n_conv_layers: {n_conv_layers},')
    print(f'n_fc_layers: {n_fc_layers}, n_neurons_2nd_layer: {n_neurons_2nd_layer},')
    print(f'dropout_rate: {dropout_rate}')
    print('---------------------------------------------------------------------------------')
    
    data = generate_data(epochs_list, window_size, step_size)
    
    model = nn.conv(temporal_filters=temporal_filters,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    window_size=window_size,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    activation=activation,
                    batch_normalization=batch_normalization,
                    n_conv_layers=n_conv_layers,
                    n_fc_layers=n_fc_layers,
                    n_neurons_2nd_layer=n_neurons_2nd_layer,
                    dropout_rate=dropout_rate)
    
    callback = EarlyStopping(monitor='accuracy', min_delta=1e-3, mode='max', patience=5)
     
    history = model.fit(data['x_train'], 
                         data['y_train'],
                         batch_size=32,
                         epochs=100,
                         #IGNORE EARLY STOPPING callbacks=[callback],
                         verbose=2)
     
    scores = model.evaluate(data['x_test'], data['y_test'], verbose=1)
     
    val_loss = scores[0]
    val_accuracy = scores[1]
     
    print(f'Final score: val_loss of {val_loss}; val_accuracy of {val_accuracy}')

