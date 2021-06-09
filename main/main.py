import os.path
import numpy as np
import mne
import keras.utils
import ray
from tensorflow.keras.utils import Sequence
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

##### CREDITS #####
# Cross-validation: https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/#code-example-k-fold-cross-validation-with-tensorflow-and-keras
# https://scikit-learn.org/stable/modules/cross_validation.html


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
        
        self.batch_size = 32
        self.epochs = 1
        
        self.learning_rate = 0.0
        self.accuracy = 0.0
    
    def step(self):
        
        window_size = int(self.config['window_size'])
        step_size = int(self.config['step_size'])
        kernel_size = int(self.config['kernel_size'] + 1)
        pool_size = int(self.config['pool_size'])
        learning_rate = self.config['learning_rate']
        temporal_filters = int(self.config['temporal_filters'])
        
        print('------------------------------------------------------------------------')
        print(f'Current configuration: window_size: {window_size}, step_size: {step_size}')
        print(f'kernel_size: {kernel_size}, pool_size: {pool_size}')
        print(f'learning_rate: {learning_rate}, temporal_filters: {temporal_filters}')
        print('------------------------------------------------------------------------')
        
        data = generate_data(self.config['epochs_list'], window_size, step_size)
        
        inputs = np.concatenate((data['x_train'], data['x_test']), axis=0)
        targets = np.concatenate((data['y_train'], data['y_test']), axis=0)
        
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
                    optimizer=self.config['optimizer'],
                    activation=self.config['activation'])
            
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold} ...')
            
            model.fit(inputs[train], 
                      targets[train],
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=0)
            
            #Test loss: score[0]
            #Test accuracy: score[1]
            scores = model.evaluate(inputs[test], targets[test], verbose=0)
            print(f'Score for fold {fold}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
            acc_per_fold.append(scores[1])
            loss_per_fold.append(scores[0])
            
            fold += 1
        
        return {'accuracy': np.mean(acc_per_fold)}
    
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
          hyperparam_mutations={"window_size": [25,50,100],
                                "step_size": lambda: np.arange(10,100,10),
                                "kernel_size": [3,5,7,9],
                                "pool_size": [2,3],
                                "learning_rate": lambda: 10**np.random.randint(-5, -2),
                                "temporal_filters": lambda: np.linspace(10,50,5).astype('int')
          })
    
    byo = BayesOptSearch()
    hys = HyperOptSearch()
    
    hys = ConcurrencyLimiter(hys, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler()
    
    results = tune.run(
                NNModel,
                resources_per_trial={'gpu': 1},
                name="nn_test",
                scheduler=scheduler,
                search_alg=hys,
                metric="accuracy",
                mode="max",
                stop={"training_iteration": 2},
                num_samples=3,
                config={
                    "epochs_list": epochs_list,
                    "window_size": tune.quniform(10,100,1),
                    "step_size": tune.quniform(10,100,1),
                    "kernel_size": tune.quniform(2,8,2),
                    "pool_size": tune.quniform(2,3,1),
                    "optimizer": tune.choice(["adam","sgd"]),
                    "activation": tune.choice(["relu","elu"]),
                    "learning_rate": tune.loguniform(1e-5,1e-2),
                    "temporal_filters": tune.quniform(10,50,10)
                    }
              )

    df = results.dataframe(metric="accuracy", mode="max")
    #print(df)
    print(results.best_config)
