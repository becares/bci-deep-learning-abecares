import os.path
import os
import main
import lemb
import mne
import data
import nn
import numpy as np
import argparse
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from itertools import zip_longest
from collections import Counter

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def new_validation(y_pred, y_test, crops_per_epoch):
    
    assert (crops_per_epoch % 2 != 0)
    acc = 0
    y_pred_groups = grouper(y_pred, crops_per_epoch)
    y_test_groups = grouper(y_test, crops_per_epoch)
    y_pred_groups = [y[0] for y in y_pred_groups]
    y_test_groups = [y[0] for y in y_test_groups]
    
    for i, pred in enumerate(y_pred_groups):
        test = y_test_groups[i]
        max_pred, _ = Counter(pred).most_common(1)[0]
        max_test, _ = Counter(test).most_common(1)[0]
        if max_pred == max_test:
            acc += 1
    
    return acc / len(y_test_groups)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('size')
    args = parser.parse_args()
    
    cwd = os.getcwd()

    size = int(args.size)
    subject = args.subject
    
    name = f'subject_{subject}{size}'

    epochs_list = main.load_epochs(args.subject, size)

    best_B = {'window_size': 50.0, 'step_size': 0.1, 'kernel_size': 6.0, 'pool_size': 2.0, 'optimizer': 'adam', 'activation': 'relu', 'learning_rate': 0.0022275208884486608, 'temporal_filters': 40.0, 'batch_normalization': True, 'n_conv_layers': 1.0, 'n_fc_layers': 1.0, 'n_neurons_2nd_layer': 0.4, 'dropout_rate': 0.6}
    best_C = {'window_size': 100.0, 'step_size': 0.6000000000000001, 'kernel_size': 4.0, 'pool_size': 3.0, 'optimizer': 'adam', 'activation': 'elu', 'learning_rate': 0.0006421923877743608, 'temporal_filters': 40.0, 'batch_normalization': False, 'n_conv_layers': 0.0, 'n_fc_layers': 2.0, 'n_neurons_2nd_layer': 0.25, 'dropout_rate': 0}
    best_E = {'window_size': 100.0, 'step_size': 0.1, 'kernel_size': 8.0, 'pool_size': 2.0, 'optimizer': 'adam', 'activation': 'relu', 'learning_rate': 0.001886233015685169, 'temporal_filters': 50.0, 'batch_normalization': False, 'n_conv_layers': 1.0, 'n_fc_layers': 0.0, 'n_neurons_2nd_layer': 0.30000000000000004, 'dropout_rate': 0.6}
    best_F = {'window_size': 100.0, 'step_size': 0.1, 'kernel_size': 8.0, 'pool_size': 3.0, 'optimizer': 'adam', 'activation': 'relu', 'learning_rate': 0.0003200077962021841, 'temporal_filters': 50.0, 'batch_normalization': False, 'n_conv_layers': 1.0, 'n_fc_layers': 2.0, 'n_neurons_2nd_layer': 0.7000000000000001, 'dropout_rate': 0.6}

    if subject=='B':
        best = best_B
    if subject=='C':
        best = best_C
    if subject=='E':
        best = best_E
    if subject=='F':
        best = best_F

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

    data = main.generate_data(epochs_list, window_size, step_size)
    
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

    early_stopping = EarlyStopping(monitor='accuracy', min_delta=1e-3, mode='max', patience=5)
    old_val_accs = []
    new_val_accs = []
        
    for it in range(10):
        
        csv_logger = CSVLogger(fr'{cwd}/{name}_final_model_history_{it}.csv')
        
        history = model.fit(data['x_train'], 
                            data['y_train'],
                            batch_size=32,
                            epochs=100,
                            validation_split=0.2,
                            callbacks=[early_stopping, csv_logger],
                            verbose=2)
            
        old_val_accuracy = model.evaluate(data['x_test'], data['y_test'], verbose=1)[1]

        predictions = model.predict(data['x_test'])
        y_pred = (predictions > 0.5)
        y_test = (data['y_test'] > 0.5)
        
        new_val_accuracy = new_validation(y_pred, y_test, data['crops_per_epoch'])
        
        matrix = confusion_matrix(data['y_test'].argmax(axis=1), y_pred.argmax(axis=1))
        np.savetxt(fr'{cwd}/{name}_confusion_matrix_{it}.csv', matrix, delimiter=',')
        
        old_val_accs.append(old_val_accuracy)
        new_val_accs.append(new_val_accuracy)
    
    old_val_acc_mean = np.mean(old_val_accs)
    old_val_acc_std = np.std(old_val_accs)
    
    new_val_acc_mean = np.mean(new_val_accs)
    new_val_acc_std = np.std(new_val_accs)
    
    print(f'old_val_acc_std: {old_val_acc_std}, new_val_acc_std: {new_val_acc_std}')
    print(f'old_val_acc_mean: {old_val_acc_mean}, new_val_acc_mean: {new_val_acc_mean}')
    
    
        
