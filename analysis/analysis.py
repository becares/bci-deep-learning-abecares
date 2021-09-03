import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import json

def raw_data_to_csv(s,n):
    filenames = glob.glob(f'../results_27072021/subject_{s}{n}/BlackBox_*/result.json')

    window_size = []
    step_size = []
    kernel_size = []
    pool_size = []
    optimizer = []
    activation = []
    learning_rate = []
    temporal_filters = []
    batch_normalization = []
    n_conv_layers = []
    n_fc_layers = []
    n_neurons_2nd_layer = []
    dropout_rate = []

    val_loss = []
    val_accuracy = []

    date = []

    for filename in filenames:
        
        f = open(filename, 'r').read()
        if f != '':
            js = json.loads(f)
            
            val_accuracy.append(js['val_accuracy'])
            val_loss.append(js['val_loss'])
            
            date.append(js['date'])

            items = list(js['config'].values())
            
            window_size.append(items[0])
            step_size.append(round(items[1], 2))
            kernel_size.append(items[2]+1)
            pool_size.append(items[3])
            optimizer.append(items[4])
            activation.append(items[5])
            learning_rate.append(items[6])
            temporal_filters.append(items[7])
            batch_normalization.append(items[8])
            n_conv_layers.append(items[9])
            n_fc_layers.append(items[10])
            n_neurons_2nd_layer.append(round(items[11], 2))
            dropout_rate.append(items[12])

    config = {  'window_size': window_size, 'step_size': step_size, 'kernel_size': kernel_size, 'pool_size': pool_size, 'optimizer': optimizer, 
                'activation': activation, 'learning_rate': learning_rate, 'temporal_filters': temporal_filters, 'batch_normalization': batch_normalization, 
                'n_conv_layers': n_conv_layers, 'n_fc_layers': n_fc_layers, 'n_neurons_2nd_layer': n_neurons_2nd_layer, 'dropout_rate': dropout_rate,
                'val_loss': val_loss, 'val_accuracy': val_accuracy, 'date': date
            }

    df = pd.DataFrame.from_dict(config)
    df.to_csv(f'subject_{s}{n}_analysis/subject_{s}{n}_df.csv')

def boxplot_hyperparams(s,n):
    
    df = pd.read_csv(f'subject_{s}{n}_analysis/subject_{s}{n}_df.csv')

    rows = ['window_size', 'step_size', 'kernel_size', 'pool_size',
               'optimizer', 'activation', 'temporal_filters',
               'batch_normalization', 'n_conv_layers', 'n_fc_layers', 'n_neurons_2nd_layer',
               'dropout_rate'
               ]
    
    axes = df.plot.scatter('learning_rate', 'val_accuracy', s=3, c='blue')
    axes.set_xscale('log')
    axes.figure.savefig(f'subject_{s}{n}_analysis/subject_{s}{n}_learning_rate_fig.png')
    
    for row in rows:
        ax = df.boxplot(by=row, column=['val_accuracy'], grid=False)
        ax.set_xlabel(df[row].value_counts(sort=False).to_numpy())
        ax.set_ylabel('val_accuracy')
        ax.set_title('')
        ax.figure.savefig(f'subject_{s}{n}_analysis/subject_{s}{n}_{row}_fig.png')

def plot_validation_results():
    df = pd.read_csv(f'../validation/val_data_short.csv')
    
    b_data = df.loc[0:5].to_numpy()[:,1]
    c_data = df.loc[6:11].to_numpy()[:,1]
    e_data = df.loc[12:17].to_numpy()[:,1]
    f_data = df.loc[18:23].to_numpy()[:,1]
    print(b_data)
    print(c_data)
    print(e_data)
    print(f_data)
    
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(figsize=(1024*px, 640*px))

    plt.plot(['10%','15%','25%','50%','75%','100%'], b_data, color='darkred')
    plt.plot(['10%','15%','25%','50%','75%','100%'], c_data, color='darkgreen')
    plt.plot(['10%','15%','25%','50%','75%','100%'], e_data, color='darkblue')
    plt.plot(['10%','15%','25%','50%','75%','100%'], f_data, color='gold')

    plt.legend(['Subject B', 'Subject C', 'Subject E', 'Subject F'])

    plt.xlabel('Data crop used in the experiment')
    plt.ylabel('val_accuracy')

    plt.title('Survey validation accuracy results for all subjects and data crops.')

    plt.show()

def plot_tensorflow_like_graph(s,n):
    df = pd.read_csv(f'subject_{s}{n}_analysis/subject_{s}{n}_df.csv')
    arr = df.sort_values(by='date')['val_accuracy'].to_numpy()
    
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(figsize=(1024*px, 640*px))

    plt.scatter(range(len(arr)), arr)
    plt.xlabel('Iteration')
    plt.ylabel('val_accuracy')
    plt.title('Scatterplot of all accuracies obtained during the optimization phase.')

    plt.savefig(f'subject_{s}{n}_analysis/subject_{s}{n}_scatter.png')


if __name__ == '__main__':
    for s in ['B','C','E','F']:
        for n in [25,50,75,100]:
            plot_tensorflow_like_graph(s,n)
