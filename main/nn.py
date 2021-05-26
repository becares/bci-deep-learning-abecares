import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling1D, Reshape
from keras.models import Sequential


def conv(temporal_filters=50, kernel_size=7, pool_size=2, input_shape=(21,100,1), 
         num_classes=2, learning_rate=1e-4, optimizer='sgd', activation='relu'):

    model = Sequential()
    
    model.add(Conv2D(temporal_filters, kernel_size=(1, kernel_size), input_shape=input_shape))
    model.add(Conv2D(temporal_filters, (input_shape[0], 1), activation=activation))
    model.add(Reshape((-1, temporal_filters)))
    model.add(MaxPooling1D(pool_size=pool_size))
    
    model.add(Conv1D(pool_size * temporal_filters, kernel_size=kernel_size, activation=activation))
    model.add(MaxPooling1D(pool_size=pool_size))
    
    model.add(Conv1D(pool_size * pool_size * temporal_filters, kernel_size=kernel_size, activation=activation))
    model.add(MaxPooling1D(pool_size=pool_size))
    
    model.add(Flatten())
    
    model.add(Dense(500, activation=activation))
    model.add(Dense(100, activation=activation))
    
    model.add(Dense(num_classes, activation='softmax'))

    if optimizer=='adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(lr=learning_rate)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
    #model.summary()
    return model
