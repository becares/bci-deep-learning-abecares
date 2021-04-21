import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling1D, Reshape
from keras.models import Sequential
# Comentario
# Comentario de prueba

def conv():
    input_shape = (21, 100, 1)
    num_classes = 2
    temporal_filters = 50

    model = Sequential()
    model.add(Conv2D(temporal_filters, kernel_size=(1, 7), strides=(1, 1),
                     input_shape=input_shape))
    model.add(Conv2D(temporal_filters, (input_shape[0], 1), activation='relu'))
    model.add(Reshape((-1, temporal_filters)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(2 * temporal_filters, 7, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(4 * temporal_filters, 7, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=1e-4),
                  metrics=['accuracy'])
    model.summary()
    return model
