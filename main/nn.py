import keras
from keras.layers import Dense, Flatten, Conv2D, Conv1D, MaxPooling1D
from keras.layers import Reshape, Activation, BatchNormalization, Dropout
from keras.models import Sequential


def conv(temporal_filters=50, kernel_size=7, pool_size=2, window_size=100, 
         num_classes=2, learning_rate=1e-4, optimizer='sgd', activation='relu',
         batch_normalization=False, n_conv_layers=2, n_fc_layers=2, n_neurons_2nd_layer=0.5,
         dropout_rate=0):

    input_shape = (21, window_size, 1)
    model = Sequential()
    
    if kernel_size < window_size:
        kernel_size = 1
    model.add(Conv2D(temporal_filters, kernel_size=(1, kernel_size), input_shape=input_shape))
    
    model.add(Conv2D(temporal_filters, (input_shape[0], 1)))
    if batch_normalization:
        model.add(BatchNormalization())
    model.add(Activation(activation=activation))
    model.add(Reshape((-1, temporal_filters)))
    model.add(MaxPooling1D(pool_size=pool_size))
    
    for _ in range(n_conv_layers+1):
        temporal_filters *= pool_size
        model.add(Conv1D(temporal_filters, kernel_size=kernel_size))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(MaxPooling1D(pool_size=pool_size))
    
    """# Ir añadiendo capas con un for dependiendo del parametro n_conv_layers (1,2)
    model.add(Conv1D(pool_size * temporal_filters, kernel_size=kernel_size, activation=activation))
    #Añadir aqui batch normalization, acordarse de quitar activation y añadir otra capa de activation
    model.add(MaxPooling1D(pool_size=pool_size))
    
    # Cuidado porque se multiplica más de una vez, mirar como hacer esto dependiendo de cuantas capas hay
    model.add(Conv1D(pool_size * pool_size * temporal_filters, kernel_size=kernel_size, activation=activation))
    #Añadir aqui batch normalization, acordarse de quitar activation y añadir otra capa de activation
    model.add(MaxPooling1D(pool_size=pool_size))"""
    
    model.add(Flatten())
    n_neurons = model.output_shape[1]
    # Mirar cuantas neuronas hay despues de hacer el flatten, y despues añadir n_fc_layers (0,1,2) con otro for.
    # Las capas tienen layer_percent (20,75)% de la anterior, por ejemplo si flatten tiene 1000 y el % es 50, la primera fc tiene 500 y la 2da 250
    
    for _ in range(n_fc_layers+1):
        n_neurons = int(n_neurons * n_neurons_2nd_layer) 
        model.add(Dense(n_neurons))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dropout(dropout_rate))
    
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
