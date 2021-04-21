# Proyecto BCI-Deep Learning Adrián Bécares

El objetivo principal del proyecto es diseñar una red profunda para la clasificación de señales EEG.

Características:
* La base de datos con señales EEG utilizada en la experimentación será la descrita en Kaya et al. (2018) salvo que un análisis previo descubra una base de datos más adecuada. Los requerimientos buscados son:
  * Gran número de trials registrados por sujeto.
  * Gran cantidad de señales relacionadas con reposo y/o señales sin etiquetar.
    * Pocas clases a discernir.
    * No es necesario disponer de una gran cantidad de sujetos.
* La base datos mencionada tiene distintos apartados con distintas formas de adquirir las señales EEG. Se deberá decidir los apartados a utilizar y se repetirá la experimentación con varios sujetos.
* La arquitectura de la red profunda se basará en la red profunda propuesta en Schirrmeister et al. (2017) salvo que un análisis previo descubra una arquitectura más adecuada. Los requerimientos buscados son:
  * Una red descrita en un artículo actual, riguroso y bien explicado.
  * Una red de complejidad media, evitando características muy poco usadas en el contexto.
  * Una red cuya implementación esté disponible (preferiblemente en Python).
* El experimento deberá realizarse sobre muestras de entrenamiento de diferente tamaño para evaluar cómo afecta esta característica al resultado obtenido. Deberá diseñarse un procedimiento que utilice submuestras de la base de datos completa. En relación a esto deberá decidirse el método de validación a utilizar (probablemente validación cruzada).
* Se definirá un procedimiento sencillo de early-stopping para finalizar el proceso de entrenamiento de la red.
* Se utilizará un algoritmo de optimización bayesiana para establecer los parámetros más adecuados en cada caso. Los parámetros a optimizar serán los siguientes (se mencionan los rangos aproximados en los que se probarán los parámetros):
  * Tasa de aprendizaje: [10-2, 10-5] escala logarítmica
  * Tamaño del kernel {3, 5, 7, 9} y max pooling {2, 3}
  * Tamaño de ventana (puntos) [50, 150] y desplazamiento (%) [10, 100].
  * Algoritmo de aprendizaje: {SGD, Adam}
  * Batch normalization: {Sí, No}
  * Función de activación: {ReLU, ELU}
  * Número de capas convolucionales: {2, 3, 4}
  * Número de filtros de la primera capa (multiplicar por el pooling a partir de ahí): [10, 50]
  * Dropout: [0, 0,6]
  * Número de capas FC: {0, 1, 2}
  * Número de neuronas en la segunda capa (porcentaje de la anterior): [20, 75]

Referencias:

Kaya, M. et al. (2018) A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces. Sci Data 5, 180211. https://doi.org/10.1038/sdata.2018.211

Schirrmeister, R. T. et al. (2017) Deep Learning With Convolutional Neural Networks for EEG Decoding and Visualization. Human Brain Mapping 38:5391–5420.
