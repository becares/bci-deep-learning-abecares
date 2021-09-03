# BCI system for motor imagery classificationusing convolutional neural networks

The  following  project  aims  to  analyze  the  ability  of  Convolutional  Neural  Networks(CNNs) to discriminate raw Electroencephalographic (EEG) signals for Brain-computerinterfaces (BCI), in order to develop a solid and reliable model that is capable of solvingthese medical and clinical applications. The project also aims to serve as foundations forfuture research projects of the UPV/EHU research groupAldapa, as well as being a start-ing framework to apply modern techniques such as Transfer Learning or Semi-supervisedLearning.To achieve this, this report collects and explains the mathematical and theoretical foun-dations of the architectures and models used for the development, based on the article of [Schirrmeister et al., 2017] and the large EEG database provided by [Kaya et al., 2018]. Following the model implementation, an experimentation is designed and tested, amongwith  an  Hyperparameter  Optimization  setup  for  the  developed  model.  Finally,  the  results show that the performance of the model depends on the subject and EEG recordingsession. It also shows that some hyperparameters influence the model, for example theoptimization algorithm, but other hyperparameters barely affect the performance of theimplementation.

References:

Kaya, M. et al. (2018) A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces. Sci Data 5, 180211. https://doi.org/10.1038/sdata.2018.211

Schirrmeister, R. T. et al. (2017) Deep Learning With Convolutional Neural Networks for EEG Decoding and Visualization. Human Brain Mapping 38:5391–5420.
