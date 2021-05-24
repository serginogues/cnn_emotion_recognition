### COMPUTER VISION ASSIGNMENT 2: Deep Learning based on CCNs ###
Simone Grassi I6263794 - Sergi Nogues Ferres I6267073

# config.py 
It provides all the foundamental imports and the constants used in the code. 
- SET_SEED allows to change the stochastic operations (original 11).
If a GPU is avaliable, the code will choose it as default running device.
- FER_PATH is the Fer2013 dataset position
- SAVE_PATH is the automatic saving path for a new trained architecture
- NUM_EPOCHS number of training epochs (original 100)
- BATCH_SIZE dimension of a batch (original 25)
- NUM_LABELS number of output classes (original 7)
- MODEL_NAME network architecture we want to use, possible values are 'letnet5', 'simple', 'deep'. (original 'simple')

# cnn.py
It provides all the functions for creating a new network instance, according to config/MODEL_NAME, training it, printing the architecture, and evaluating the results. It is possible to choose the learning rate LR (original 0.001) and the optimizer function commenting/decommenting from the line 30.

# architectures.py
It contains all the neural network architectures used in our experiments

# preprocess.py
It groups all the functions for loading the dataset and creates a training an a testing torch dataloader. It is directly executed by cnn.py.

# HOW TO USE THIS CODE -> main.py
To run an experiment is required to set the running parameters in config.py and cnn.py, then from the file main.py is possible to run the following functions:
- visualize_filter(model.cpu()) to visualize the learned kernels
- print_architecture(model.cpu()) to print a summary of the architecture used
After running these function, don't forget to bring the model back to device with model.to(device), otherwise the following train and evaluation will raise an error.
- train() to train the model
- test() to test the model

Originaly all these functions are called in the above order.









