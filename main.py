"""
Emotion Recognition with a CNN
"""
from cnn import train, test

TRAIN = True
if __name__ == '__main__':

    if TRAIN:
        train()
    else:
        test()
