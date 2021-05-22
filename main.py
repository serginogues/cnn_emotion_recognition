"""
Emotion Recognition with a CNN
"""
from cnn import train, test, visualize_filter, model

TRAIN = True
if __name__ == '__main__':

    visualize_filter(model.cpu())

    if TRAIN:
        train()
    else:
        test()
