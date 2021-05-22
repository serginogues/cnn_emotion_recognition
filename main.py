"""
Emotion Recognition with a CNN
"""
from cnn import train, test, visualize_filter, model, print_architecture
from config import *

if __name__ == '__main__':

    visualize_filter(model.cpu())
    print_architecture(model.cpu())
    model.to(device)

    train()
    test()
