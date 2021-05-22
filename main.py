"""
Emotion Recognition with a CNN
"""
from cnn import train, test, visualize_filter, model, print_architecture

if __name__ == '__main__':

    # visualize_filter(model.cpu())
    # print_architecture(model.cpu())

    train()
    test()
