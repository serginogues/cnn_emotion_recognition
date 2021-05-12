from preprocess import *

input_size = 48 * 48
num_labels = len(Labels)


def accuracy(output, labels):
    predictions, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
