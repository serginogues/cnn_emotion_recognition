import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import h5py
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Initial settings seed and device
SET_SEED = 11
# Compatibility with CUDA and GPU -> remember to move into GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# make deterministic the stochastic operation to have better comparable tests
if SET_SEED != -1:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SET_SEED)
        torch.cuda.manual_seed_all(SET_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(SET_SEED)
    torch.manual_seed(SET_SEED)



FER_PATH = '../archive/fer2013.csv'  # Path to the csv file with emotion, pixel & usage.
NUM_EPOCHS = 100
BATCH_SIZE = 25
NUM_LABELS = 7
LR = 0.001
