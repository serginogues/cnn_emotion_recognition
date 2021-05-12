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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FER_PATH = 'data/fer2013.csv'  # Path to the csv file with emotion, pixel & usage.
NUM_EPOCHS = 100
BATCH_SIZE = 128
LR = 0.1
