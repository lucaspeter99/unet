import json
import sys
from datetime import datetime

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import utils
import os
from modified_models import UNet
import torch
import numpy
import matplotlib.pyplot as plt
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam
import csv
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn.functional as F
import plotly.graph_objects as go
import numpy as np
from sklearn import metrics
import numpy as np
import plotly.express as px
import deca_datasets

DatasetClass = deca_datasets.Colon

data_dir = data_dir = './deca_data/Task10_Colon/Task10_Colon/'
rotate = False
deform = False
alpha = 0.2
sigma = 0.1

train_dataset = DatasetClass(data_dir, training = True, rotate=rotate, deform=deform, alpha=alpha, sigma=sigma, start=-2)

train_data = []

for (i, d) in enumerate(train_dataset):
    (x,_) = d
    train_data.append(x.squeeze().numpy())

train_data = np.array(train_data, dtype = float)
print(train_data.shape)
data = train_data.flatten()
print(data.shape)
print(data)

std = data.std(0)
mean = data.mean(0)

print("std: ",std)
print("mean: ", mean)