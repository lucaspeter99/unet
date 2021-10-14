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
import datasets
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam
import csv

dataset_classes = {
    0: datasets.NormalizedAngioNative,
    1: datasets.NormalizedPerfusionCBV,
    2: datasets.NormalizedPerfusionCBF,
    3: datasets.NormalizedPerfusionTmax,
    4: datasets.NormalizedPerfusionAll,
    5: datasets.NormalizedAllModalities,
    6: datasets.NormalizedAngioNativeTmax,
    7: datasets.ReslicedAngioNative,
    8: datasets.ReslicedPerfusionCBV,
    9: datasets.ReslicedPerfusionCBF,
    10: datasets.ReslicedPerfusionTmax,
    11: datasets.ReslicedPerfusionAll,
    12: datasets.ReslicedAllModalities,
    13: datasets.ReslicedAngioNativeTmax
}

data_dir = './data'
gen_dir = './results'


for dir in os.listdir(gen_dir):
    path = os.path.join(gen_dir, dir)
    gen_path = os.path.join(path, 'gen.trc')
    if os.path.isfile(gen_path):
        print(gen_path, ' GEN FOUND')
        # load val data set
        class_id = None
        for (dataset_id, dataset_class) in dataset_classes.items():
            if dataset_class.__name__ in path:
                class_id = dataset_id
        if class_id is not None:
            print('Class ', class_id)
            print('Loading Dataset...')
            dataset = dataset_classes[class_id](data_dir,rotate = False, deform = False, start=-14)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            # load model:
            print('Loading model...')
            model = torch.load(gen_path, map_location=torch.device('cpu'))
            # predict
            print('plotting...')
            for j, data in enumerate(loader):
                x, y = data

                x = x.float().to('cpu')
                y = (y > 0.5).float().to('cpu')[0,:,:,:,:]
                output = model(x)
                output = output.detach()
                print('[',j,'/14]')
                #output = output.unsqueeze(0).unsqueeze(0)
                utils.plot_comparison(score=output, target=y, item=j, plot_dir=os.path.join(path, 'plots', 'validation'))
        else:
            print('No matching Class found')

    else:
        print(gen_path, ' NO GEN FOUND')