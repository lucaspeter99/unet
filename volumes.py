import csv
import json
import math
import sys
from datetime import datetime
import utils
import os
from modified_models import UNetSepMedDataCheckpoint, UNetSepCheckpoint,  UNetSmallSepMedDataCheckpoint, UNetSmallCheckpoint, UNetSepMedDataCombinedCheckpoint,UNetSepMedDataCombinedRecaSepCheckpoint
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint_sequential
import numpy
import matplotlib.pyplot as plt
import datasets
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam, SGD
import nibabel as nib
from sklearn import model_selection
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

resliced_vols = pd.read_csv("./results/resliced_volumes.csv", index_col = 'id')
norm_vols = pd.read_csv("./results/norm_volumes.csv", index_col = 'id')

s1 = pd.read_csv("./results/s1.csv", index_col = 'id')
s2 = pd.read_csv("./results/s2.csv", index_col = 'id')

with open('./results/vols.csv', 'w', newline='') as file:
    w = csv.writer(file)
    w.writerow(["seed", "type", "average vol"])
    seed1 = s1.loc[s1['testing'] == 1]
    seed2 = s2.loc[s2['testing'] == 1]
    s1_res = s1.join(resliced_vols)
    s2_res = s2.join(resliced_vols)
    s1_norm = s1.join(norm_vols)
    s2_norm = s2.join(norm_vols)
    avg_vol = numpy.mean(numpy.array(s1_res.loc[s1_res['testing'] == 1]['gt_vol']))
    w.writerow([555, "resliced", avg_vol])
    avg_vol = numpy.mean(numpy.array(s2_res.loc[s2_res['testing'] == 1]['gt_vol']))
    w.writerow([666, "resliced", avg_vol])
    avg_vol = numpy.mean(numpy.array(s1_norm.loc[s1_norm['testing'] == 1]['gt_vol']))
    w.writerow([555, "normalized", avg_vol])
    avg_vol = numpy.mean(numpy.array(s2_norm.loc[s2_norm['testing'] == 1]['gt_vol']))
    w.writerow([666, "normalized", avg_vol])


# DatasetClass = dataset_classes[5]
# data_dir = './train_data'
#
# data = DatasetClass(data_dir, rotate=False, deform=False, alpha=0, sigma=0, onlyRotDefSpecials=True,
#                             use_medical_data = True,
#                            start = -0
#                            )
#
# reca_statuses = []
#
# # get reca statuses for all patients
# for i, item in enumerate(data.items):
#     reca_statuses.append(item[0][0, 0, 0, 0].item())
#
# # indices are determined by stratified split based on reca status
# x_train, x_test, y_train, y_test = model_selection.train_test_split(list(range(len(data))), reca_statuses,
#                                                                     random_state=666, shuffle=True, test_size=(1 / 6),
#                                                                     stratify=reca_statuses)
#
# train_dataset = Subset(data, x_train) # training set
# test_dataset = Subset(data, x_test) # test set
#
# with open('./results/s2.csv', 'w', newline='') as file:
#     w = csv.writer(file)
#     w.writerow(["id", "training", "testing"])
#     for d in train_dataset:
#         w.writerow([d[2], 1, 0])
#     for d in test_dataset:
#         w.writerow([d[2], 0, 1])
#
#
# with open('./results/norm_volumes.csv', 'w', newline='') as file:
#     w = csv.writer(file)
#     w.writerow(["id", "gt_vol"])
#
#     for d in data:
#         id = d[2]
#         voxel  = torch.sum(d[1]).item()
#
#         ml_per_voxel = (numpy.prod(numpy.array(data.dimensionality)) / 1000) / (pow(data.size, 3))
#         mls = voxel * ml_per_voxel
#         w.writerow([id, mls])





