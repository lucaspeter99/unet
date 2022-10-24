import json
import sys
from datetime import datetime

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import utils
import os
from modified_models import UNetSmallSepMedDataCheckpoint
import torch
import numpy
import matplotlib.pyplot as plt
import datasets
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam
import csv
import pandas as pd
import nibabel as nib
import copy


def vis_saliency(saliencies, plot_dir, modalities):
    max_sal = 0
    for mod in modalities:
        mod_max = torch.max(saliencies[mod])
        if mod_max > max_sal:
            max_sal = mod_max

    #vis
    sal_plot_dir = os.path.join(plot_dir, 'saliency')
    if not os.path.exists(sal_plot_dir):
        os.makedirs(sal_plot_dir)
    fig, ax_array = plt.subplots(16, 10, figsize=(10,10))
    fig.set_size_inches(90.0, 160.0)
    for i, ax_row in enumerate(ax_array):
        for j, axes in enumerate(ax_row):
            print(saliencies[modalities[j]].shape)
            if i == 0:
                axes.set_yticklabels([])
                axes.set_xticklabels([])
                axes.grid('off')
                axes.imshow(saliencies[modalities[j]][0, :, 3 * i, :], vmin = 0, vmax=max_sal, cmap='hot', interpolation='none')
                axes.set_title(modalities[j], fontsize = 64)
            else:
                axes.set_yticklabels([])
                axes.set_xticklabels([])
                axes.grid('off')
                axes.imshow(saliencies[modalities[j]][0, :, 3 * i, :], vmin = 0, vmax=max_sal ,cmap='hot', interpolation='none')


    plt.savefig(os.path.join(sal_plot_dir, f'comp_saliency_combined.png'), dpi=100)
    plt.close(fig)


saliency_dir = "./eval_dresden/SEED555_2022-08-03_500_bs12_lr0.0001_UNetSepMedDataCheckpoint_Adam_BCELoss_ReslicedAllModalities_medDataSplit/epoch150/outputs/saliency"

#modalities = sorted(list(os.listdir(saliency_dir)))
modalities = ['reca_status', 'age', 'nihss', 'time_to_ct', 'sex', 'CTA', 'CTN', 'CTP_CBV', 'CTP_CBF', 'CTP_Tmax']
patient_id_files = os.listdir(os.path.join(saliency_dir, modalities[0]))
print(modalities)
print(patient_id_files)

saliencies = {
    'reca_status': torch.zeros((1,48,48,48)),
    'age': torch.zeros((1,48,48,48)),
    'nihss': torch.zeros((1,48,48,48)),
    'time_to_ct': torch.zeros((1,48,48,48)),
    'sex': torch.zeros((1,48,48,48)),
    'CTA': torch.zeros((1,48,48,48)),
    'CTN': torch.zeros((1,48,48,48)),
    'CTP_CBV': torch.zeros((1,48,48,48)),
    'CTP_CBF': torch.zeros((1,48,48,48)),
    'CTP_Tmax': torch.zeros((1,48,48,48))
}
print(saliencies['age'])

for file in patient_id_files:
    for mod in modalities:
        data = torch.tensor(nib.load(os.path.join(saliency_dir, mod, file)).get_fdata()).unsqueeze(dim=0)
        saliencies[mod] += data

vis_saliency(saliencies, "./eval_dresden/SEED555_2022-08-03_500_bs12_lr0.0001_UNetSepMedDataCheckpoint_Adam_BCELoss_ReslicedAllModalities_medDataSplit/epoch150/", modalities)