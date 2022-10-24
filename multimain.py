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
import warnings
import gc
import pynvml
from main2 import create_and_train_unet
from scipy import stats as st
import shutil

run_path = str(sys.argv[1]) if len(sys.argv) > 1 else './runs.csv'
samples_to_keep = int(sys.argv[2]) if len(sys.argv) > 2 else 3

df = pd.read_csv(run_path, index_col="run")
runs = df.index

for run in runs:
    print("Running: ", df.iloc[[run]])
    results_dir = create_and_train_unet(
                        tag=df.iloc[[run]]["tag"].item(),
                        dc_key = df.iloc[[run]]["dc_key"].item(),
                        lr =df.iloc[[run]]["lr"].item(),
                        num_epochs =df.iloc[[run]]["num_epochs"].item(),
                        mse =df.iloc[[run]]["mse"].item(),
                        batchnorm =df.iloc[[run]]["batchnorm"].item(),
                        leaky =df.iloc[[run]]["leaky"].item(),
                        max_pool =df.iloc[[run]]["max_pool"].item(),
                        start_data =df.iloc[[run]]["start_data"].item(),
                        use_medical_data =df.iloc[[run]]["use_medical_data"].item(),
                        split_med_channels =df.iloc[[run]]["split_med_channels"].item(),
                        only_split_tici =df.iloc[[run]]["only_split_tici"].item(),
                        split_modalities =df.iloc[[run]]["split_modalities"].item(),
                        windowing =df.iloc[[run]]["windowing"].item(),
                        use_gpu =df.iloc[[run]]["use_gpu"].item(),
                        batch_size =df.iloc[[run]]["batch_size"].item(),
                        seed =df.iloc[[run]]["seed"].item(),
                        epoch_sample_interval =df.iloc[[run]]["epoch_sample_interval"].item()
    )

    #decide which samples to keep
    sample_dirs = next(os.walk(results_dir))[1]
    dices = []
    aucs = []
    vol_diffs = []
    sample_paths = []
    for sample_path in sample_dirs:
        sample_path = os.path.join(results_dir, sample_path)
        sample_paths.append(sample_path)
        val_path = os.path.join(sample_path, "plots", "validation")
        stats = pd.read_csv(os.path.join(val_path, "stats.csv"), index_col= "id")
        aucs.append(numpy.nanmean(numpy.array(stats["auc"].tolist())))
        dices.append(numpy.nanmean(numpy.array(stats["dice"].tolist())))
        vol_diffs.append(numpy.nanmean(numpy.array(stats["vol_diff"].tolist())))

    #save stats for all epochs
    csv_path = os.path.join(results_dir,'epoch_stats.csv')  # filepath
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)  # write file
        writer.writerow(["epochs", "dice", "auc", "vol_diff"])
        for j, epochs in enumerate(sample_dirs):  # write rest of the file
            writer.writerow([epochs, dices[j], aucs[j], vol_diffs[j]])

    #standardize arrays
    dices = st.zscore(numpy.array(dices))
    aucs = st.zscore(numpy.array(aucs))
    vol_diffs = -1 * st.zscore(numpy.array(vol_diffs))

    #add them all to compute a metric
    metrics = dices + aucs + vol_diffs

    #create tuples
    tuples = list(zip(sample_paths, metrics))
    #sort tuples by metric
    sorted_by_metric = sorted(tuples, key=lambda tup: tup[1], reverse= True)
    for path, _ in sorted_by_metric[samples_to_keep:]:
        print("removing ", path, " ...")
        shutil.rmtree(path)







