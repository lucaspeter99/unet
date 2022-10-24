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

device = 'cpu'

def save_and_vis_saliency(sal_tensor, output_dir, plot_dir, patientid, ctn, gt, pred):
    output_dir = os.path.join(output_dir, 'saliency')
    modalities = ['reca_status', 'age', 'nihss', 'time_to_ct', 'sex', 'CTA', 'CTN','CTP_CBV', 'CTP_CBF','CTP_Tmax']
    #maxs = []
    for i, mod in enumerate(modalities):
        #directory
        output_mod_dir = os.path.join(output_dir, mod)
        if not os.path.exists(output_mod_dir):
            os.makedirs(output_mod_dir)
        #get data
        mod_saliency = sal_tensor[i,:,:,:]
        #save in dir
        img = nib.Nifti1Image(mod_saliency.detach().numpy(), numpy.eye(4))
        nib.save(img, os.path.join(output_mod_dir, str(patientid) + ".nii.gz"))
        #maxs.append(torch.max(mod_saliency))
    max_sal = torch.max(sal_tensor)
    #vis
    sal_plot_dir = os.path.join(plot_dir, 'saliency')
    if not os.path.exists(sal_plot_dir):
        os.makedirs(sal_plot_dir)
    # plt.figure(num=1, figsize=[240.0, 100.0], dpi=100)
    fig, ax_array = plt.subplots(16, 13, figsize=(10,10))
    fig.set_size_inches(120.0, 160.0)
    modalities.insert(0,'CTN')
    modalities.append('Ground Truth')
    modalities.append('Prediction')
    for i, ax_row in enumerate(ax_array):
        for j, axes in enumerate(ax_row):
            if i == 0:
                axes.set_yticklabels([])
                axes.set_xticklabels([])
                axes.grid('off')
                if j == 0:
                    axes.imshow(ctn[:, 3 * i, :], cmap='Greys_r', interpolation='none')
                elif j == 11:
                    axes.imshow(gt[:, 3 * i, :], cmap='Greys_r', interpolation='none')
                elif j == 12:
                    axes.imshow(pred[:, 3 * i, :], cmap='Greys_r', interpolation='none')
                else:
                    axes.imshow(sal_tensor[j-1, :, 3 * i, :], vmin = 0, vmax=max_sal, cmap='hot', interpolation='none')
                axes.set_title(modalities[j], fontsize = 64)
            else:
                axes.set_yticklabels([])
                axes.set_xticklabels([])
                axes.grid('off')
                if j == 0:
                    axes.imshow(ctn[:, 3 * i, :], cmap='Greys_r', interpolation='none')
                elif j == 11:
                    axes.imshow(gt[:, 3 * i, :], cmap='Greys_r', interpolation='none')
                elif j == 12:
                    axes.imshow(pred[:, 3 * i, :], cmap='Greys_r', interpolation='none')
                else:
                    axes.imshow(sal_tensor[j-1, :, 3 * i, :], vmin = 0, vmax=max_sal ,cmap='hot', interpolation='none')


    plt.savefig(os.path.join(sal_plot_dir, f'comp_saliency_{patientid}.png'), dpi=100)
    plt.close(fig)



def compute_metrics(data_obj, key, generator, metrics, plot_dir, thresholds, ml_per_voxel, gen_dir,
                    best_threshold=None):
    plot_dir = os.path.join(plot_dir, key)

    output_dir = os.path.join(gen_dir, 'outputs')
    y0_output_dir = os.path.join(output_dir, "y0")
    y1_output_dir = os.path.join(output_dir, "y1")
    prediction_output_dir = os.path.join(output_dir, "prediction")
    gt_output_dir = os.path.join(output_dir, "gt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(y0_output_dir, exist_ok=True)
    os.makedirs(y1_output_dir, exist_ok=True)
    os.makedirs(prediction_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    print("Plotting for ", key, " set")
    for l, data in enumerate(data_obj):  # run thru dataset
        print("[", l + 1, "/", len(data_obj), "]")
        x, y, patientid = data  # item in dataset
        #input_clone = copy.deepcopy(x)
        x = x.float().to(device)
        # get binary ground truth
        y = (y > 0.5).float().to(device)
        y = y.cpu().detach()  # move tensor to cpu
        #image has to require_grad() to compute saliency
        x.requires_grad_()
        # predict with model
        score = generator(x.unsqueeze(0))
        #get maxs for saliency
        output_max = torch.max(score)
        #compute saliency
        output_max.backward()
        #now saliency is x.grad
        with torch.no_grad():
            input_image = x[6,:,:,:].detach()
        save_and_vis_saliency(sal_tensor=torch.abs(x.grad), output_dir=output_dir, plot_dir=plot_dir, patientid=patientid, ctn= input_image, gt = y[0,:,:,:], pred=score[0,0,:,:,:].detach())

        score = score.cpu().detach()  # move tensor to cpu

        # ------------------------------
        # save prediction, gt with nib
        # ------------------------------
        img = nib.Nifti1Image(score[0, 0, :, :, :].detach().numpy(), numpy.eye(4))
        nib.save(img, os.path.join(prediction_output_dir, str(patientid) + ".nii.gz"))

        img = nib.Nifti1Image(y[0, :, :, :].detach().numpy(), numpy.eye(4))
        nib.save(img, os.path.join(gt_output_dir, str(patientid) + ".nii.gz"))

        del img
        # plot comparison output to gt
        if use_medical_data:
            background = x[5].detach().cpu()
        else:
            background = x[0].detach().cpu()
        center = utils.plot_comparison(score=score, target=y, item=patientid, plot_dir=plot_dir, background=background)

        # plot mismatches
        if use_medical_data:
            # load xs with 0 and 1 in tici scale pos
            x = x.squeeze(0)
            x0 = torch.full((1, 48, 48, 48), 0).to(device)
            for i in range(9):
                x0 = torch.cat((x0, x[i + 1].unsqueeze(0)), 0)

            x1 = torch.full((1, 48, 48, 48), 1).to(device)
            for i in range(9):
                x1 = torch.cat((x1, x[i + 1].unsqueeze(0)), 0)

            # predict
            y0 = generator(x0.unsqueeze(0))
            y0 = y0.cpu().detach()

            y1 = generator(x1.unsqueeze(0))
            y1 = y1.cpu().detach()

            # plot mismatch between predictions
            utils.plot_mismatch(y0, y1, patientid, plot_dir, center)
            # ------------------------------
            # save y0, y1
            # ------------------------------
            img = nib.Nifti1Image(y0[0, 0, :, :, :].detach().numpy(), numpy.eye(4))
            nib.save(img, os.path.join(y0_output_dir, str(patientid) + ".nii.gz"))

            img = nib.Nifti1Image(y1[0, 0, :, :, :].detach().numpy(), numpy.eye(4))
            nib.save(img, os.path.join(y1_output_dir, str(patientid) + ".nii.gz"))

            del img

        try:
            # save metrics
            tpr, auc, dices, diffs, vol_diffs, accuracies, precisions, recalls, fprs, f1s, gt_vol = utils.plot_metrics(
                score=score, target=y, item=patientid, plot_dir=plot_dir, thresholds=thresholds,
                ml_per_voxel=ml_per_voxel)
            metrics[key]['tpr'].append(tpr)
            metrics[key]['auc'].append(auc)
            metrics[key]['dice'].append(dices)
            metrics[key]['diff'].append(diffs)
            metrics[key]['vol_diff'].append(vol_diffs)
            metrics[key]['accuracy'].append(accuracies)
            metrics[key]['precision'].append(precisions)
            metrics[key]['recall'].append(recalls)
            metrics[key]['fpr'].append(fprs)
            metrics[key]['f1'].append(f1s)
            metrics[key]['gt_vol'].append(gt_vol)
            metrics[key]['reca'].append(x[0, 0, 0, 0].item())
            metrics[key]['id'].append(patientid)
            del tpr, auc, dices, diffs, vol_diffs, accuracies, precisions, recalls, fprs, gt_vol
        except ValueError as ve:
            print(f'{ve.__class__.__name__} in main2.compute_metrics: {ve}')

        del x, y, score#, input_clone

    # save all statistics for different thresholds
    statistics = [
        'dice',
        'diff',
        'vol_diff',
        'accuracy',
        'precision',
        'recall',
        'fpr',
        'f1', ]
    thresholds.insert(0, 'id')
    for statistic in statistics:
        csv_path = os.path.join(plot_dir, statistic + '.csv')  # filepath
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)  # write file
            writer.writerow(thresholds)  # write header
            for j, stat_list in enumerate(metrics[key][statistic]):  # write rest of the file
                id = metrics[key]['id'][j]
                stat_list_dummy = stat_list.copy()
                stat_list_dummy.insert(0, id)
                writer.writerow(stat_list_dummy)
    thresholds.remove('id')

    # save aucs
    with open(os.path.join(plot_dir, 'auc.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'auc'])
        for j, auc in enumerate(metrics[key]['auc']):
            id = metrics[key]['id'][j]
            writer.writerow([id, auc])

    if key == 'validation':
        # save plots for average statistics on different thresholds
        for statistic in statistics:
            # get average for different tresholds
            averages = []

            for i, threshold in enumerate(thresholds):
                statistic_for_threshold = []
                for stat_list in metrics[key][statistic]:
                    statistic_for_threshold.append(stat_list[i])
                averages.append(sum(statistic_for_threshold) / len(statistic_for_threshold))

            # plot the average
            utils.plot_average_stats_for_different_thesholds(thresholds, averages, plot_dir, name=statistic,
                                                             best_threshold=best_threshold)

        statistics.insert(0, 'gt_vol')
        statistics.insert(0, 'auc')
        statistics.insert(0, 'id')
        csv_path = os.path.join(plot_dir, 'stats.csv')

        # for val data set metrics only have to be saved for the best threshold
        index_best_threshold = thresholds.index(best_threshold)
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(statistics)
            statistics.remove('id')
            statistics.remove('auc')
            statistics.remove('gt_vol')
            for j, id in enumerate(metrics[key]['id']):
                auc = metrics[key]['auc'][j]
                gt_vol = metrics[key]['gt_vol'][j]
                row = [id, auc, gt_vol]
                for statistic in statistics:
                    row.append(metrics[key][statistic][j][index_best_threshold])
                writer.writerow(row)







#this plots roc_means.png
def plot_metrics(metrics, key, plot_dir, best_threshold = 0, border_color='black'):
    print("plotting roc means...")
    plot_dir = os.path.join(plot_dir, key)
    fig = plt.figure()
    #plot combined roc curves
    tprs = numpy.array(metrics[key]['tpr']) #list of all roc curves
    tpr_mean = [numpy.nanmean(tprs[:, i]) for i in range(101)] #mean curve
    tpr_upper = [numpy.percentile(tprs[:, i], 97.5) for i in range(101)]
    tpr_lower = [numpy.percentile(tprs[:, i], 2.5) for i in range(101)]
    x = numpy.arange(101) / 100
    #plot the actual curves
    for tpr in tprs:
        plt.plot(x, tpr)
    #plot the mean and upper, lower
    plt.plot(x, tpr_mean, color=border_color, linewidth=3, linestyle='dashed')
    plt.plot(x, tpr_upper, color=border_color, linewidth=3, linestyle='dotted')
    plt.plot(x, tpr_lower, color=border_color, linewidth=3, linestyle='dotted')
    auc_mean = numpy.nanmean(numpy.array(metrics[key]['auc']))
    #add corresonding subtitles with other metrics
    if key == 'validation':
        print("Getting average values...")
        dice_mean = round(numpy.nanmean(numpy.array(metrics[key]['dice'])[:,best_threshold]),2)
        vol_diff_mean = round(numpy.nanmean(numpy.array(metrics[key]['vol_diff'])[:,best_threshold]),2)
        f1_mean = round(numpy.nanmean(numpy.array(metrics[key]['f1'])[:,best_threshold]),2)

        plt.suptitle(f'mean AUC: {auc_mean} / mean vol.Diff: {vol_diff_mean} \n mean DICE: {dice_mean}/  mean f1: {f1_mean}')
        logger.log(f"Mean VAL AUC:{auc_mean}\n"
                   f"Mean VAL DICE:{dice_mean}\n"
                   f"Mean VAL VOL DIFF:{vol_diff_mean}\n"
                   f"Mean VAL F1:{f1_mean}\n")
    else:
        plt.suptitle(f'mean AUC: {auc_mean}')
        logger.log(f"Mean AUC:{auc_mean}")
    fig.savefig(os.path.join(plot_dir, 'roc_means'))
    plt.close()


def run_stats(val_dataset, gen, csv_dir, plot_dir, ml_per_voxel):

    # metrics will be saved in here
    metrics = {
        'validation': {
            'tpr': [],
            'auc': [],
            'dice': [],
            'diff': [],
            'vol_diff': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'f1': [],
            'id': [],
            'reca': [],
            'gt_vol': [],
        }
    }
    possible_thresholds = list([round(i * 0.02, 2) for i in range(50)])  # thresholds that are considered

    print("Determining best threshold from training set...")
    df = pd.read_csv(os.path.join(csv_dir, 'vol_diff.csv'), index_col='id')  # read csv file
    best_mean = sys.maxsize  # best mean vol diff, since it is being compared set to a large value in the beginning
    best_threshold_on_training_set = 0  # best threshold, set to 0 at the start
    for threshold in possible_thresholds:  # for all thresholds
        stat_list = df[str(threshold)].tolist()  # get list of metric of all patients for that threshold
        mean_stat = numpy.nanmean(numpy.array(stat_list))  # get mean of that list
        if mean_stat < best_mean:  # if mean is better update best mean and best threshold
            best_mean = mean_stat
            best_threshold_on_training_set = threshold
    print("Best threshold is ", best_threshold_on_training_set)
    best_threshold_on_training_set_index = possible_thresholds.index(float(best_threshold_on_training_set))

    # compute metrics with best threshold on test set
    compute_metrics(data_obj=val_dataset, key='validation', generator=gen, metrics=metrics, plot_dir=plot_dir,
                    ml_per_voxel=ml_per_voxel, thresholds=possible_thresholds,
                    best_threshold=float(best_threshold_on_training_set), gen_dir=gen_dir)
    # finishing with some logging
    plot_metrics(metrics, 'validation', plot_dir, best_threshold_on_training_set_index)

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

val_data_dir = './dresden_data/data'
gen_dir = './eval_dresden'

model = 'SEED555_2022-08-03_500_bs12_lr0.0001_UNetSepMedDataCheckpoint_Adam_BCELoss_ReslicedAllModalities_medDataSplit/epoch150/'

log_file = os.path.join(gen_dir, "log")
logger = utils.LeanLogger(name=log_file)

if 'medData' in model:
    use_medical_data = True
else:
    use_medical_data = False

gen_dir = os.path.join(gen_dir,model)
plot_dir = gen_dir + "plots/"
output_dir = gen_dir + "/outputs"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#get class of dataset
hyperparams = model.split('_')
data_set_class = eval('datasets.' + hyperparams[8])
print('Using ', data_set_class)

#get windowing
if hyperparams[9] == 'windowed':
    windowing = True
    print("Windowing active")
else:
    windowing = False
    print("No windowing")

print("Loading Dataset")
dataset = data_set_class(val_data_dir,rotate = False, deform = False, start=-0, load_old_lesions = False, use_windowing = windowing)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


ml_per_voxel = (numpy.prod(numpy.array(dataset.dimensionality)) / 1000) / (pow(dataset.size, 3))
print("ML PER VOXEL", ml_per_voxel)


# load model:
print('Loading model...')
model = torch.load(gen_dir + "gen.trc", map_location=torch.device(device))
#model.eval()
print('plotting...')
run_stats(val_dataset=dataset, gen = model, csv_dir = gen_dir, plot_dir=plot_dir, ml_per_voxel=ml_per_voxel)

