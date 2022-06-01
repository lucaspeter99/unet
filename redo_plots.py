import json
import sys
from datetime import datetime

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
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
import pandas as pd
import nibabel as nib

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
val_data_dir = './val_data'
gen_dir = './results'
val_plot_dir = 'plots/updated_val/'


# def plot_metrics(tprs, dices, aucs,vdiffs, plot_dir, border_color='black'):
#     tprs = numpy.array(tprs)
#     tpr_mean = [tprs[:, i].mean() for i in range(101)]
#     tpr_upper = [numpy.percentile(tprs[:, i], 97.5) for i in range(101)]
#     tpr_lower = [numpy.percentile(tprs[:, i], 2.5) for i in range(101)]
#     x = numpy.arange(101) / 100
#     for tpr in tprs:
#         plt.plot(x, tpr)
#     plt.plot(x, tpr_mean, color=border_color, linewidth=3, linestyle='dashed')
#     plt.plot(x, tpr_upper, color=border_color, linewidth=3, linestyle='dotted')
#     plt.plot(x, tpr_lower, color=border_color, linewidth=3, linestyle='dotted')
#     auc_mean = numpy.array(aucs).mean()
#     dice_mean = numpy.array(dices).mean()
#     vdiff_mean = numpy.array(vdiffs).mean()
#     plt.suptitle(f'mean AUC: {auc_mean} / mean vDiff: {vdiff_mean} \n mean DICE: {dice_mean}')
#     plt.savefig(os.path.join(plot_dir, 'roc_means'))
#     plt.close()

def plot_metrics(metrics, key, plot_dir, best_threshold = 0, border_color='black'):
    plot_dir = os.path.join(plot_dir, key)
    fig = plt.figure()
    #plot combined roc curves
    tprs = numpy.array(metrics[key]['tpr'])
    tpr_mean = [tprs[:, i].mean() for i in range(101)]
    tpr_upper = [numpy.percentile(tprs[:, i], 97.5) for i in range(101)]
    tpr_lower = [numpy.percentile(tprs[:, i], 2.5) for i in range(101)]
    x = numpy.arange(101) / 100
    for tpr in tprs:
        plt.plot(x, tpr)
    plt.plot(x, tpr_mean, color=border_color, linewidth=3, linestyle='dashed')
    plt.plot(x, tpr_upper, color=border_color, linewidth=3, linestyle='dotted')
    plt.plot(x, tpr_lower, color=border_color, linewidth=3, linestyle='dotted')
    auc_mean = numpy.array(metrics[key]['auc']).mean()
    if key == 'validation':
        dice_mean = round(numpy.array(metrics[key]['dice'])[:,best_threshold].mean(),2)
        vol_diff_mean = round(numpy.array(metrics[key]['vol_diff'])[:,best_threshold].mean(),2)
        f1_mean = round(numpy.array(metrics[key]['f1'])[:,best_threshold].mean(),2)

        plt.suptitle(f'mean AUC: {auc_mean} / mean vol.Diff: {vol_diff_mean} \n mean DICE: {dice_mean}/  mean f1: {f1_mean}')
    else:
        plt.suptitle(f'mean AUC: {auc_mean}')
    fig.savefig(os.path.join(plot_dir, 'roc_means'))
    plt.close()






for dir in os.listdir(gen_dir):
    path = os.path.join(gen_dir, dir)
    plot_dir = os.path.join(path, val_plot_dir)
    gen_path = os.path.join(path, 'gen.trc')
    if "SEP_2022-05-04_150_bs10_lr0.001_UNet_Adam_MSELoss_ReslicedAllModalities_medData" not in gen_path:
        continue
    if os.path.isfile(gen_path):
        medData = False
        if 'medData' in gen_path:
            medData = True
        print(gen_path, ' GEN FOUND')
        print("saving to: ", plot_dir)
        #create path for eval and output
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        output_dir = os.path.join(gen_dir, dir, 'outputs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #GET BEST THRESHOLD FOR DICE
        dice_thresholds = list([round(i * 0.05, 2) for i in range(20)])
        print("Determining best threshold from training set...")
        df = pd.read_csv(os.path.join(plot_dir, 'training', 'vol_diff.csv'),
                         index_col='id')
        best_mean = sys.maxsize
        possible_thresholds = list([round(i * 0.05, 2) for i in range(20)])
        best_threshold_on_training_set = 0
        for threshold in possible_thresholds:
            stat_list = df[str(threshold)].tolist()
            mean_stat = sum(stat_list) / len(stat_list)
            if mean_stat < best_mean:
                best_mean = mean_stat
                best_threshold_on_training_set = threshold
        print("Best threshold is ", best_threshold_on_training_set)
        best_threshold_on_training_set_index = possible_thresholds.index(float(best_threshold_on_training_set))
        # load val data set
        class_id = None
        for (dataset_id, dataset_class) in dataset_classes.items():
            if dataset_class.__name__ in path:
                class_id = dataset_id
        if class_id is not None:
            print('Class ', class_id)
            print('Loading Dataset...')
            dataset = dataset_classes[class_id](val_data_dir,rotate = False, deform = False, start=-3, load_old_lesions = False)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            # load model:
            print('Loading model...')
            model = torch.load(gen_path, map_location=torch.device('cpu'))
            # predict
            print('plotting...')
            tprs = []
            dices = []
            aucs = []
            diffs =[]
            vol_diffs = []
            accuracies = []
            precisions = []
            recalls = []
            fprs = []
            f1s = []
            ids = []
            recas = []
            for j, data in enumerate(loader):
                #LOAD DATA, PREDICT
                x, y, patientid = data
                recas.append(x[0,0,0,0,0].item())
                patientid = patientid[0] #?????????why
                x = x.float().to('cpu')
                y = (y > 0.5).float().to('cpu')[0,:,:,:,:]
                output = model(x)
                output = output.detach()
                #save output
                img = nib.Nifti1Image(output[0,0,:,:,:].detach().numpy(), np.eye(4))
                nib.save(img, os.path.join(output_dir, str(patientid)+".nii.gz"))
                del img


                print('[',j+1 ,'/', len(loader) ,']:', patientid)
                #PLOT RISC MAPS
                utils.plot_comparison(score=output, target=y, item=patientid, plot_dir=plot_dir)
                #PLOT MISMATCHES
                if medData:
                    x = x.squeeze(0)
                    x0 = x[0].unsqueeze(0)
                    for i in range(9):
                        if i != 0:
                            x0 = torch.cat((x0, x[i+1].unsqueeze(0)), 0)
                        else:
                            x0 = torch.cat(
                                (x0,torch.full((1,48, 48, 48),0)), 0)

                    x1 = x[0].unsqueeze(0)
                    for i in range(9):
                        if i != 0:
                            x1 = torch.cat((x1, x[i + 1].unsqueeze(0)), 0)
                        else:
                            x1 = torch.cat((x1, torch.full((1,48, 48, 48),1)), 0)

                    # predict
                    y0 = model(x0.unsqueeze(0))
                    y0 = y0.cpu().detach()
                    #y0 = (y0 > thresholds[0]).float().to('cpu')  # thresholds[0] is best threshold determined by dice on training data set (as this code only runs on val data set)
                    y1 = model(x1.unsqueeze(0))
                    y1 = y1.cpu().detach()
                    #y1 = (y1 > thresholds[0]).float().to('cpu')
                    utils.plot_mismatch(y0, y1, patientid, plot_dir)
                #GET STATS
                try:
                    tpr, auc, dice, diff, vol_diff, accuracy, precision, recall, fpr, f1 = utils.plot_metrics(score=output, target=y, item=patientid, plot_dir=plot_dir, thresholds=possible_thresholds)
                    tprs.append(tpr)
                    dices.append(dice)
                    aucs.append(auc)
                    diffs.append(diff)
                    vol_diffs.append(vol_diff)
                    accuracies.append(accuracy)
                    precisions.append(precision)
                    recalls.append(recall)
                    fprs.append(fpr)
                    f1s.append(f1)
                    ids.append(id)

                    del tpr, auc, dice, diff, vol_diff, accuracy, precision, recall, fpr, f1
                except ValueError as ve:
                    print(f'{ve.__class__.__name__} in utils.plot_roc: {ve}')


            statistics = [
                (tprs, "tprs"),
                (dices, "dices"),
                (aucs, "aucs"),
                (diffs, "diffs"),
                (vol_diffs, "vol_diffs"),
                (accuracies, "accuracies"),
                (precisions, "precisions"),
                (recalls, "recalls"),
                (fprs, "fprs"),
                (f1s, "f1s"),
                (ids, "ids"),
                (recas, "recas"),
            ]

            #save stats for all metrics
            possible_thresholds.insert(0, 'id')
            for statistic in statistics:

                csv_path = os.path.join(plot_dir, statistic[1] + '.csv')
                with open(csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(possible_thresholds)
                    for j, stat_list in enumerate(statistic[0]):
                        id = ids[j]
                        stat_list_dummy = stat_list.copy()
                        stat_list_dummy.insert(0, id)
                        writer.writerow(stat_list_dummy)
            possible_thresholds.remove('id')

            # save plots for average statistics on different thresholds
            for statistic in statistics:
                # get average for different tresholds
                averages = []

                for i, threshold in enumerate(possible_thresholds):
                    statistic_for_threshold = []
                    for stat_list in statistic[0]:
                        statistic_for_threshold.append(stat_list[i])
                    averages.append(sum(statistic_for_threshold) / len(statistic_for_threshold))

                # plot the average
                utils.plot_average_stats_for_different_thesholds(possible_thresholds, averages, plot_dir, name=statistic, best_threshold=best_threshold_on_training_set)

            statistics.insert(0, 'id')
            csv_path = os.path.join(plot_dir, 'stats.csv')
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(statistics)
                statistics.remove('id')
                for j, id in enumerate(ids):
                    auc = aucs[j]
                    row = [id, auc]
                    for statistic in statistics:
                        row.append(statistic[j][best_threshold_on_training_set_index])
                    writer.writerow(row)
        else:
            print('No matching Class found')

    else:
        print(gen_path, ' NO GEN FOUND')