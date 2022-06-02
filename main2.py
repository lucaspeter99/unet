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
from torch.utils.tensorboard import SummaryWriter
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

#print uitlized GPU memory
def gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return(f"GPU memory occupied: {info.used//1024**2} MB.")

#ignore warnings
warnings.filterwarnings("ignore")

# data_dir = './new_data'
# val_data_dir = './val_data'
data_dir = './train_data'
val_data_dir = './val_data'
results_dir = './results'

#dataset classes defined in datasets.py for easier access
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


#runtime variables
tag = str(sys.argv[1]) if len(sys.argv) > 1 else 'DELETE'  # project name tag
dc_key = int(sys.argv[2]) if len(sys.argv) > 2 else 5  # dictionary key to choose dataset class
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0001  # learning rate
num_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 75
mse = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

batchnorm = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True  # wether or not to use batchnorm
leaky = bool(int(sys.argv[7])) if len(sys.argv) > 7 else True  # whether to use leaky ReLU
max_pool = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True  # if True use MaxPooling, else AvgPooling

start_data = int(sys.argv[9]) if len(sys.argv) > 9 else -30
use_medical_data = bool(int(sys.argv[10])) if len(sys.argv) > 10 else True
split_med_channels = bool(int(sys.argv[11])) if len(sys.argv) > 11 else False
only_split_tici = bool(int(sys.argv[12])) if len(sys.argv) > 12 else False
split_modalities = bool(int(sys.argv[13])) if len(sys.argv) > 13 else False

windowing = bool(int(sys.argv[14])) if len(sys.argv) > 14 else False
use_gpu = bool(int(sys.argv[15])) if len(sys.argv) > 15 else True

dropout = bool(int(sys.argv[16])) if len(sys.argv) > 16 else False
batch_size = int(sys.argv[17]) if len(sys.argv) > 17 else 5

#set variables
deform = False
rotate = False
alpha = 0.2
beta = 0.1
sigma = 0.1

sgd = False
onlyRotDefSpecials = False
lr_decay = False
lr_min = 0

epoch_sample_interval = 5
weight_decay = 0.0
workers=8
beta1 = 0.99
val_size = 100


#create dataset
DatasetClass = dataset_classes[dc_key]
data = DatasetClass(data_dir, rotate=rotate, deform=deform, alpha=alpha, sigma=sigma, onlyRotDefSpecials=True,
                        use_medical_data = use_medical_data,
                       start = start_data
                       )


ml_per_voxel = (numpy.prod(numpy.array(data.dimensionality)) / 1000) / (pow(data.size, 3))
print("ML PER VOXEL", ml_per_voxel)


#split dataset with torch.Subset, for that need indices that are contained in each subset
if use_medical_data: #can only split be reca if med data is used
    training_ids = []
    validation_ids = []

    reca_statuses = []

    #get reca statuses for all patients
    for i, item in enumerate(data.items):
        reca_statuses.append(item[0][0,0,0,0].item())

    #indices are determined by stratified split based on reca status
    x_train, x_test, y_train, y_test = model_selection.train_test_split(list(range(len(data))), reca_statuses, shuffle= True, test_size=(1/6),stratify=reca_statuses)

    print("reca 1 on train set: ",sum(y_train))
    print("reca 1 on test set: ",sum(y_test))

    dataset = Subset(data, x_train) # training set
    val_dataset = Subset(data, x_test) # test set
else: #if no med data is used, split at predermined point
    split_at = int(len(data)*5/6)
    dataset = Subset(data, list(range(split_at)))
    val_dataset = Subset(data, list(range(split_at, len(data))))

#set device torch uses for computation
if use_gpu:
    device = torch.device('cuda') # gpu
else:
    device = torch.device('cpu') # cpu

criterion = MSELoss() if mse else BCELoss() # loss function
in_channels = data.in_channels # amount of modalities
out_channels = data.out_channels # amount of gts, should always be 1
features = data.features # image size

#create model, architechture is based on boolean variables set above
if use_medical_data and split_modalities and split_med_channels and only_split_tici: # all imaging seperate, reca seperate, rest of med data combined
    gen = UNetSepMedDataCombinedRecaSepCheckpoint(in_channels=in_channels,
                        out_channels=out_channels,
                        features=features,
                        batchnorm=batchnorm,
                        leaky=leaky,
                        max_pool=max_pool,
                        dropout = dropout).to(device)

elif use_medical_data and split_modalities and split_med_channels and not only_split_tici: # all imaging seperate, all med data seperate
    gen = UNetSepMedDataCheckpoint(in_channels=in_channels,
                        out_channels=out_channels,
                        features=features,
                        batchnorm=batchnorm,
                        leaky=leaky,
                        max_pool=max_pool).to(device)

elif use_medical_data and split_modalities and not split_med_channels: # all imaging seperate, all med data in one pathway
    gen = UNetSepMedDataCombinedCheckpoint(in_channels=in_channels,
                        out_channels=out_channels,
                        features=features,
                        batchnorm=batchnorm,
                        leaky=leaky,
                        max_pool=max_pool).to(device)

elif use_medical_data and not split_modalities and not split_med_channels: # all imaging combined in on pathway, all med data combined in on pathway
    gen = UNetSmallSepMedDataCheckpoint(in_channels=in_channels,
                        out_channels=out_channels,
                        features=features,
                        batchnorm=batchnorm,
                        leaky=leaky,
                        max_pool=max_pool).to(device)

elif not use_medical_data and split_modalities: # all imaging seperate, no med data
    gen = UNetSepCheckpoint(in_channels=in_channels,
                        out_channels=out_channels,
                        features=features,
                        batchnorm=batchnorm,
                        leaky=leaky,
                        max_pool=max_pool).to(device)

elif not use_medical_data and not split_modalities: # imaging in one pathway
    gen = UNetSmallCheckpoint(in_channels=in_channels,
                        out_channels=out_channels,
                        features=features,
                        batchnorm=batchnorm,
                        leaky=leaky,
                        max_pool=max_pool).to(device)

else: # everything else that is not implemented
    print("PARAMETERS WRONG")
    gen = UNetSmallCheckpoint(in_channels=in_channels,
                        out_channels=out_channels,
                        features=features,
                        batchnorm=batchnorm,
                        leaky=leaky,
                        max_pool=max_pool).to(device)
if use_gpu:
    print(gpu_utilization())

#set optimizer used to update weights within the model
if sgd:
    opt_gen = SGD(gen.parameters(), lr=lr, weight_decay=weight_decay)
else:
    opt_gen = Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)

#create loaders for the datasets
#these loaders are efficient (work in parallel with num_workers) and create batches
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

#create directories etc
date = datetime.now().strftime('%Y-%m-%d')
name = f'{tag}_{date}_{num_epochs}_bs{batch_size}_lr{lr}{"_DecayTo_" if lr_decay else ""}{lr_min if lr_decay else ""}'
name += f'_{gen.__class__.__name__}_{opt_gen.__class__.__name__}_{criterion.__class__.__name__}_{data.__class__.__name__}{"_windowed" if windowing else ""}'
name += f'{"_rot" if rotate else ""}{"_def" if deform else ""}{"_RotDefSpecialsOnly" if onlyRotDefSpecials else ""}'
name += f'{"_medData" if use_medical_data else ""}{"Split" if split_med_channels else ""}'
results_dir = os.path.join(results_dir, name)
plot_dir = os.path.join(results_dir, 'plots')
train_plot_dir = os.path.join(plot_dir, 'training')
val_plot_dir = os.path.join(plot_dir, 'validation')
output_dir = os.path.join(results_dir, 'outputs')
y0_output_dir = os.path.join(output_dir, "y0")
y1_output_dir = os.path.join(output_dir, "y1")
prediction_output_dir = os.path.join(output_dir, "prediction")
gt_output_dir = os.path.join(output_dir, "gt")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(train_plot_dir, exist_ok=True)
os.makedirs(val_plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(y0_output_dir, exist_ok=True)
os.makedirs(y1_output_dir, exist_ok=True)
os.makedirs(prediction_output_dir, exist_ok=True)
os.makedirs(gt_output_dir, exist_ok=True)


#create log giles
log_file = os.path.join(results_dir, '%s_log' % name)
state_file = os.path.join(results_dir, 'state')
epoch_file = os.path.join(results_dir, 'epoch_%02d')

# logging
logger = utils.LeanLogger(name=log_file)


if torch.multiprocessing.current_process().name == 'MainProcess':
    writer = SummaryWriter(results_dir, comment=name)  # tensorboard writer

#log information at the start of the training process
def startup_logs():
    logger.log('\n== Model ==')
    logger.log(f'Medical Data: {use_medical_data}')
    logger.log(f'Batchnorm: {batchnorm}')
    logger.log(f'Leaky: {leaky}')
    logger.log(f'Max Pooling: {max_pool}')
    logger.log('\n== Data ==')
    logger.log(f'Dataset: {dataset.__class__.__name__}')
    logger.log(f'Deform: {deform}')
    logger.log(f'Rotate: {rotate}')
    logger.log(f'alpha: {alpha}')
    logger.log(f'sigma: {sigma}')
    logger.log(f'Training Dataset Size: {len(dataset)}')
    logger.log(f'Validation Dataset Size: {len(val_dataset)}')
    logger.log('\n== Routine ==')
    logger.log(f'Optimizer: {opt_gen.__class__.__name__}')
    logger.log(f'Learning Rate: {lr}')
    logger.log(f'Learning Rate Decay: {lr_decay}')
    logger.log(f'Weight decay: {weight_decay}')
    logger.log(f'Loss: {criterion.__class__.__name__}')
    logger.log(f'Epochs: {num_epochs}')
    logger.log(f'Batch size: {batch_size}')
    if use_gpu:
        logger.log('\n== Hardware ==')
        logger.log(f'Device: {str(device)}')
        logger.log(f'Device name:{torch.cuda.get_device_name()}')
        logger.log(f'Device count:{torch.cuda.device_count()}')
        logger.log(f'Current device:{torch.cuda.current_device()}')
        logger.log(f'Capability: {torch.cuda.get_device_capability()}')

#compute metrics and save plots for all items in data_obj

def compute_metrics(data_obj, key, generator, metrics, plot_dir, thresholds, ml_per_voxel , best_threshold = None):
    plot_dir = os.path.join(plot_dir, key)
    print("Plotting for ", key, " set")
    for l, data in enumerate(data_obj): # run thru dataset
        print("[",l+1,"/", len(data_obj),"]")
        x, y, patientid = data #item in dataset
        x = x.float().to(device)
        #get binary ground truth
        y = (y > 0.5).float().to(device)
        y = y.cpu().detach() #move tensor to cpu
        #predict with model
        score = generator(x.unsqueeze(0))
        score = score.cpu().detach() #move tensor to cpu


        #------------------------------
        #save prediction, gt with nib
        #------------------------------
        img = nib.Nifti1Image(score[0, 0, :, :, :].detach().numpy(), numpy.eye(4))
        nib.save(img, os.path.join(prediction_output_dir, str(patientid) + ".nii.gz"))

        img = nib.Nifti1Image(y[0, :, :, :].detach().numpy(), numpy.eye(4))
        nib.save(img, os.path.join(gt_output_dir ,str(patientid) + ".nii.gz"))

        del img
        #plot comparison output to gt
        utils.plot_comparison(score=score, target=y, item=patientid, plot_dir=plot_dir)

        #plot mismatches
        if use_medical_data:
            # load xs with 0 and 1 in tici scale pos
            x = x.squeeze(0)
            x0 = torch.full((1, 48, 48, 48),0).to(device)
            for i in range(9):
                x0 = torch.cat((x0, x[i + 1].unsqueeze(0)), 0)


            x1 = torch.full((1, 48, 48, 48),1).to(device)
            for i in range(9):
                x1 = torch.cat((x1, x[i + 1].unsqueeze(0)), 0)

            # predict
            y0 = generator(x0.unsqueeze(0))
            y0 = y0.cpu().detach()

            y1 = generator(x1.unsqueeze(0))
            y1 = y1.cpu().detach()

            #plot mismatch between predictions
            utils.plot_mismatch(y0, y1, patientid, plot_dir)
            # ------------------------------
            # save y0, y1
            # ------------------------------
            img = nib.Nifti1Image(y0[0, 0, :, :, :].detach().numpy(), numpy.eye(4))
            nib.save(img, os.path.join(y0_output_dir, str(patientid) + ".nii.gz"))

            img = nib.Nifti1Image(y1[0, 0, :, :, :].detach().numpy(), numpy.eye(4))
            nib.save(img, os.path.join(y1_output_dir, str(patientid) + ".nii.gz"))

            del img




        try:
            #save metrics
            tpr,  auc, dices, diffs, vol_diffs, accuracies, precisions, recalls, fprs, f1s, gt_vol = utils.plot_metrics(score=score, target=y, item=patientid, plot_dir=plot_dir, thresholds=thresholds, ml_per_voxel=ml_per_voxel)
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

        del x, y, score


    #save all statistics for different thresholds
    statistics = [
            'dice',
            'diff',
            'vol_diff',
            'accuracy',
            'precision',
            'recall',
            'fpr',
            'f1',]
    thresholds.insert(0, 'id')
    for statistic in statistics:
        csv_path = os.path.join(plot_dir, statistic + '.csv') # filepath
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file) #write file
            writer.writerow(thresholds) # write header
            for j, stat_list in enumerate(metrics[key][statistic]): # write rest of the file
                id = metrics[key]['id'][j]
                stat_list_dummy = stat_list.copy()
                stat_list_dummy.insert(0, id)
                writer.writerow(stat_list_dummy)
    thresholds.remove('id')

    #save aucs
    with open(os.path.join(plot_dir, 'auc.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'auc'])
        for j, auc in enumerate(metrics[key]['auc']):
            id = metrics[key]['id'][j]
            writer.writerow([id, auc])

    if key == 'validation':
        #save plots for average statistics on different thresholds
        for statistic in statistics:
            #get average for different tresholds
            averages = []

            for i, threshold in enumerate(thresholds):
                statistic_for_threshold = []
                for stat_list in metrics[key][statistic]:
                    statistic_for_threshold.append(stat_list[i])
                averages.append(sum(statistic_for_threshold) / len(statistic_for_threshold))



            #plot the average
            utils.plot_average_stats_for_different_thesholds(thresholds, averages, plot_dir, name = statistic, best_threshold=best_threshold)

        statistics.insert(0, 'gt_vol')
        statistics.insert(0, 'auc')
        statistics.insert(0, 'id')
        csv_path = os.path.join(plot_dir, 'stats.csv')

        #for val data set metrics only have to be saved for the best threshold
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
        logger.log(f"Mean VAL AUC:{auc_mean}")
    fig.savefig(os.path.join(plot_dir, 'roc_means'))
    plt.close()


#actual training routine
def train_unet():
    startup_logs()
    start_time = datetime.now()
    n_iter = 0
    n_img = 0
    if lr_decay: #learning rate decays with time, not currently used
        gamma = (lr_min/lr)**(1/float(num_epochs-1)) # fit function so for x = 0 y = lr, x = num_epochs-1 y = lr_min
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer= opt_gen,gamma = gamma) # multiplicative lr decay by gamma until last_epoch

    #for every epoch (how often the dataset is shown to the model):
    for epoch in range(num_epochs):
        # for each batch in the dataloader ( = iteration):
        for i, data in enumerate(train_loader):
            x, y, _ = data # data, id of patient is not needed


            x = x.float().to(device) # input modalities
            y = (y > 0.5).float().to(device) # ground truth
            if use_gpu: # to check performance
                print(gpu_utilization())
                torch.cuda.empty_cache() # clear data from cache that is not used

            output = gen(x) #prediction of the model for x

            print("prediction", torch.sum(output.flatten()))    # just to make sure that the output of the model still makes sense
                                                                # in some cases the model tends to output zero for all inputs. This is due to
                                                                    # 1) wrong implementation of the architecture
                                                                    # 2) interaction between batchnorm layers and checkpointing, as two forward passes messes with the momentum term

            # dotg = make_dot(output, show_attrs=False)
            # dotg.render('layout', view=True)

            loss_gen = criterion(output, y) # compute loss

            auc = utils.auc(y, output) # compute auc
            if use_gpu: #empty cache again to make sure
                torch.cuda.empty_cache()

            #set gradients to zero
            opt_gen.zero_grad()

            #log loss, auc
            if torch.multiprocessing.current_process().name == 'MainProcess':
                writer.add_scalar('loss_gen', loss_gen, n_iter)
                writer.add_scalar('auc', auc, n_iter)
                writer.add_scalar('gen_lr', opt_gen.param_groups[0]['lr'])


            loss_gen.backward()# calculate gradients for generator
            opt_gen.step()#update weights
            #get times
            duration = datetime.now() - start_time
            #log stats
            logger.log('----------------------------------------------------')
            logger.log('[%d/%d][%d/%d]\tloss_gen: %.4f\tauc: %.4f'
                       % (epoch + 1,
                          num_epochs,
                          i + 1,
                          len(train_loader),
                          loss_gen.item(),
                          auc
                       ))
            if use_gpu:
                logger.log('%s\tTime elapsed: %s' % (gpu_utilization(),str(duration)[:-4])) #log some more stuff
            n_iter += 1
            # delete data to clear space
            del x,y,output
            if use_gpu:
                torch.cuda.empty_cache()
            gc.collect()

        if lr_decay:
            lr_scheduler.step()

    #training is done
    #save the model
    torch.save(gen, os.path.join(results_dir, 'gen.trc'))

    #metrics will be saved in here
    metrics = {
        'training': {
            'tpr': [],
            'auc': [],
            'dice': [],
            'diff': [],
            'vol_diff': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'fpr': [],
            'f1':[],
            'id': [],
            'reca': [],
            'gt_vol': [],
        },
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
            'f1':[],
            'id': [],
            'reca': [],
            'gt_vol': [],
        }
    }
    possible_thresholds = list([round(i * 0.02, 2) for i in range(50)]) # thresholds that are considered



    compute_metrics(data_obj=dataset, key='training', generator=gen, metrics=metrics, plot_dir=plot_dir,ml_per_voxel= ml_per_voxel , thresholds=possible_thresholds.copy()) # get metrics, plot for those thresholds on training set
    #get best threshold
    print("Determining best threshold from training set...")
    df = pd.read_csv(os.path.join(plot_dir, 'training', 'vol_diff.csv'),
                     index_col='id')#read csv file
    best_mean = sys.maxsize #best mean vol diff, since it is being compared set to a large value in the beginning
    best_threshold_on_training_set = 0 #best threshold, set to 0 at the start
    for threshold in possible_thresholds: #for all thresholds
        stat_list = df[str(threshold)].tolist() # get list of metric of all patients for that threshold
        mean_stat = numpy.nanmean(numpy.array(stat_list)) # get mean of that list
        if mean_stat < best_mean: # if mean is better update best mean and best threshold
            best_mean = mean_stat
            best_threshold_on_training_set = threshold
    print("Best threshold is ", best_threshold_on_training_set)
    best_threshold_on_training_set_index = possible_thresholds.index(float(best_threshold_on_training_set))
    #compute metrics with best threshold on test set
    compute_metrics(data_obj=val_dataset, key='validation', generator=gen, metrics=metrics, plot_dir=plot_dir, thresholds=possible_thresholds, best_threshold=float(best_threshold_on_training_set))
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as file:
        json.dump(metrics, file)

    #finishing with some logging
    plot_metrics(metrics, 'training', plot_dir)
    plot_metrics(metrics, 'validation', plot_dir, best_threshold_on_training_set_index)

    end_time = datetime.now()
    duration = end_time - start_time
    logger.log(f'Finished training at {end_time.strftime("%H:%M:%S")}, duration: {str(duration)[:-4]}')
    logger.log('Generator saved')


#call the actual routine
train_unet()
