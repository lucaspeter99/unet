import json
import math
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
from torch.optim import Adam, SGD

data_dir = './data'
results_dir = './results'

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


tag = str(sys.argv[1]) if len(sys.argv) > 1 else 'BS'  # project name tag
dc_key = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # dictionary key to choose dataset class
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01  # learning rate
lr_decay = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False # wether or not to decrease lr after each epoch
lr_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.001 # min value for the lr
num_epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 10
mse = bool(int(sys.argv[7])) if len(sys.argv) > 7 else True
onlyRotDefSpecials = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True # only deform/ rotate a subset of data, overrides deform and rotate for the special cases

batchnorm = bool(int(sys.argv[9])) if len(sys.argv) > 9 else True  # wether or not to use batchnorm
leaky = bool(int(sys.argv[10])) if len(sys.argv) > 10 else True  # whether to use leaky ReLU
max_pool = bool(int(sys.argv[11])) if len(sys.argv) > 11 else True  # if True use MaxPooling, else AvgPooling
sgd = bool(int(sys.argv[12])) if len(sys.argv) > 12 else False  # if True use SGD optimizer, else Adam optimizer

deform = bool(int(sys.argv[13])) if len(sys.argv) > 13 else False  # data augmentation: use elastic deformation on the images
rotate = bool(int(sys.argv[14])) if len(sys.argv) > 14 else False  # data augmentation: rotate x and y of the input tensor
alpha = float(sys.argv[15]) if len(sys.argv) > 15 else 0.2  # alpha parameter for elastic deformation on data
sigma = float(sys.argv[16]) if len(sys.argv) > 16 else 0.1  # sigma parameter for elastic deformation on data

epoch_sample_interval = 5
weight_decay = 0.0
batch_size=10
workers=0
beta1 = 0.99

DatasetClass = dataset_classes[dc_key]

nans = [1,8,14,15,28,51,77,99,112] # subset of the dataset: all of the data that isnt supposed to predict a lesion at all

dataset = DatasetClass(data_dir, rotate=rotate, deform=deform, alpha=alpha, sigma=sigma, onlyRotDefSpecials=True, specials=nans)

device = torch.device('cpu')

criterion = MSELoss() if mse else BCELoss()
in_channels = dataset.in_channels
out_channels = dataset.out_channels
features = dataset.features




gen = UNet(in_channels=in_channels,
           out_channels=out_channels,
           features=features,
           batchnorm=batchnorm,
           leaky=leaky,
           max_pool=max_pool).to(device)

if sgd:
    opt_gen = SGD(gen.parameters(), lr=lr, weight_decay=weight_decay)
else:
    opt_gen = Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)


data_len = len(dataset)
print('LEN DATASET: ', data_len)

train_data = Subset(dataset, list(range(data_len-17)))
val_data = Subset(dataset, list(range(data_len-17, data_len)))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers)

date = datetime.now().strftime('%Y-%m-%d')
name = f'{tag}_{date}_{num_epochs}_bs{batch_size}_lr{lr}{"_DecayTo_" if lr_decay else ""}{lr_min if lr_decay else ""}'
name += f'_{gen.__class__.__name__}_{opt_gen.__class__.__name__}_{criterion.__class__.__name__}_{dataset.__class__.__name__}'
name += f'{"_rot" if rotate else ""}{"_def" if deform else ""}{"_RotDefSpecialsOnly" if onlyRotDefSpecials else ""}'
results_dir = os.path.join(results_dir, name)
plot_dir = os.path.join(results_dir, 'plots')
train_plot_dir = os.path.join(plot_dir, 'training')
val_plot_dir = os.path.join(plot_dir, 'validation')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(train_plot_dir, exist_ok=True)
    os.makedirs(val_plot_dir, exist_ok=True)


log_file = os.path.join(results_dir, '%s_log' % name)
state_file = os.path.join(results_dir, 'state')
epoch_file = os.path.join(results_dir, 'epoch_%02d')

# logging
logger = utils.LeanLogger(name=log_file)


if torch.multiprocessing.current_process().name == 'MainProcess':
    writer = SummaryWriter(results_dir, comment=name)  # tensorboard writer


def startup_logs():
    # logger.log(f'Job: {tag}')
    # logger.log(f'Hyperparameters: {hyperparams_msg}')
    logger.log('\n== Model ==')
    logger.log(f'Batchnorm: {batchnorm}')
    logger.log(f'Leaky: {leaky}')
    logger.log(f'Max Pooling: {max_pool}')
    logger.log('\n== Data ==')
    logger.log(f'Dataset: {dataset.__class__.__name__}')
    logger.log(f'Deform: {deform}')
    logger.log(f'Rotate: {rotate}')
    logger.log(f'alpha: {alpha}')
    logger.log(f'sigma: {sigma}')
    logger.log(f'Training Dataset Size: {len(train_data)}')
    logger.log(f'Validation Dataset Size: {len(val_data)}')
    logger.log('\n== Routine ==')
    # logger.log(f'GAN: {gan}')
    logger.log(f'Optimizer: {opt_gen.__class__.__name__}')
    logger.log(f'Learning Rate: {lr}')
    logger.log(f'Learning Rate Decay: {lr_decay}')
    logger.log(f'Weight decay: {weight_decay}')
    logger.log(f'Loss: {criterion.__class__.__name__}')
    logger.log(f'Epochs: {num_epochs}')
    logger.log(f'Batch size: {batch_size}')
    # logger.log(f'Random Seed: {seed}')
    logger.log('\n== Hardware ==')
    # logger.log(f'Device: {str(device)}')
    # logger.log(f'Device name:{torch.cuda.get_device_name()}')
    # logger.log(f'Device count:{torch.cuda.device_count()}')
    # logger.log(f'Current device:{torch.cuda.current_device()}')
    # logger.log(f'Capability: {torch.cuda.get_device_capability()}')
    # logger.log(f'\nStarted {"GAN" if gan else "UNet"} training at {start_time.strftime("%H:%M:%S")}')

def compute_metrics(data_obj, key, generator, metrics, plot_dir):
    plot_dir = os.path.join(plot_dir, key)
    for i, data in enumerate(data_obj):
        x, y = data
        x = x.float().to(device)
        y = (y > 0.5).float().to(device)
        score = generator(x.unsqueeze(0))
        score = score.cpu().detach()
        y = y.cpu().detach()
        utils.plot_comparison(score=score, target=y, item=i, plot_dir=plot_dir)
        try:
            tpr, auc, dice = utils.plot_metrics(score=score, target=y, item=i, plot_dir=plot_dir)
            metrics[key]['tpr'].append(tpr)
            if not numpy.isnan(auc):
                metrics[key]['auc'].append(auc)
            if not numpy.isnan(dice):
                metrics[key]['dice'].append(dice)
            del tpr, auc, dice
        except ValueError as ve:
            print(f'{ve.__class__.__name__} in utils.plot_roc: {ve}')
        del x, y, score


def plot_metrics(metrics, key, plot_dir, border_color='black'):
    plot_dir = os.path.join(plot_dir, key)
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
        logger.log(f"Mean VAL AUC:{auc_mean}")
    dice_mean = numpy.array(metrics[key]['dice']).mean()
    plt.suptitle(f'mean AUC: {auc_mean} / mean DICE: {dice_mean}')
    plt.savefig(os.path.join(plot_dir, 'roc_means'))
    plt.close()

def train_unet():
    startup_logs()
    start_time = datetime.now()
    n_iter = 0
    n_img = 0
    if lr_decay:
        gamma = (lr_min/lr)**(1/float(num_epochs-1)) # fit function so for x = 0 y = lr, x = num_epochs-1 y = lr_min
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer= opt_gen,gamma = gamma) # multiplicative lr decay by gamma until last_epoch
    for epoch in range(num_epochs):
        # for each batch in the dataloader
        for i, data in enumerate(train_loader):
            x, y = data

            x = x.float().to(device)
            y = (y > 0.5).float().to(device)

            output = gen(x)
            # print(output.shape)
            #
            # dotg = make_dot(output, show_attrs=False)
            # dotg.render('layout', view=True)

            loss_gen = criterion(output, y)

            auc = utils.auc(y.cpu(), output.cpu())

            # update generator network
            opt_gen.zero_grad()

            if torch.multiprocessing.current_process().name == 'MainProcess':
                writer.add_scalar('loss_gen', loss_gen, n_iter)
                writer.add_scalar('auc', auc, n_iter)
                writer.add_scalar('gen_lr', opt_gen.param_groups[0]['lr'])

            # calculate gradients for generator
            loss_gen.backward()
            opt_gen.step()
            duration = datetime.now() - start_time
            logger.log('[%d/%d][%d/%d]\tloss_gen: %.4f\tlr: %.4f\tauc: %.4f\tTime elapsed: %s'
                       % (epoch + 1,
                          num_epochs,
                          i + 1,
                          len(train_loader),
                          loss_gen.item(),
                          lr_scheduler.get_last_lr()[0] if lr_decay else lr,
                          auc,
                          str(duration)[:-4])
                       )

            n_iter += 1
            if (epoch % epoch_sample_interval == 0 or epoch == num_epochs - 1) and i == len(train_loader) - 1:
                if torch.multiprocessing.current_process().name == 'MainProcess':
                    writer.add_image('generated_X-Y', output[0, :, :, :, 23], n_img)
                    writer.add_image('generated_X-Z', output[0, :, :, 23, :], n_img)
                    writer.add_image('generated_Y-Z', output[0, :, 23, :, :], n_img)

                    writer.add_image('groundtruth_X-Y', y[0, :, :, :, 23], n_img)
                    writer.add_image('groundtruth_X-Z', y[0, :, :, 23, :], n_img)
                    writer.add_image('groundtruth_Y-Z', y[0, :, 23, :, :], n_img)

                    n_img += 1
                del x, y

        for xv, yv in val_loader:  # cross validation
            xv = xv.float().to(device)
            yv = yv.float().to(device)
            output = gen(xv)
            validation_loss = criterion(output, yv).item()
            if torch.multiprocessing.current_process().name == 'MainProcess':
                writer.add_scalar('validation_loss', validation_loss, n_iter)
            logger.log(f'validation_loss: {validation_loss}')
            gen.zero_grad()

            del xv, yv
        if lr_decay:
            lr_scheduler.step()

    metrics = {
        'training': {
            'tpr': [],
            'auc': [],
            'dice': []
        },
        'validation': {
            'tpr': [],
            'auc': [],
            'dice': []
        }
    }
    compute_metrics(data_obj=train_data, key='training', generator=gen, metrics=metrics, plot_dir=plot_dir)
    compute_metrics(data_obj=val_data, key='validation', generator=gen, metrics=metrics, plot_dir=plot_dir)
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as file:
        json.dump(metrics, file)
    plot_metrics(metrics, 'training', plot_dir)
    plot_metrics(metrics, 'validation', plot_dir)

    end_time = datetime.now()
    duration = end_time - start_time
    logger.log(f'Finished training at {end_time.strftime("%H:%M:%S")}, duration: {str(duration)[:-4]}')
    torch.save(gen, os.path.join(results_dir, 'gen.trc'))
    logger.log('Generator saved')

train_unet()
