import pickle
import csv
import os
import torch
from os.path import isfile
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy


def view_data(data, text=None,  file=None):
    plt.imshow(data, cmap='Greys')
    if text:
        plt.text(0, -1, text)
    if file:
        plt.savefig(file)
    plt.show()


def save_file(data, path, write_bytes=True):
    if write_bytes:
        path = path if os.path.splitext(path)[1] else path + '.pkl'
        with open(path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        path = path if os.path.splitext(path)[1] else path + '.txt'
        with open(path, 'w') as file:

            file.write(data)


def load_file(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except pickle.UnpicklingError:
        with open(path, 'r') as file:
            return file.read()


def dice_coefficient(score, target):
    score = score if type(score) is torch.Tensor else torch.tensor(score)
    target = target if type(target) is torch.Tensor else torch.tensor(target)
    overlap = ((target > 0.5) == (score > 0.5)).float().sum() * 2
    numel = target.numel() + score.numel()
    return (overlap / numel).item()


def vnet_dice_coefficient(input, target):  # 1 if inputs are identical
    input = (input > 0.5).float()
    target = (target > 0.5).float()
    s = input.shape[-1]  # spatial dimension
    input = input.reshape(s, s, s)
    target = target.reshape(s, s, s)

    numerator = torch.mul(input, target)
    numerator = torch.sum(numerator)
    numerator *= 2

    input2 = torch.pow(input, torch.tensor(2))
    target2 = torch.pow(target, torch.tensor(2))

    input2 = torch.sum(input2)
    target2 = torch.sum(target2)

    denominator = input2 + target2

    return torch.div(numerator, denominator).item()


def plot_comparison(score, target, item, plot_dir, cmap='Greys_r'):
    fig = plt.figure(figsize=[5.0, 16.0])

    grounds = []
    comps = []
    gens = []

    for i in range(10):
        grounds.append(plt.subplot2grid((10, 3), (i, 0)))
        comps.append(plt.subplot2grid((10, 3), (i, 1)))
        gens.append(plt.subplot2grid((10,3), (i, 2)))
    for i, it in enumerate(grounds):
        it.imshow(target[0, 12 + (2*i), :, :], cmap=cmap, interpolation='none')
        gens[i].imshow(score[0, 0, 12 + (2*i), :, :], cmap=cmap, vmin = 0, vmax = 1, interpolation='none')
        comps[i].imshow(score[0, 0, 12 + (2*i), :, :], cmap=cmap, vmin = 0, vmax = 1, interpolation='none')
        comps[i].imshow(target[0, 12 + (2*i), :, :], cmap='cividis', alpha = 0.4, interpolation='none')

    # ground1 = plt.subplot2grid((3, 4), (0,0))
    # ground2 = plt.subplot2grid((3, 4), (0,1))
    # ground3 = plt.subplot2grid((3, 4), (0,2))
    # ground4 = plt.subplot2grid((3, 4), (0,3))
    #
    # comp1 = plt.subplot2grid((3, 4), (1,0))
    # comp2 = plt.subplot2grid((3, 4), (1,1))
    # comp3 = plt.subplot2grid((3, 4), (1,2))
    # comp4 = plt.subplot2grid((3, 4), (1,3))
    #
    # gen1 = plt.subplot2grid((3, 4), (2,0))
    # gen2 = plt.subplot2grid((3, 4), (2,1))
    # gen3 = plt.subplot2grid((3, 4), (2,2))
    # gen4 = plt.subplot2grid((3, 4), (2,3))
    #
    #
    #
    # ground1.imshow(target[0, 20, :, :], cmap=cmap)
    # ground2.imshow(target[0, 23, :, :], cmap=cmap)
    # ground3.imshow(target[0, 26, :, :], cmap=cmap)
    # ground4.imshow(target[0, 29, :, :], cmap=cmap)
    #
    # gen1.imshow(score[0, 0, 20, :, :], cmap=cmap, vmin = 0, vmax = 1)
    # gen2.imshow(score[0, 0, 23, :, :], cmap=cmap, vmin = 0, vmax = 1)
    # gen3.imshow(score[0, 0, 26, :, :], cmap=cmap, vmin = 0, vmax = 1)
    # gen4.imshow(score[0, 0, 29, :, :], cmap=cmap, vmin = 0, vmax = 1)

    # fpr, tpr, thresholds = metrics.roc_curve(target.flatten(), score.flatten(), drop_intermediate=True)
    # fpr_sorted = sorted(range(len(fpr)), key=lambda k: fpr[k])
    # tpr_sorted = sorted(range(len(tpr)), key=lambda k: tpr[k], reverse=True)
    # metric = numpy.array([0]*len(thresholds))
    # for i, fp in enumerate(fpr_sorted):
    #     for j, tp in enumerate(tpr_sorted):
    #         if fp == tp:
    #             metric[fp] = i + j
    # threshold = thresholds[metric.argmax()]
    # score = (score > threshold).float()
    # comp1.imshow(score[0, 0, 20, :, :], cmap=cmap)
    # comp2.imshow(score[0, 0, 23, :, :], cmap=cmap)
    # comp3.imshow(score[0, 0, 26, :, :], cmap=cmap)
    # comp4.imshow(score[0, 0, 29, :, :], cmap=cmap)


    fig.suptitle(f'groundtruth / overlay / probabilites {item}')

    plt.savefig(os.path.join(plot_dir, f'comp_ext_{item}.png'))
    plt.close(fig)


def plot_metrics(score, target, item, plot_dir):
    fig = plt.figure()
    sub_plot = fig.add_subplot(111)
    # target = (target > 0.5)
    score = score.detach() if type(score) == torch.Tensor else score
    fpr, tpr, thresholds = metrics.roc_curve(target.flatten(), score.flatten(), drop_intermediate=True)
    if numpy.isnan(tpr[0]):
        raise ValueError('target contains only zeros, can\'t compute ROC curve')
    x = numpy.arange(101) / 100
    tpr_interp = numpy.interp(x, fpr, tpr)  # interpolate roc-curve to get same length results
    auc = metrics.auc(x, tpr_interp)
    dice = vnet_dice_coefficient(score, target)
    fig.suptitle(f'AUC: {auc} / DICE: {dice}')
    sub_plot.plot(x, tpr_interp)
    plt.savefig(os.path.join(plot_dir, f'roc_{item}'))
    plt.close(fig)
    return list(tpr_interp), auc, dice


def auc(target, score):
    target = (target > 0.5).float()
    score = score.detach() if type(score) == torch.Tensor else score
    fpr, tpr, thresholds = metrics.roc_curve(target.flatten(), score.flatten())
    return metrics.auc(fpr, tpr)


def load_model(file, train=True):
    model = torch.load(file)
    model = model.cpu().train(train)
    return model


def parse_logfile(path):
    log = None
    disc_losses = []
    gen_losses = []
    with open(path, 'r') as file:
        log = file.readlines()
    for line in log:
        if line.startswith('['):
            _, disc, gen, time = line.split('\t')
            disc_losses.append(float(disc.split()[1]))
            gen_losses.append(float(gen.split()[1]))
    assert len(disc_losses) == len(gen_losses)
    return disc_losses, gen_losses


class LeanLogger:

    def __init__(self, name):
        self.log_file = name + '.txt'
        self.csv_file = name + '.csv'
        self.track('gen_losses', 'disc_losses')

    def log(self, data):
        data = str(data)
        print(data)
        data = data if data.startswith('\n') or not isfile(self.log_file) else '\n' + data
        with open(self.log_file, 'a+') as file:
            file.write(data)

    def track(self, disc_loss, gen_loss):
        with open(self.csv_file, 'a+', newline='') as file:
            writer = csv.writer(file, dialect='excel')
            writer.writerow([gen_loss, disc_loss])


