import pickle
import csv
import os
import torch
from os.path import isfile
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy
import csv


#deprecated
def view_data(data, text=None,  file=None):
    plt.imshow(data, cmap='Greys')
    if text:
        plt.text(0, -1, text)
    if file:
        plt.savefig(file)
    plt.show()

#deprecated
def save_file(data, path, write_bytes=True):
    if write_bytes:
        path = path if os.path.splitext(path)[1] else path + '.pkl'
        with open(path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        path = path if os.path.splitext(path)[1] else path + '.txt'
        with open(path, 'w') as file:

            file.write(data)

#deprecated
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


def plot_mismatch(y0, y1, item, plot_dir, center, cmap='viridis'):
    fig = plt.figure(figsize=[5.0, 16.0])



    y0s = []
    mismatches = []
    y1s = []
    diff = torch.sub(y1, y0) # erfolgreich - nicht erfolgreich
    avg_diff = diff.mean()
    min_diff = torch.min(diff).item()
    max_diff = torch.max(diff).item()

    for i in range(10):
        y0s.append(plt.subplot2grid((10, 3), (i, 0)))
        mismatches.append(plt.subplot2grid((10, 3), (i, 1)))
        y1s.append(plt.subplot2grid((10,3), (i, 2)))

    for i, it in enumerate(y0s):
        it.imshow(y0[0, 0, :,int(center[1]) + (2*i) - 10, :], cmap='Greys_r', vmin = 0, vmax = 1, interpolation='none')
        mismatches[i].imshow(diff[0, 0, :,int(center[1]) + (2*i) - 10, :], vmin = -0.5, vmax = 0.5,cmap=cmap, interpolation='none')
        y1s[i].imshow(y1[0, 0, :,int(center[1]) + (2*i) - 10, :], cmap='Greys_r', vmin = 0, vmax = 1, interpolation='none')

    #mismatch_sum = torch.sum(torch.abs(diff), (0,1,2,3,4)).item()

    fig.suptitle(f'{item}\navg: {avg_diff}\nmin diff: {min_diff} / max diff: {max_diff}\nreca0 / reca1 - reca0 / reca1')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, f'mismatch_{item}.png'))
    plt.close(fig)

def plot_comparison(score, target, item, plot_dir, background, cmap='Greys_r'):
    fig = plt.figure(figsize=[5.0, 16.0])

    grounds = []
    comps = []
    gens = []
    center = get_center_of_mass(target[0,:,:,:])

    if center[0] > 37: # avoid out of bounds error
        center = (37, center[1], center[2])
    if center[0] < 10:
        center = (10, center[1], center[2])

    for i in range(10):
        grounds.append(plt.subplot2grid((10, 3), (i, 0)))
        comps.append(plt.subplot2grid((10, 3), (i, 1)))
        gens.append(plt.subplot2grid((10,3), (i, 2)))
    for i, it in enumerate(grounds):
        it.imshow(target[0, :, int(center[1]) + (2*i) - 10, :], cmap=cmap, interpolation='none')
        it.imshow(background[:, int(center[1]) + (2*i) - 10, :], cmap=cmap, interpolation='none', alpha = 0.4)
        gens[i].imshow(score[0, 0, :, int(center[1]) + (2*i) - 10, :], cmap=cmap, vmin = 0, vmax = 1, interpolation='none')
        comps[i].imshow(score[0, 0, :, int(center[1]) + (2*i) - 10, :], cmap=cmap, vmin = 0, vmax = 1, interpolation='none')
        comps[i].imshow(target[0, :, int(center[1]) + (2*i) - 10, :], cmap='cividis', alpha = 0.4, interpolation='none')



    fig.suptitle(f'groundtruth / overlay / probabilites {item}')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, f'comp_ext_{item}.png'))
    plt.close(fig)
    return center


def unet_dice_coefficient(input, target, thresholds):  # 1 if inputs are identical
    input = input.flatten()
    target = target.flatten()
    dices = []
    thr_target = (target > 0.5).float()
    for threshold in thresholds:
        thr_input = (input > threshold).float()

        numerator = torch.dot(thr_input, thr_target)
        numerator *= 2

        input2 = torch.dot(thr_input, thr_input)
        target2 = torch.dot(thr_target, thr_target)

        denominator = input2 + target2
        dice = torch.div(numerator, denominator).item()
        dices.append(dice)

    return dices
def get_center_of_mass(tensor):
    shape = tensor.shape
    m100 = 0
    m010 = 0
    m001 = 0
    m000 = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                mass = tensor[i,j,k].item()
                m100 += i * mass
                m010 += j * mass
                m001 += k * mass
                m000 += mass
    if m000 + m100 + m010 + m010 > 1:  # see if there actually are values in the tensor
        center = (m100 / m000, m010 / m000, m001 / m000)
        return center
    else:
        return (shape[0] / 2, shape[1] / 2, shape[2] / 2)

def unet_diffs(score, target, thresholds):
    diffs = []
    target = (target > 0.5).float().to('cpu')
    for threshold in thresholds:
        thr_score = (score > threshold).float().to('cpu')
        diff = torch.abs(torch.sub(target, thr_score))
        volume_diff = torch.sum(torch.abs(diff), (0, 1, 2, 3)).item()
        diffs.append(volume_diff)

    return diffs

def unet_vol_diffs(score, target, thresholds, ml_per_voxel):
    vol_diffs = []
    target = (target > 0.5).float().to('cpu')
    sum_target = torch.sum(target, (0,1,2,3)).item()
    for threshold in thresholds:
        thr_score = (score > threshold).float().to('cpu')
        thr_score_sum = torch.sum(thr_score, (0,1,2,3)).item()
        volume_diff = abs(sum_target - thr_score_sum)
        vol_diffs.append(volume_diff * ml_per_voxel)

    return vol_diffs

def unet_statistics(score, target, thresholds):
    accuracies = []
    precisions = []
    recalls = []
    fprs = []
    f1s = []

    target = (target > 0.5).float().flatten().to('cpu')
    for threshold in thresholds:
        thr_score = (score > threshold).float().flatten().to('cpu')
        #get True positives etc.
        confusion_matrix = metrics.confusion_matrix(target, thr_score)
        try:
            tn, fp, fn, tp = confusion_matrix.ravel()
        except:
            tn, fp, fn, tp = (0,0,0,0)

        try:
            accuracy = (tp+tn)/(tp+tn+fp+fn)
        except:
            accuracy = 0
        try:
            precision = (tp)/(tp+fp)
        except:
            precision = 0
        try:
            recall = (tp)/(tp+fn)
        except:
            recall = 0
        try:
            fpr = (fp)/(fp+tn)
        except:
            fpr = 0
        try:
            f1 = (2*precision*recall)/(precision+recall)
        except:
            f1 = 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
        f1s.append(f1)

    return accuracies, precisions, recalls, fprs, f1s

def plot_metrics(score, target, item, plot_dir, thresholds, ml_per_voxel):
    score = score.squeeze(0)
    fig = plt.figure()
    sub_plot = fig.add_subplot(111)
    gt_vol = torch.sum(target.flatten(), [0]).item()

    gt_vol *= ml_per_voxel

    #get auc, tpr
    score = score.detach() if type(score) == torch.Tensor else score
    try:
        fpr, tpr, _ = metrics.roc_curve(target.flatten(), score.flatten(), drop_intermediate=True)
        if numpy.isnan(tpr[0]):
            raise ValueError('target contains only zeros, can\'t compute ROC curve')
        x = numpy.arange(101) / 100
        tpr_interp = numpy.interp(x, fpr, tpr)  # interpolate roc-curve to get same length results

        auc = metrics.auc(x, tpr_interp)
    except:
        x = numpy.arange(101) / 100
        auc = float('nan')
        tpr_interp = [float('nan')]*101
    #get dice
    try:
        dices = unet_dice_coefficient(score, target, thresholds)
    except:
        dices = [float('nan')]*len(thresholds)

    #get difference
    diffs = unet_diffs(score, target, thresholds)

    # get difference in the volumes
    vol_diffs = unet_vol_diffs(score, target, thresholds, ml_per_voxel)

    accuracies, precisions, recalls, fprs, f1s = unet_statistics(score, target, thresholds)

    #add auc, best dice to fig
    fig.suptitle(f'AUC: {auc}')
    #save plot
    sub_plot.plot(x, tpr_interp)
    plt.savefig(os.path.join(plot_dir, f'roc_{item}'))
    plt.close(fig)

    return list(tpr_interp), auc, dices, diffs, vol_diffs, accuracies, precisions, recalls, fprs, f1s, gt_vol

def plot_average_stats_for_different_thesholds(thresholds, stats, plot_dir, name, best_threshold = 0):
    print("Plotting average ",name)
    fig =plt.figure()
    plt.plot(thresholds, stats)
    plt.axvline(best_threshold, color='red')
    fig.savefig(os.path.join(plot_dir, name))
    plt.close(fig)


def auc(target, score):
    try:
        target = (target > 0.5).float().to('cpu')
        score = score.detach().to('cpu') if type(score) == torch.Tensor else score.to('cpu')
        fpr, tpr, thresholds = metrics.roc_curve(target.flatten(), score.flatten())
        x = numpy.arange(101) / 100
        tpr_interp = numpy.interp(x, fpr, tpr)  # interpolate roc-curve to get same length results
        auc = metrics.auc(x, tpr_interp)
    except:
        auc = float('nan')
    return auc


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


