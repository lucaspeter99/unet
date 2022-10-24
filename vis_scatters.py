import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17}

plt.rc('font', **font)

colors = {
    'lightred': '#f8cecc',
    'darkred': '#b85450',
    'lightgreen': '#d5e8d4',
    'darkgreen': '#82b366',
    'lightblue': '#dae8fc',
    'darkblue': '#6c8ebf',
    'darkyellow': '#dfb139',
}
os.makedirs('./figs', exist_ok=True)

df = pd.read_excel('./windowing_666.xlsx')

mse = df.loc[df['windowing'] == 1]
bce = df.loc[df['windowing'] == 0]

metrics = ['dice', 'auc', 'num_epochs', 'vol_diff']
names = ['DICE', 'AUC', 'EPOCH', 'VOLUME DIFF']

for i,metric in enumerate(metrics):

    box = plt.boxplot(x=mse[metric], positions= [0], widths=[0.66], labels =['windowing'], patch_artist=True,
                      medianprops=dict(linestyle=':', linewidth=2.5, color=colors['darkred']),
                      whiskerprops=dict(linestyle='-', linewidth=2.5, color=colors['darkred']),
                      capprops=dict(linestyle='-', linewidth=2.5, color=colors['darkred']),
                      boxprops=dict(linestyle='-', linewidth=2.5, color=colors['darkred'])

                      )
    box['boxes'][0].set_facecolor(colors['lightred'])
    box = plt.boxplot(x=bce[metric], positions=[1], widths=[0.66], labels=['no windowing'], patch_artist=True,
                      medianprops=dict(linestyle=':', linewidth=2.5, color=colors['darkblue']),
                      whiskerprops=dict(linestyle='-', linewidth=2.5, color=colors['darkblue']),
                      capprops=dict(linestyle='-', linewidth=2.5, color=colors['darkblue']),
                      boxprops=dict(linestyle='-', linewidth=2.5, color=colors['darkblue'])

                      )
    box['boxes'][0].set_facecolor(colors['lightblue'])
    plt.ylabel(names[i])
    plt.xlabel('LOSS')
    plt.tight_layout()
    plt.savefig('./figs/windowing_s2_' + names[i] + '.png')
    plt.close()
