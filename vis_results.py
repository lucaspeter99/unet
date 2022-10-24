import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17}

plt.rc('font', **font)

def plot_hist_with_average(series, plot_name, xlabel):
    avg_dice = series.mean()

    plt.hist(series, color=colors['darkblue'])
    plt.vlines(x=avg_dice, ymin=0, ymax=plt.gca().get_ylim()[1], linestyles='dashed', colors=colors['darkred'], linewidths = 3.0)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig('./figs/' + plot_name + '.png')
    plt.close()

def plot_comparison_hist(multiple_series, colors, labels, plot_name, xlabel):
    plt.hist(x = multiple_series, color = colors, label = labels, rwidth=0.95)
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig('./figs/' + plot_name + '.png')
    plt.close()

def read_and_plot_comp_hist(df, column, matches, metric, colors, labels, tag):
    multiple_series = []
    for match in matches:
        df_temp = df.loc[df[column] == match]
        multiple_series.append(df_temp[metric])
    plot_comparison_hist(multiple_series,
                         colors,
                         labels,
                         metric + "_" + column + "_comp" + tag,
                         metric.upper())

df = pd.read_excel('./display_results_555.xlsx')

seed1 = df.loc[df['seed'] == 555]
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(seed1)
seed_name = "seed1"

plot_hist_with_average(np.array(seed1['dice']), 'dice_' + seed_name, 'DICE')
plot_hist_with_average(np.array(seed1['auc']), 'auc_' + seed_name, 'AUC')
plot_hist_with_average(np.array(seed1['vdiff.']), 'vdiff_' + seed_name, 'VDIFF.')


hist_colors = [colors['darkblue'], colors['darkgreen'], colors['darkred'], colors['darkyellow']]

read_and_plot_comp_hist(seed1, "type", ["normalized","resliced"], "dice", hist_colors[:2], ["Normalized", "Resliced"], seed_name)
read_and_plot_comp_hist(seed1, "T", [0,1,2,3], "dice", hist_colors[:4], ["T0", "T1", "T2", "T3"], seed_name)
read_and_plot_comp_hist(seed1, "batch_size", [12,24], "dice", hist_colors[:2], ["bs12", "bs24"], seed_name)

read_and_plot_comp_hist(seed1, "type", ["normalized","resliced"], "auc", hist_colors[:2], ["Normalized", "Resliced"], seed_name)
read_and_plot_comp_hist(seed1, "T", [0,1,2,3], "auc", hist_colors[:4], ["T0", "T1", "T2", "T3"], seed_name)
read_and_plot_comp_hist(seed1, "batch_size", [12,24], "auc", hist_colors[:2], ["bs12", "bs24"], seed_name)

read_and_plot_comp_hist(seed1, "type", ["normalized","resliced"], "vdiff.", hist_colors[:2], ["Normalized", "Resliced"], seed_name)
read_and_plot_comp_hist(seed1, "T", [0,1,2,3], "vdiff.", hist_colors[:4], ["T0", "T1", "T2", "T3"], seed_name)
read_and_plot_comp_hist(seed1, "batch_size", [12,24], "vdiff.", hist_colors[:2], ["bs12", "bs24"], seed_name)


