import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick


DATA_DIR = 'data/'
OUT_DIR = 'out/'

def extract_data(file_path):
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        steps = []
        values = []
        for row in reader:
            steps.append(float(row['Step']))
            values.append(float(row['Value']))
    return steps, values

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

'''
Plot different models loss on the same graph
'''
def plot_loss(in_files, out_file, names, colors, smooth_val):
    num_models = len(in_files)
    limits_x = []
    # limits_y = []
    for i in range(num_models):
        steps, values = extract_data(in_files[i])
        limits_x.append(max(steps))
        # limits_y.append(max(values))
        if smooth_val[i] != -1:
            smooth_val = smooth(values, smooth_val[i])
        else:
            smooth_val = values
        plt.plot(steps, smooth_val, color=colors[i], alpha=0.8, label=names[i])

    plt.ylim(ymin=0)
    # plt.ylim(ymax=max(limits_y))
    plt.ylim(ymax=500)
    plt.xlim(xmin=0)
    plt.xlim(xmax=max(limits_x))
    plt.grid(linestyle='dashed')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    # plt.legend()
    plt.savefig(out_file)

'''
Plot different models acc on the same graph
'''
def plot_acc(in_files, out_file, names, colors, smooth_val):
    num_models = len(in_files)
    limits_x = []
    limits_y = []
    for i in range(num_models):
        steps, values = extract_data(in_files[i])
        limits_x.append(max(steps))
        # limits_y.append(max(values))

        if smooth_val[i] != -1:
            smooth_val = smooth(values, smooth_val[i])
        else:
            smooth_val = values
        plt.plot(steps, smooth_val, color=colors[i], alpha=0.8)

    plt.ylim(ymin=0)
    # plt.ylim(ymax=max(limits_y))
    plt.ylim(ymax=0.31)
    plt.xlim(xmin=0)
    plt.xlim(xmax=max(limits_x))
    plt.grid(linestyle='dashed')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    # plt.legend(names)
    plt.savefig(out_file)
    plt.gcf().clear()


# Model Comparison plots
plot_loss([DATA_DIR + "train_loss.csv"],
          OUT_DIR + 'train_loss.png',
          ["MvitNet"],
          ['b'],
          [9])

plot_acc([DATA_DIR + "train_acc.csv"],
         OUT_DIR + 'train_acc.png',
         ["MvitNet"],
         ['b'],
         [9])

plot_loss([DATA_DIR + "test_loss.csv"],
          OUT_DIR + 'test_loss.png',
          ["MvitNet"],
          ['b'],
          [-1])

plot_acc([DATA_DIR + "test_acc.csv"],
         OUT_DIR + 'test_acc.png',
         ["MvitNet"],
         ['b'],
         [-1])
