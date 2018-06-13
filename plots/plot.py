import csv
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = 'data/'
OUT_DIR = 'img/'


def _extract_data(file_path):
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        steps = []
        values = []
        for row in reader:
            steps.append(float(row['Step']))
            values.append(float(row['Value']))
    return steps, values


def _smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


'''
Plot different models loss on the same graph
'''


def plot_loss(in_files, out_file, names, colors, smooth_val, limits):
    num_models = len(in_files)
    limits_x = []
    limits_y = []
    for i in range(num_models):
        steps, values = _extract_data(in_files[i])
        limits_x.append(max(steps))
        limits_y.append(max(values))
        if smooth_val[i] != -1:
            values = _smooth(values, smooth_val[i])
        plt.plot(steps, values, color=colors[i], alpha=0.8, label=names[i])

    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    if limits[0] == -1:
        plt.xlim(xmax=max(limits_x))
    else:
        plt.xlim(xmax=limits[0])
    if limits[1] == -1:
        plt.ylim(ymax=max(limits_y))
    else:
        plt.ylim(ymax=limits[1])

    plt.grid(linestyle='dashed')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    if len(names) > 1:
        plt.legend(names)
    plt.savefig(out_file)
    plt.gcf().clear()


'''
Plot different models acc on the same graph
'''


def plot_acc(in_files, out_file, names, colors, smooth_val, limits):
    num_models = len(in_files)
    limits_x = []
    limits_y = []
    for i in range(num_models):
        steps, values = _extract_data(in_files[i])
        limits_x.append(max(steps))
        limits_y.append(max(values))

        if smooth_val[i] != -1:
            values = _smooth(values, smooth_val[i])
        plt.plot(steps, values, color=colors[i], alpha=0.8, label=names[i])

    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    if limits[0] == -1:
        plt.xlim(xmax=max(limits_x))
    else:
        plt.xlim(xmax=limits[0])
    if limits[1] == -1:
        plt.ylim(ymax=max(limits_y))
    else:
        plt.ylim(ymax=limits[1])

    plt.grid(linestyle='dashed')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    if len(names) > 1:
        plt.legend(names)
    plt.savefig(out_file)
    plt.gcf().clear()


# Best model plots

# plot_loss([DATA_DIR + "train_loss.csv"],
#           OUT_DIR + 'train_loss.png',
#           ["MvitNet"],
#           ['b'],
#           [9],
#           [-1, 500])
#
# plot_acc([DATA_DIR + "train_acc.csv"],
#          OUT_DIR + 'train_acc.png',
#          ["MvitNet"],
#          ['b'],
#          [9],
#          [-1, 0.31])
#
# plot_loss([DATA_DIR + "test_loss.csv"],
#           OUT_DIR + 'test_loss.png',
#           ["MvitNet"],
#           ['b'],
#           [-1],
#           [-1, 500])
#
# plot_acc([DATA_DIR + "test_acc.csv"],
#          OUT_DIR + 'test_acc.png',
#          ["MvitNet"],
#          ['b'],
#          [-1],
#          [-1, 0.31])

# Best model Train + Test

plot_loss([DATA_DIR + "train_loss.csv", DATA_DIR + "test_loss.csv"],
          OUT_DIR + 'all_loss.png',
          ["Training set", "Validation set"],
          ['b', 'r'],
          [9, -1],
          [-1, 500])

plot_acc([DATA_DIR + "train_acc.csv", DATA_DIR + "test_acc.csv"],
         OUT_DIR + 'all_acc.png',
         ["Training set", "Validation set"],
         ['b', 'r'],
         [9, -1],
         [-1, 0.31])
