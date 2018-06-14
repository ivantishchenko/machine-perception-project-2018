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


def plot_loss(in_files, out_file, names, colors, smooth_val, limits, dims=(640, 480)):
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

    # set dims
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(dims[0] / float(DPI), dims[1] / float(DPI))

    plt.savefig(out_file)
    plt.gcf().clear()


'''
Plot different models acc on the same graph
'''


def plot_acc(in_files, out_file, names, colors, smooth_val, limits, dims=(640, 480)):
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

    # set dims
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(dims[0] / float(DPI), dims[1] / float(DPI))

    plt.savefig(out_file)
    plt.gcf().clear()


# Best model plots

plot_loss([DATA_DIR + "incres_train_loss.csv", DATA_DIR + "resnet_train_loss.csv", DATA_DIR + "cpm_train_loss.csv", DATA_DIR + "incep_train_loss.csv"],
          OUT_DIR + 'compare_train_loss.png',
          ["InceptionResNet", "ResNet", "CPM", "Inception"],
          ['b', 'r', 'g', 'y'],
          [8,8,8,8],
          [-1, -1],
          (640, 400))

plot_loss([DATA_DIR + "incres_train_acc.csv", DATA_DIR + "resnet_train_acc.csv", DATA_DIR + "cpm_train_acc.csv", DATA_DIR + "incep_train_acc.csv"],
          OUT_DIR + 'compare_train_acc.png',
          ["InceptionResNet", "ResNet", "CPM", "Inception"],
          ['b', 'r', 'g', 'y'],
          [8,8,8,8],
          [-1, -1],
          (640, 400))

plot_loss([DATA_DIR + "incres_test_loss.csv", DATA_DIR + "resnet_test_loss.csv", DATA_DIR + "cpm_test_loss.csv", DATA_DIR + "incep_test_loss.csv"],
          OUT_DIR + 'compare_test_loss.png',
          ["InceptionResNet", "ResNet", "CPM", "Inception"],
          ['b', 'r', 'g', 'y'],
          [-1,-1,-1,-1],
          [-1, -1],
          (640, 400))

plot_loss([DATA_DIR + "incres_test_acc.csv", DATA_DIR + "resnet_test_acc.csv", DATA_DIR + "cpm_test_acc.csv", DATA_DIR + "incep_test_acc.csv"],
          OUT_DIR + 'compare_test_acc.png',
          ["InceptionResNet", "ResNet", "CPM", "Inception"],
          ['b', 'r', 'g', 'y'],
          [-1,-1,-1,-1],
          [-1, -1],
          (640, 400))

# Best model Train + Test

# plot_loss([DATA_DIR + "incres_train_loss.csv", DATA_DIR + "incres_test_loss.csv"],
#           OUT_DIR + 'incres_all_loss.png',
#           ["Training", "Testing"],
#           ['b', 'r'],
#           [10, -1],
#           [-1, 350],
#           (640, 400))
#
# plot_acc([DATA_DIR + "incres_train_acc.csv", DATA_DIR + "incres_test_acc.csv"],
#          OUT_DIR + 'incres_all_acc.png',
#          ["Training", "Testing"],
#          ['b', 'r'],
#          [10, -1],
#          [-1, 0.45],
#          (640, 400))
