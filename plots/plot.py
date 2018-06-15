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


def _smooth_i(x,window_len=11,window='hanning'):
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def _smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

'''
Plot different models loss on the same graph
'''


def plot_loss(in_files, out_file, names, colors, smooth_val, limits, dims=(640, 480), yticks=None):
    num_models = len(in_files)
    limits_x = []
    limits_y = []
    for i in range(num_models):
        steps, values = _extract_data(in_files[i])
        limits_x.append(max(steps))
        limits_y.append(max(values))

        if smooth_val[i] != -1:
            values = _smooth_i(values, smooth_val[i])
        plt.plot(steps, values[:1000], color=colors[i], alpha=0.8, label=names[i])

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

    if yticks is not None:
        plt.yticks(yticks)

    # set dims
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(dims[0] / float(DPI), dims[1] / float(DPI))

    plt.savefig(out_file, bbox_inches='tight')
    plt.gcf().clear()


'''
Plot different models acc on the same graph
'''


def plot_acc(in_files, out_file, names, colors, smooth_val, limits, dims=(640, 480), yticks=None):
    num_models = len(in_files)
    limits_x = []
    limits_y = []
    for i in range(num_models):
        steps, values = _extract_data(in_files[i])
        limits_x.append(max(steps))
        limits_y.append(max(values))

        if smooth_val[i] != -1:
            values = _smooth_i(values, smooth_val[i])
        plt.plot(steps, values[:1000], color=colors[i], alpha=0.8, label=names[i])

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

    if yticks is not None:
        plt.yticks(yticks)

    # set dims
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(dims[0] / float(DPI), dims[1] / float(DPI))

    plt.savefig(out_file, bbox_inches='tight')
    plt.gcf().clear()


# Best model plots

smooth_var = 10


ticks = np.arange(0, 500, 100)
plot_loss([DATA_DIR + "incres_train_loss.csv", DATA_DIR + "resnet_train_loss.csv", DATA_DIR + "glover_train_loss.csv", DATA_DIR + "incep_train_loss.csv"],
          OUT_DIR + 'compare_train_loss.png',
          ["Inception-ResNet-v2", "ResNet34", "CPM", "Inception-v3"],
          ['b', 'r', 'g', 'y'],
          [smooth_var*2,smooth_var*2,smooth_var*2,smooth_var*2],
          [243000, 400],
          (640, 400),
          ticks)

ticks = np.arange(0, 0.5, 0.1)
plot_acc([DATA_DIR + "incres_train_acc.csv", DATA_DIR + "resnet_train_acc.csv", DATA_DIR + "glover_train_acc.csv", DATA_DIR + "incep_train_acc.csv"],
          OUT_DIR + 'compare_train_acc.png',
         ["Inception-ResNet-v2", "ResNet34", "CPM", "Inception-v3"],
          ['b', 'r', 'g', 'y'],
          [smooth_var,smooth_var,smooth_var,smooth_var],
          [243000, 0.4],
          (640, 400),
          ticks)

ticks = np.arange(0, 500, 100)
plot_loss([DATA_DIR + "incres_test_loss.csv", DATA_DIR + "resnet_test_loss.csv", DATA_DIR + "glover_test_loss.csv", DATA_DIR + "incep_test_loss.csv"],
          OUT_DIR + 'compare_test_loss.png',
          ["Inception-ResNet-v2", "ResNet34", "CPM", "Inception-v3"],
          ['b', 'r', 'g', 'y'],
          [-1,-1,-1,-1],
          [243000, 400],
          (640, 400),
          ticks)

ticks = np.arange(0, 0.5, 0.1)
plot_acc([DATA_DIR + "incres_test_acc.csv", DATA_DIR + "resnet_test_acc.csv", DATA_DIR + "glover_test_acc.csv", DATA_DIR + "incep_test_acc.csv"],
          OUT_DIR + 'compare_test_acc.png',
          ["Inception-ResNet-v2", "ResNet34", "CPM", "Inception-v3"],
          ['b', 'r', 'g', 'y'],
          [-1,-1,-1,-1],
          [243000, 0.4],
          (640, 400),
          ticks)

# Best model Train + Test

smooth_var = 10

ticks = np.arange(0, 250, 50)
plot_loss([DATA_DIR + "incres_train_loss.csv", DATA_DIR + "incres_test_loss.csv"],
          OUT_DIR + 'incres_all_loss.png',
          ["Training", "Validation"],
          ['b', 'r'],
          [smooth_var, -1],
          [-1, 200],
          (640, 400),
          ticks)

ticks = np.arange(0, 0.55, 0.1)
plot_acc([DATA_DIR + "incres_train_acc.csv", DATA_DIR + "incres_test_acc.csv"],
         OUT_DIR + 'incres_all_acc.png',
         ["Training", "Validation"],
         ['b', 'r'],
         [smooth_var, -1],
         [-1, 0.45],
         (640, 400),
         ticks)
