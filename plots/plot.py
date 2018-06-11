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
def plot_loss(in_files, out_file, names, colors):
    num_models = len(in_files)
    limits_x = []
    for i in range(num_models):
        steps, values = extract_data(in_files[i])
        limits_x.append(max(steps))
        plt.plot(steps, smooth(values, 5),  color=colors[i], alpha=0.8, label=names[i])

    plt.ylim(ymin=0)
    plt.ylim(ymax=1000)
    plt.xlim(xmin=0)
    plt.xlim(xmax=max(limits_x))
    plt.grid(linestyle='dashed')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(out_file)

'''
Plot different models acc on the same graph
'''
def plot_acc(in_files, out_file, names, colors):
    num_models = len(in_files)
    limits_x = []
    limits_y = []
    for i in range(num_models):
        steps, values = extract_data(in_files[i])
        limits_x.append(max(steps))
        limits_y.append(max(values))
        plt.plot(steps, smooth(values, 5),  color=colors[i], alpha=0.8)

    plt.ylim(ymin=0)
    plt.ylim(ymax=max(limits_y))
    plt.xlim(xmin=0)
    plt.xlim(xmax=max(limits_x))
    plt.grid(linestyle='dashed')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend(names)
    plt.savefig(out_file)
    plt.gcf().clear()


# Model Comparison plots
plot_loss([DATA_DIR + 'mvit_loss.csv', DATA_DIR + "incept_loss.csv"],
          OUT_DIR + 'loss.png',
          ["MvitNet", "InceptionNet"],
          ['b', 'r'])

plot_acc([DATA_DIR + 'mvit_acc.csv', DATA_DIR + "incept_acc.csv"],
         OUT_DIR + 'acc.png',
         ["MvitNet", "InceptionNet"],
         ['b', 'r'])

# single
plot_loss([DATA_DIR + 'mvit_loss.csv'],
          OUT_DIR + 'test.png',
          ["MvitNet"],
          ['b'])
