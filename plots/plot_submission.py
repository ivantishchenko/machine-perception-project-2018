import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv

colours = {0: 'black', 1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'yellow'}

DATASET_PATH = '../datasets/testing.h5'
SUBMISSION_PATH = 'sub/'
OUT_PATH = 'img/sub/'
N = 5


def plot_submission(dataset, submission, out_pattern, select_idx, normalized=True):
    with open(submission, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header

        buf = []
        for row in reader:
            buf.append(row)

        for count in range(N):
            kp_raw = buf[select_idx[count]][1:]
            i = int(buf[select_idx[count]][0])
            kp_np = np.asarray(kp_raw, dtype=np.float32)
            kp_np = np.reshape(kp_np, (21, 2))
            img = dataset['img'][i]
            img = img.transpose(1, 2, 0)
            kpx = kp_np[:, 0]
            kpy = kp_np[:, 1]
            if normalized:
                img = img / 255
            plt.imshow(img)
            # Root of hand drawn
            plt.scatter(kpx[0], kpy[0], color=colours[0])
            # Attach all knuckles to root of hand, then draw then fingers
            for i in range(1, 6):
                plt.plot([kpx[0], kpx[i * 4]], [kpy[0], kpy[i * 4]], color=colours[0])
                plt.plot(kpx[i * 4 - 3:(i + 1) * 4 - 3], kpy[i * 4 - 3:(i + 1) * 4 - 3], marker='o', color=colours[i])
            plt.savefig(out_pattern + str(count) + '.png', bbox_inches='tight')
            # plt.savefig(OUT_PATTERN + str(count) + '.png')
            plt.gcf().clear()


g = h5py.File(DATASET_PATH, 'r')
testset = g['test']

select_idx = np.random.random_integers(0, len(testset['img']) - 1, N)

# plot submissions
plot_submission(testset, SUBMISSION_PATH + 'incres.csv', OUT_PATH + 'incres', select_idx)
plot_submission(testset, SUBMISSION_PATH + 'incep.csv', OUT_PATH + 'incep', select_idx)
plot_submission(testset, SUBMISSION_PATH + 'resnet.csv', OUT_PATH + 'resnet', select_idx)
plot_submission(testset, SUBMISSION_PATH + 'cpm.csv', OUT_PATH + 'cpm', select_idx)
