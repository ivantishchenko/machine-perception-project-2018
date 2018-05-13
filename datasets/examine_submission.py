import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv

colours = {0: 'black', 1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'yellow'}

project_root = os.path.dirname(os.path.abspath(os.path.dirname(sys.argv[0])))
submission_path = os.path.join(project_root, 'outputs/Glover/to_submit_to_kaggle.csv')


def view_submission(dataset, submission, normalized = True):
    with open(submission, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            kp_raw = row[1:]
            i = int(row[0])
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
            plt.show()


g = h5py.File('testing.h5', 'r')
testset = g['test']
print(submission_path)
view_submission(testset, submission_path)
