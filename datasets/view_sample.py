import h5py
import numpy as np
import matplotlib.pyplot as plt

colours = {0: 'black', 1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'yellow'}


def view_group(dataset, count=-1, offset=0, normalized=False):
    if count == -1:
        samples = len(dataset['img'])
    else:
        samples = count

    print("{} images in this set. Starting at {}".format(samples, offset))
    for i in range(offset, samples):
        img = dataset['img'][i]
        kp_2D = dataset['kp_2D'][i]
        kp_vis = dataset['vis_2D'][i]
        kp_vis = np.squeeze(kp_vis)
        kpx = kp_2D[:, 0]
        kpy = kp_2D[:, 1]
        img = img.transpose(1, 2, 0)
        if normalized:
            # Image colours are set to 2 * ([0, 1] - 0.5] == [-1, 1]
            img = 2 * ((img / 255) - 0.5)
        plt.imshow(img)
        # Root of hand drawn
        plt.scatter(kpx[0], kpy[0], color=colours[0])
        # Attach all knuckles to root of hand, then draw then fingers
        for j in range(1, 6):
            plt.plot([kpx[0], kpx[j*4]], [kpy[0], kpy[j*4]], color=colours[0])
            plt.plot(kpx[j*4-3:(j+1)*4-3], kpy[j*4-3:(j+1)*4-3], marker='o', color=colours[j])
        print("KP visibility image {}: {}".format(i, kp_vis))
        plt.show()


f = h5py.File('dataset.h5', 'r')
trainset = f['train']
validateset = f['validate']

g = h5py.File('testing.h5', 'r')
print("{} images in this set.".format(len(g['test']['img'])))
print("{} images in this set.".format(len(trainset['img'])))
print("{} images in this set.".format(len(validateset['img'])))

view_group(trainset)
