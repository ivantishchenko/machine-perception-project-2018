import datasources.augmentation as aug
import h5py
import datasources.augmentation as aug
import numpy as np

DATA_PATH = '../datasets/training.h5'

f = h5py.File(DATA_PATH, 'r')
N_samples = len(f['train']['img'])

for i in range(N_samples):
    img = f['train']['img'][i]
    kp = f['train']['kp_2D'][i]

    img, kp_2D = aug.default_processing(img, kp)
    aug.show_image(img, kp_2D)

    op_array = np.random.randint(2, size=aug.NUM_TRANSFORMATIONS)
    if op_array[0] == 1:
        img, kp_2D = aug.flip_horizontal(img, kp_2D)
    if op_array[1] == 1:
        img, kp_2D = aug.flip_vertical(img, kp_2D)
    if op_array[2] == 1:
        img, kp_2D = aug.rotate(img, kp_2D)
    if op_array[3] == 1:
        img, kp_2D = aug.shear(img, kp_2D)
    if op_array[4] == 1:
        img, kp_2D = aug.change_contrast(img, kp_2D)
    if op_array[5] == 1:
        img, kp_2D = aug.change_brightness(img, kp_2D)
    if op_array[6] == 1:
        img, kp_2D = aug.dropout(img, kp_2D)
    if op_array[7] == 1:
        img, kp_2D = aug.salt_pepper(img, kp_2D)

    # img, kp_2D = aug.salt_pepper(img, kp_2D)
    aug.show_image(img, kp_2D)