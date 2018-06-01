import h5py
import datasources.augmentation as aug
from util.img_transformations import crop_hand, resize
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import time

DATA_PATH = '../datasets/training.h5'

f = h5py.File(DATA_PATH, 'r')
N_samples = len(f['train']['img'])

for i in range(N_samples):
    # for i in range(N_samples):
    img = f['train']['img'][i]
    kp_2D = f['train']['kp_2D'][i]
    vis = f['train']['vis_2D'][i]
    img = img.transpose(1, 2, 0)

    op_array = np.random.randint(2, size=aug.NUM_TRANSFORMATIONS)
    op_array[3] = 1
    op_array[2] = 1

    kp_var = iaa.Sequential([
        iaa.Flipud(0.5),
        iaa.Fliplr(0.5),
        iaa.Affine(scale=0.63,
                   rotate=(-90, 90),
                   shear=(-15, 15)
                   )
    ]).to_deterministic()

    img, kp_2D = aug.perform_augmentation_all(kp_var, img, kp_2D)

    # if op_array[0] == 1:
    #     img, kp_2D = aug.flip_horizontal(img, kp_2D)
    #     # print("h-flip")
    # if op_array[1] == 1:
    #     img, kp_2D = aug.flip_vertical(img, kp_2D)
    #     # print("v-flip")
    # if op_array[2] == 1:
    #     img, kp_2D = aug.rotate(img, kp_2D, None)
    #     # print("rotate")
    # if op_array[3] == 1:
    #     img, kp_2D = aug.shear(img, kp_2D)
    #     # print("shear")

    img, kp_2D = crop_hand(img, kp_2D)
    img, kp_2D = resize(img, kp_2D, (128, 128))

    kp_invar = iaa.Sequential([
        iaa.SomeOf((0, None), [
            iaa.ContrastNormalization((0.8, 1.2)),
            iaa.Multiply((0.8, 1.2)),
            iaa.Dropout(0.01),
            iaa.SaltAndPepper(0.01)
        ], random_order=True)
    ]).to_deterministic()

    img, kp_2D = aug.perform_augmentation_all(kp_invar, img, kp_2D)

    # if op_array[4] == 1:
    #     img, kp_2D = aug.change_contrast(img, kp_2D)
    #     # print("contrast")
    # if op_array[5] == 1:
    #     img, kp_2D = aug.change_brightness(img, kp_2D)
    #     # print("brightness")
    # if op_array[6] == 1:
    #     img, kp_2D = aug.dropout(img, kp_2D)
    #     # print("dropout")
    # if op_array[7] == 1:
    #     img, kp_2D = aug.salt_pepper(img, kp_2D)
    #     # print("salt-n-pepper")

    img = img / 255.0
    # print("Visibility prior: {}".format(np.squeeze(vis)))
    new_vis = aug.get_vis(kp_2D, vis)
    # print("Visibility post : {}".format(np.squeeze(new_vis)))
    # aug.show_image(img, kp_2D)

# testing effiecny 
# start = time.time()
# for i in range(5000):
#     img = f['train']['img'][i]
#     kp = f['train']['kp_2D'][i]
#     img, kp_2D = aug.default_processing(img, kp)
#     img, kp_2D = aug.rotate(img, kp_2D)
#     img, kp_2D = aug.shear(img, kp_2D)
#     vis = aug.check_vis(kp_2D)
# end = time.time()
# print("Vectorized t = {}".format(end - start))

# start = time.time()
# for i in range(5000):
#     img = f['train']['img'][i]
#     kp = f['train']['kp_2D'][i]
#     img, kp_2D = aug.default_processing(img, kp)
#     img, kp_2D = aug.rotate(img, kp_2D)
#     img, kp_2D = aug.shear(img, kp_2D)
#     vis = aug.check_vis_inif(kp_2D)
# end = time.time()
# print("Looping t = {}".format(end - start))