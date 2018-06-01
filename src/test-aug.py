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
    res_size = (128, 128)
    img = f['train']['img'][i]
    kp_2D = f['train']['kp_2D'][i]
    vis = f['train']['vis_2D'][i]
    img = img.transpose(1, 2, 0)

    kp_var = iaa.Sequential([  # Keypoint variant modifications
        iaa.Flipud(0.5),  # 50% chance to flip horizontally
        iaa.Fliplr(0.5),  # 50% chance to flip vertically
        iaa.Affine(scale=0.63,  # Ensures none of the following ops put keypoints out of the image
                   rotate=(-90, 90),  # Rotate the image between [-90, 90] degrees randomly
                   shear=(-15, 15)  # Shear the image between [-15, 15] degrees randomly
                   )
    ]).to_deterministic()
    kp_invar = iaa.Sequential([  # Keypoint invariant modifications
        iaa.SomeOf((0, None), [  # Run up to all operations
            iaa.ContrastNormalization((0.8, 1.2)),  # Contrast modifications
            iaa.Multiply((0.8, 1.2)),  # Brightness modifications
            iaa.Dropout(0.01),  # Drop out single pixels
            iaa.SaltAndPepper(0.01)  # Add salt-n-pepper noise
        ], random_order=True)  # Randomize the order of operations
    ]).to_deterministic()

    img, kp_2D = aug.perform_augmentation_all(kp_var, img, kp_2D)
    # Update keypoint visibility (this should be equal to noop for the current design, no points go out
    # of the image

    img, kp_2D = crop_hand(img, kp_2D)
    img, kp_2D = resize(img, kp_2D, res_size)
    print("Visibility prior: {}".format(np.squeeze(vis)))
    vis = aug.get_vis(kp_2D, vis)
    print("Visibility post : {}".format(np.squeeze(vis)))

    img, kp_2D = aug.perform_augmentation_all(kp_invar, img, kp_2D)

    img = img / 255.0
    # img = img.transpose(2, 0, 1)
    aug.show_image(img, kp_2D)

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