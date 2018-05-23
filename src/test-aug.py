import datasources.augmentation as aug
import h5py
import datasources.augmentation as aug

DATA_PATH = '../datasets/training.h5'

f = h5py.File(DATA_PATH, 'r')
N_samples = len(f['train']['img'])

for i in range(N_samples):
    img = f['train']['img'][i]
    kp = f['train']['kp_2D'][i]

    img, kp_2D = aug.default_processing(img, kp)
    aug.show_image(img, kp_2D)

    img, kp_2D = aug.dropout(img, kp_2D)
    aug.show_image(img, kp_2D)