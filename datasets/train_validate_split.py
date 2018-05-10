"""
A small script to generate a new h5-file which contains training and
validation data
"""

import h5py
import numpy as np

VALIDATION_PERCENTAGE = 15
SEED = 42

f = h5py.File('training.h5', 'r')
n_samples = len(f['train']['img'])
index_array = list(range(n_samples))
np.random.seed(SEED)
permuted_indexes = np.random.permutation(index_array)
validation_count = n_samples // 100 * VALIDATION_PERCENTAGE
training_count = n_samples - validation_count
validation_indexes = permuted_indexes[:validation_count]
training_indexes = permuted_indexes[validation_count:]
assert len(training_indexes) == training_count, "Counting errors"

# Generate output file
dataset = h5py.File('dataset.h5', 'a')

# Validation data
val = dataset.create_group('validate')
val_imgs = val.create_dataset('img', (validation_count, 3, 320, 320), dtype='uint8')
val_2dkp = val.create_dataset('kp_2D', (validation_count, 21, 2), dtype='float32')
val_kp_v = val.create_dataset('vis_2D', (validation_count, 21, 1), dtype='uint8')
for i in ['img', 'kp_2D', 'vis_2D']:
    for idx, j in enumerate(validation_indexes):
        dataset['validate'][i][idx] = f['train'][i][j]

# Training data
train = dataset.create_group('train')
train_imgs = train.create_dataset('img', (training_count, 3, 320, 320), dtype='uint8')
train_2dkp = train.create_dataset('kp_2D', (training_count, 21, 2), dtype='float32')
train_kp_v = train.create_dataset('vis_2D', (training_count, 21, 1), dtype='uint8')
for i in ['img', 'kp_2D', 'vis_2D']:
    for idx, j in enumerate(training_indexes):
        dataset['train'][i][idx] = f['train'][i][j]

dataset.close()
f.close()
