#!/usr/bin/env python3
"""Create subset of data for Machine Perception project."""
import os

import h5py
import numpy as np
import pandas as pd

# MPIIGaze indices to train on or test on
person_ids_to_test_on = (1, 5, 7, 8, 14)
person_ids_to_train_on = tuple([i for i in range(15) if i not in person_ids_to_test_on])

output_dir = '../datasets'

with h5py.File(os.path.expanduser('~/github/GazeD/cache/MPIIGaze/eye_gaze.h5'), 'r') as h5f:
    all_data = {}
    for in_group, out_group, indices in [('train', 'train', person_ids_to_train_on),
                                         ('test', 'validation', person_ids_to_train_on),
                                         ('test', 'test', person_ids_to_test_on)]:
        all_data[out_group] = {
            'eye': [],
            'head': [],
            'gaze': [],
        }
        for index in indices:
            person_id = 'p%02d' % index
            person_data = h5f[in_group][person_id]
            for tag in ('eye', 'head', 'gaze'):
                values = person_data[tag]
                if out_group == 'validation':  # Sub-sample for quicker validation
                    values = values[::10, :]
                all_data[out_group][tag] += list(values)
        for tag in ('eye', 'head', 'gaze'):
            all_data[out_group][tag] = np.asarray(all_data[out_group][tag])


def store_all(output_file, test_indices_selector):
    """Store `all_data` to desired path as HDF5 record."""
    output_path = '%s/%s' % (output_dir, output_file)
    with h5py.File(output_path, 'w') as h5f:
        for group_name, group_data in all_data.items():
            h5f_group = h5f.create_group(group_name)
            for key, values in group_data.items():
                if group_name == 'test':
                    values = values[test_indices_selector, :]
                    print(group_name, key, values.shape)
                h5f_group.create_dataset(key, data=values)

# Get indices for private/public split
full_test_data = all_data['test']
num_test_entries = full_test_data['eye'].shape[0]
sample_csv = pd.DataFrame(full_test_data['gaze'], columns=['pitch', 'yaw'])
sample_csv.index.name = 'Id'
sample_csv.to_csv('%s/kaggle_sample.csv' % output_dir)

is_private = np.random.choice([0, 1], size=(num_test_entries,), p=[0.25, 0.75]).astype(bool)

solution_csv = sample_csv.copy()
solution_csv['Usage'] = ['Private' if flag else 'Public' for flag in is_private]
solution_csv.index.name = 'Id'
solution_csv.to_csv('%s/kaggle_solution.csv' % output_dir)

# Remove labels from test data
store_all('MPIIGaze_kaggle_wookie.h5', is_private)
del all_data['test']['gaze']
store_all('MPIIGaze_kaggle_students.h5', np.ones_like(is_private))
