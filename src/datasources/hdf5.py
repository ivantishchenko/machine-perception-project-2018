"""HDF5 data source for hand joint recognition."""
from threading import Lock
from typing import List

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf
from numpy.random import RandomState

import datasources.augmentation as aug

from core import BaseDataSource
from util.img_transformations import crop_hand, resize
from imgaug import augmenters as iaa
# from util.img_transformations import crop_hand, resize, rotate, flipLR


class HDF5Source(BaseDataSource):
    """HDF5 data loading class (using h5py)."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 keys_to_use: List[str],
                 hdf_path: str,
                 testing=False,
                 validation=False,
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys."""
        hdf5 = h5py.File(hdf_path, 'r')
        self._short_name = 'HDF:%s' % '/'.join(hdf_path.split('/')[-2:])
        if testing:
            # Make it clear for the log-output that testing is currently
            # performed on the data-set
            self._short_name += ':test'

        # Internal logic, prevent's thread creation which sends data to GPU
        # Also, if set to true, will not cut the hand as per given keypoints
        self.testing = testing
        self.validation = validation

        # Random state for data augmentation
        # TODO: Interesting to maybe adjust for our project
        self.randomState = RandomState(0)

        # Create global index over all specified keys
        self._index_to_key = {}
        index_counter = 0
        for key in keys_to_use:
            n = hdf5[key]['img'].shape[0]
            for i in range(n):
                # Store a tuple of (key, index) to be able to reference the
                # index of a specific key
                self._index_to_key[index_counter] = (key, i)
                index_counter += 1
        # Return the total amount of entries, keys_to_use is implicitly
        # obtainable
        self._num_entries = index_counter

        # Reference to the opened h5 file.
        self._hdf5 = hdf5
        self._mutex = Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0

    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._num_entries

    @property
    def short_name(self):
        """Short name specifying source HDF5."""
        return self._short_name

    def cleanup(self):
        """Close HDF5 file before running base class cleanup routine."""
        self._hdf5.close()
        super().cleanup()

    def reset(self):
        """Reset index."""
        self._current_index = 0
        with self._mutex:
            super().reset()

    def entry_generator(self, yield_just_one=False):
        """Read entry from HDF5."""
        try:
            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_index >= self.num_entries:
                        if self.testing:
                            break
                        else:
                            self._current_index = 0
                    current_index = self._current_index
                    self._current_index += 1

                key, index = self._index_to_key[current_index]
                data = self._hdf5[key]
                entry = {}
                for name in ('img', 'kp_2D', 'vis_2D'):
                    if name in data:
                        entry[name] = data[name][index, :]
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Resize image and normalize intensities."""
        res_size = (128, 128)
        img = entry['img'].transpose(1, 2, 0)

        if self.validation or not self.testing:
            kp_2D = entry['kp_2D']
            if not self.validation:
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

                augmentation_flag = np.random.binomial(1, 0.7)
                if augmentation_flag == 0:
                    img, kp_2D = aug.perform_augmentation_all(kp_var, img, kp_2D)

                img, kp_2D = crop_hand(img, kp_2D)
                img, kp_2D = resize(img, kp_2D, res_size)
                # Update keypoint visibility (this should be equal to noop for the current design, no points go out
                # of the image
                entry['vis_2D'] = aug.get_vis(kp_2D, entry['vis_2D'])

                if augmentation_flag == 0:
                    img, kp_2D = aug.perform_augmentation_all(kp_invar, img, kp_2D)

            else:
                img, kp_2D = crop_hand(img, kp_2D)
                img, kp_2D = resize(img, kp_2D, res_size)
            entry['kp_2D'] = kp_2D

        img = img / 255.0
        entry['img'] = img.transpose(2, 0, 1)

        # Ensure all values in an entry are 4-byte floating point numbers
        for key, value in entry.items():
            entry[key] = value.astype(np.float32)

        return entry
