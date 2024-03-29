"""Manage registration and evaluation of summary operations."""
import numpy as np
import tensorflow as tf


class SummaryManager(object):
    """Manager to remember and run summary operations as necessary."""

    def __init__(self, model, cheap_ops_every_n_secs=2, expensive_ops_every_n_mins=2):
        """Initialize manager based on given model instance."""
        self._tensorflow_session = model._tensorflow_session
        self._model = model
        self._cheap_ops = {
            'train': {},
            'test': {},
            'full_test': {},
        }
        self._expensive_ops = {
            'train': {},
            'test': {},
            'full_test': {},
        }
        self._cheap_ops_every_n_secs = cheap_ops_every_n_secs
        self._expensive_ops_every_n_secs = 60 * expensive_ops_every_n_mins

    def _post_model_build(self):
        """Merge together cheap and expensive ops separately."""
        self._writer = tf.summary.FileWriter(self._model.output_path,
                                             self._tensorflow_session.graph)

        for dict_ in (self._cheap_ops, self._expensive_ops):
            for mode in ('train', 'test', 'full_test'):
                prefixes = list(set([k.split('/')[0] for k in dict_[mode].keys()]))
                # Do not merge loss and metric related summaries
                for prefix in ('loss', 'metric'):
                    if prefix in prefixes:
                        prefixes.remove(prefix)
                        for k, o in [(k, o) for k, o in dict_[mode].items()
                                     if k.startswith(prefix + '/')]:
                            del dict_[mode][k]
                            k = k[len('%s/%s/' % (prefix, mode)):]
                            dict_[mode][k] = o
                # Define summary ops to merge based on prefix/scope
                for prefix in prefixes:
                    dict_[mode][prefix] = tf.summary.merge([
                        o for k, o in dict_[mode].items()
                        if k == prefix or k.startswith(prefix + '/')
                    ])
                    for k in [k for k in dict_[mode].keys() if k.startswith(prefix + '/')]:
                        del dict_[mode][k]

        # Merge ops as necessary
        for dict_ in (self._cheap_ops, self._expensive_ops):
            for mode in ('train', 'test', 'full_test'):
                for prefix, ops in dict_[mode].items():
                    if isinstance(ops, list):
                        dict_[mode][prefix] = tf.summary.merge(ops)

    def get_ops(self, mode='train'):
        """Retrieve summary ops to evaluate at given iteration number."""
        if mode == 'test' or mode == 'full_test':  # Always return all ops for test case
            return {
                'cheap': self._cheap_ops[mode],
                'expensive': self._expensive_ops[mode],
            }
        elif mode == 'train':  # Select ops to evaluate based on defined frequency
            ops = {}
            check_func = self._model.time.has_been_n_seconds_since_last
            if check_func('cheap_summaries_train', self._cheap_ops_every_n_secs):
                ops['cheap'] = self._cheap_ops[mode]
            if check_func('expensive_summaries_train', self._expensive_ops_every_n_secs):
                ops['expensive'] = self._expensive_ops[mode]
            return ops

    def write_summaries(self, summary_outputs, iteration_number):
        """Write given outputs to `self._writer`."""
        for _, group in summary_outputs.items():
            for _, summary in group.items():
                self._writer.add_summary(summary, global_step=iteration_number)

    def _get_clean_name(self, operation):
        name = operation.name

        # Determine mode
        mode = 'train'
        if name.startswith('test/') or name.startswith('test_data/'):
            mode = 'test'
        elif name.startswith('loss/test/') or name.startswith('metric/test/'):
            mode = 'full_test'

        # Correct name
        if mode == 'test':
            name = name[name.index('/') + 1:]
        elif mode == 'full_test':
            name = '/'.join(name.split('/')[2:])
        if name[-2] == ':':
            name = name[:-2]
        return mode, name

    def _register_cheap_op(self, operation):
        mode, name = self._get_clean_name(operation)
        assert name not in self._cheap_ops[mode] and name not in self._expensive_ops[mode]
        self._cheap_ops[mode][name] = operation

    def _register_expensive_op(self, operation):
        mode, name = self._get_clean_name(operation)
        assert name not in self._cheap_ops[mode] and name not in self._expensive_ops[mode]
        self._expensive_ops[mode][name] = operation

    def audio(self, name, tensor, **kwargs):
        """TODO: Log summary of audio."""
        raise NotImplementedError('SummaryManager::audio not implemented.')

    def text(self, name, tensor, **kwargs):
        """TODO: Log summary of text."""
        raise NotImplementedError('SummaryManager::text not implemented.')

    def histogram(self, name, tensor, **kwargs):
        """TODO: Log summary of audio."""
        operation = tf.summary.histogram(name, tensor, **kwargs)
        self._register_expensive_op(operation)

    def image(self, name, tensor, data_format='channels_last', **kwargs):
        """TODO: Log summary of image."""
        if data_format == 'channels_first':
            tensor = tf.transpose(tensor, perm=(0, 2, 3, 1))
        operation = tf.summary.image(name, tensor, **kwargs)
        self._register_expensive_op(operation)

    def _4d_tensor(self, name, tensor, **kwargs):
        """Display all filters in a grid for visualization."""
        h, w, c, num_tensor = tensor.shape.as_list()

        # Try to visualise convolutional filters or feature maps
        # See: https://gist.github.com/kukuruza/03731dc494603ceab0c5
        # input shape: (Y, X, C, N)
        if c != 1 and c != 3:
            tensor = tf.reduce_mean(tensor, axis=2, keep_dims=True)
            c = 1
        # shape is now: (Y, X, 1|C, N)
        v_min = tf.reduce_min(tensor)
        v_max = tf.reduce_max(tensor)
        tensor -= v_min
        tensor *= 1.0 / (v_max - v_min)
        tensor = tf.pad(tensor, [[1, 0], [1, 0], [0, 0], [0, 0]], 'CONSTANT')
        tensor = tf.transpose(tensor, perm=(3, 0, 1, 2))
        # shape is now: (N, Y, X, C)
        # place tensor on grid
        num_tensor_x = int(np.round(np.sqrt(num_tensor)))
        num_tensor_y = num_tensor / num_tensor_x
        while not num_tensor_y.is_integer():
            num_tensor_x += 1
            num_tensor_y = num_tensor / num_tensor_x
        num_tensor_y = int(num_tensor_y)
        h += 1
        w += 1
        tensor = tf.reshape(tensor, (num_tensor_x, h * num_tensor_y, w, c))
        # shape is now: (N_x, Y * N_y, X, c)
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        # shape is now: (N_x, X, Y * N_y, c)
        tensor = tf.reshape(tensor, (1, w * num_tensor_x, h * num_tensor_y, c))
        # shape is now: (1, X * N_x, Y * N_y, c)
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        # shape is now: (1, Y * N_y, X * N_x, c)
        tensor = tf.pad(tensor, [[0, 0], [0, 1], [0, 1], [0, 0]], 'CONSTANT')

        self.image(name, tensor, **kwargs)

    def filters(self, name, tensor, **kwargs):
        """Log summary of convolutional filters.

        Note that this method expects the output of the convolutional layer when using
        `tf.layers.conv2d` or for the filters to be defined in the same scope as the output tensor.
        """
        assert 'data_format' not in kwargs
        with tf.name_scope('conv-filter'):
            # Find tensor holding trainable kernel weights
            name_stem = '/'.join(tensor.name.split('/')[:-1]) + '/kernel'
            matching_tensors = [t for t in tf.trainable_variables() if t.name.startswith(name_stem)]
            assert len(matching_tensors) == 1
            filters = matching_tensors[0]

            # H x W x C x N
            h, w, c, n = filters.shape.as_list()
            filters = tf.transpose(filters, perm=(3, 2, 0, 1))
            # N x C x H x W
            filters = tf.reshape(filters, (n*c, 1, h, w))
            # NC x 1 x H x W
            filters = tf.transpose(filters, perm=(2, 3, 1, 0))
            # H x W x 1 x NC

            self._4d_tensor(name, filters, **kwargs)

    def feature_maps(self, name, tensor, mean_across_channels=True, data_format='channels_last',
                     **kwargs):
        """Log summary of feature maps / image activations."""
        with tf.name_scope('feature-map'):
            if data_format == 'channels_first':
                # N x C x H x W
                tensor = tf.transpose(tensor, perm=(0, 2, 3, 1))
            # N x H x W x C
            if mean_across_channels:
                tensor = tf.reduce_mean(tensor, axis=3, keepdims=True)
                # N x H x W x 1
                tensor = tf.transpose(tensor, perm=(1, 2, 3, 0))
            else:
                n, c, h, w = tensor.shape.as_list()
                tensor = tf.reshape(tensor, (n*c, 1, h, w))
                # N x 1 x H x W
                tensor = tf.transpose(tensor, perm=(2, 3, 1, 0))
            # H x W x 1 x N

            self._4d_tensor(name, tensor, **kwargs)

    def scalar(self, name, tensor, **kwargs):
        """Log summary of scalar."""
        operation = tf.summary.scalar(name, tensor, **kwargs)
        self._register_cheap_op(operation)
