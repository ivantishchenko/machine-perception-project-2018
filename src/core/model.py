"""Base model class for Tensorflow-based model construction."""
from .data_source import BaseDataSource
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from .live_tester import LiveTester
from .time_manager import TimeManager
from .summary_manager import SummaryManager
from .checkpoint_manager import CheckpointManager
import logging
logger = logging.getLogger(__name__)


class BaseModel(object):
    """Base model class for Tensorflow-based model construction.

    This class assumes that there exist no other Tensorflow models defined.
    That is, any variable that exists in the Python session will be grabbed by the class.
    """

    def __init__(self,
                 tensorflow_session: tf.Session,
                 learning_schedule: List[Dict[str, Any]],
                 train_data: Dict[str, BaseDataSource],
                 test_data: Dict[str, BaseDataSource] = {},
                 test_losses_or_metrics: str = None):
        """Initialize model with data sources and parameters."""
        assert len(train_data) > 0
        self._tensorflow_session = tensorflow_session
        self._train_data = train_data
        self._test_data = test_data
        self._test_losses_or_metrics = test_losses_or_metrics
        self._initialized = False

        # Extract and keep known prefixes/scopes
        self._learning_schedule = learning_schedule
        self._known_prefixes = [schedule for schedule in learning_schedule]

        # Check consistency of given data sources
        train_data_sources = list(train_data.values())
        test_data_sources = list(test_data.values())
        self._batch_size = train_data_sources.pop().batch_size
        for data_source in train_data_sources + test_data_sources:
            if data_source.batch_size != self._batch_size:
                raise ValueError(('Data source "%s" has anomalous batch size of %d ' +
                                  'when detected batch size is %d.') % (data_source.short_name,
                                                                        data_source.batch_size,
                                                                        self._batch_size))

        # Register a manager for tf.Summary
        self.summary = SummaryManager(self)

        # Register a manager for checkpoints
        self.checkpoint = CheckpointManager(self)

        # Register a manager for timing related operations
        self.time = TimeManager(self)

        # Prepare for live (concurrent) validation/testing during training, on the CPU
        self._enable_live_testing = len(self._test_data) > 0
        self._tester = LiveTester(self, self._test_data)

        # Run-time parameters
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.use_batch_statistics = tf.placeholder(tf.bool, name='use_batch_statistics')

        self._build_all_models()

    @property
    def identifier(self):
        """Identifier for model based on data sources and parameters."""
        return self.__class__.__name__

    @property
    def output_path(self):
        """Path to store logs and model weights into."""
        return '%s/%s' % (os.path.abspath(os.path.dirname(__file__) + '/../../outputs'),
                          self.identifier)

    def _build_all_models(self):
        """Build training (GPU/CPU) and testing (CPU) streams."""
        self.output_tensors = {}
        self.loss_terms = {}
        self.metrics = {}

        def _build_datasource_summaries(data_sources, mode):
            """Register summary operations for input data from given data sources."""
            with tf.variable_scope('%s_data' % mode):
                for data_source_name, data_source in data_sources.items():
                    tensors = data_source.output_tensors
                    for key, tensor in tensors.items():
                        summary_name = '%s/%s' % (data_source_name, key)
                        shape = tensor.shape.as_list()
                        num_dims = len(shape)
                        if num_dims == 4:  # Image data
                            if shape[1] == 1 or shape[1] == 3:
                                self.summary.image(summary_name, tensor,
                                                   data_format='channels_first')
                            else:
                                self.summary.image(summary_name, tensor,
                                                   data_format='channels_last')
                        elif num_dims == 2:
                            self.summary.histogram(summary_name, tensor)
                        else:
                            logger.debug('I do not know how to create a summary for %s (%s)' %
                                         (summary_name, tensor.shape.as_list()))

        def _build_train_or_test(mode):
            data_sources = self._train_data if mode == 'train' else self._test_data

            # Build model
            output_tensors, loss_terms, metrics = self.build_model(data_sources, mode=mode)

            # Record important tensors
            self.output_tensors[mode] = output_tensors
            self.loss_terms[mode] = loss_terms
            self.metrics[mode] = metrics

            # Create summaries for scalars
            if mode == 'train':
                for name, loss_term in loss_terms.items():
                    self.summary.scalar('loss/%s/%s' % (mode, name), loss_term)
                for name, metric in metrics.items():
                    self.summary.scalar('metric/%s/%s' % (mode, name), metric)

        # Build the main model
        _build_datasource_summaries(self._train_data, mode='train')
        _build_train_or_test(mode='train')
        logger.info('Built model for training.')

        # If there are any test data streams, build same model with different scope
        # Trainable parameters will be copied at test time
        if self._enable_live_testing:
            _build_datasource_summaries(self._test_data, mode='test')
            with tf.variable_scope('test'):
                _build_train_or_test(mode='test')
            logger.info('Built model for testing.')

            self._tester._post_model_build()  # Create copy ops to be run before every test run
        self.summary._post_model_build()  # Merge registered summary operations

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        raise NotImplementedError('BaseModel::build_model is not yet implemented.')

    def initialize_if_not(self, training=False):
        """Initialize variables and begin preprocessing threads."""
        if self._initialized:
            return

        # Build supporting operations
        self.checkpoint.build_savers()  # Create savers
        if training:
            self._build_optimizers()

            # Start pre-processing routines
            for _, datasource in self._train_data.items():
                datasource.create_and_start_threads()
            if len(self._test_data) > 0:
                for _, datasource in self._test_data.items():
                    datasource.create_and_start_threads()

        # Initialize all variables
        self._tensorflow_session.run(tf.global_variables_initializer())
        self._initialized = True

    def _build_optimizers(self):
        """Based on learning schedule, create optimizer instances."""
        self._optimize_ops = []
        all_trainable_variables = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            for spec in self._learning_schedule:
                optimize_ops = []
                loss_terms = spec['loss_terms_to_optimize']
                assert isinstance(loss_terms, dict)
                for loss_term_key, prefixes in loss_terms.items():
                    assert loss_term_key in self.loss_terms['train'].keys()
                    variables_to_train = []
                    for prefix in prefixes:
                        variables_to_train += [
                            v for v in all_trainable_variables
                            if v.name.startswith(prefix)
                        ]
                    optimize_op = tf.train.AdamOptimizer(
                        learning_rate=spec['learning_rate'],
                        # beta1=0.9,
                        # beta2=0.999,
                    ).minimize(
                        loss=self.loss_terms['train'][loss_term_key],
                        var_list=variables_to_train,
                        name='optimize_%s' % loss_term_key,
                    )
                    optimize_ops.append(optimize_op)
                self._optimize_ops.append(optimize_ops)
                logger.info('Built optimizer for: %s' % ', '.join(loss_terms.keys()))

    def train(self, num_epochs=None, num_steps=None):
        """Train model as requested."""
        if num_steps is None:
            num_entries = np.min([s.num_entries for s in list(self._train_data.values())])
            num_steps = int(num_epochs * num_entries / self._batch_size)
        logger.info('Training for %d steps' % num_steps)
        self.initialize_if_not(training=True)

        initial_step = self.checkpoint.load_all()
        if initial_step >= num_steps:
            # Let's return immediately, we will not even try another iteration to keep the checkpoints clean
            return

        current_step = initial_step
        for current_step in range(initial_step, num_steps):
            fetches = {}

            # Select loss terms, optimize operations, and metrics tensors to evaluate
            schedule_id = current_step % len(self._learning_schedule)
            schedule = self._learning_schedule[schedule_id]
            fetches['optimize_ops'] = self._optimize_ops[schedule_id]
            loss_term_keys, _ = zip(*list(schedule['loss_terms_to_optimize'].items()))
            fetches['loss_terms'] = [self.loss_terms['train'][k] for k in loss_term_keys]
            summary_ops = self.summary.get_ops(mode='train')
            if len(summary_ops) > 0:
                fetches['summaries'] = summary_ops

            # Run one optimization iteration and retrieve calculated loss values
            self.time.start('train_iteration', average_over_last_n_timings=100)
            outcome = self._tensorflow_session.run(
                fetches=fetches,
                feed_dict={
                    self.is_training: True,
                    self.use_batch_statistics: True,
                }
            )
            self.time.end('train_iteration')

            # Print progress
            to_print = 'Train: %07d/%07d> ' % (current_step, num_steps)
            to_print += ', '.join(['%s = %f' % (k, v)
                                   for k, v in zip(loss_term_keys, outcome['loss_terms'])])
            self.time.log_every('train_iteration', to_print, seconds=5)

            # Trigger copy weights & concurrent testing (if not already running)
            if self._enable_live_testing:
                self._tester.trigger_test_if_not_testing(current_step)

            # Write summaries
            if 'summaries' in outcome:
                self.summary.write_summaries(outcome['summaries'], current_step)

            # Save model weights
            if self.time.has_been_n_seconds_since_last('save_weights', 600):
                self.checkpoint.save_all(current_step)

        # Save final weights
        self.checkpoint.save_all(current_step)

    def evaluate_for_kaggle(self, data_source):
        """Evaluate on given data for Kaggle submission."""
        self.initialize_if_not()
        first_training_data_source = next(iter(self._train_data.values()))
        input_tensors = first_training_data_source.output_tensors

        predictions = []

        read_entry = data_source.entry_generator()
        num_entries = data_source.num_entries
        batch_size = data_source.batch_size
        num_batches = int(np.ceil(num_entries / batch_size))

        logger.info('Evaluating %d batches for Kaggle submission.' % num_batches)
        for i in range(num_batches):
            print("Got to batch {}".format(i))
            a = i * batch_size
            z = min(a + batch_size, num_entries)
            current_batch_size = z - a
            img_batch = []
            for _ in range(current_batch_size):
                preprocess_out = data_source.preprocess_entry(next(read_entry))
                img_batch.append(preprocess_out['img'])
            if current_batch_size < batch_size:
                difference = batch_size - current_batch_size
                img_batch = np.pad(img_batch, pad_width=[(0, difference), (0, 0), (0, 0), (0, 0)],
                                    mode='constant', constant_values=0.0)
            kp_batch = self._tensorflow_session.run(
                self.output_tensors['train']['kp_2D'],
                feed_dict={
                    input_tensors['img']: np.asarray(img_batch),
                    self.is_training: False,
                    self.use_batch_statistics: True,
                },
            )[:current_batch_size, :]
            predictions += list(kp_batch.reshape(-1, 2*21))

        output_file = '%s/to_submit_to_kaggle_%d.csv' % (self.output_path, time.time())
        coloumns = []
        for i in range(21):
            coloumns += ['Joint %d x' % i]
            coloumns += ['Joint %d y' % i]

        final_output = pd.DataFrame(predictions, columns=coloumns)
        final_output.index.name = 'Id'
        final_output.to_csv(output_file)
        logger.info('Created submission at %s' % os.path.relpath(output_file))
