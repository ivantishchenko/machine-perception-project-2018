import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import math

class NetworkOps(object):
    """ Operations that are frequently used within networks. """
    SLOPE_LRU = 0.01
    CHANNEL_AXIS = 1
    DROPOUT = 0.2
    SEED = 42

    @classmethod
    def leaky_relu(cls, tensor, name='leaky_relu'):
        return tf.nn.leaky_relu(tensor, cls.SLOPE_LRU, name=name)

    @classmethod
    def conv(cls, tensor, kernel_size, out_chan, stride=1, name='conv2d', trainable=True):
        return tf.layers.conv2d(
            inputs=tensor,
            filters=out_chan,
            kernel_size=kernel_size,
            strides=(stride, stride),
            padding='same',
            data_format='channels_first',
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            bias_initializer=tf.constant_initializer(1e-4),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=trainable,
            name=name,
            reuse=None
        )

    @classmethod
    def max_pool(cls, tensor, name='max_pool'):
        return tf.layers.max_pooling2d(
            inputs=tensor,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            data_format='channels_first',
            name=name
        )

    @classmethod
    def batch_normalization(cls, tensor, name='batch_norm', trainable=True):
        return tf.layers.batch_normalization(
            inputs=tensor,
            axis=cls.CHANNEL_AXIS,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            training=trainable,
            trainable=trainable,
            name=name,
            reuse=not trainable,
            renorm=False,
            renorm_clipping=None,
            renorm_momentum=0.99,
            fused=None,
            virtual_batch_size=None,
            adjustment=None
        )

    @classmethod
    def dropout(cls, tensor, rate=-1, name='dropout', trainable=True):
        if rate == -1:
            droprate = cls.DROPOUT
        else:
            droprate = rate
        return tf.layers.dropout(
            tensor,
            rate=droprate,
            noise_shape=None,
            seed=cls.SEED,
            training=trainable,
            name=name
        )

    @classmethod
    def fully_connected(cls, tensor, out_chan, activation_fun=None, name='full_conn', trainable=True):
        if activation_fun is tf.nn.leaky_relu:
            return tf.nn.leaky_relu(
                tf.contrib.layers.fully_connected(
                    inputs=tensor,
                    num_outputs=out_chan,
                    activation_fn=None,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    weights_regularizer=None,
                    biases_initializer=tf.constant_initializer(1e-4),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=trainable,
                    scope=None
                ),
                alpha=cls.SLOPE_LRU,
                name=name
            )
        else:
            return tf.contrib.layers.fully_connected(
                inputs=tensor,
                num_outputs=out_chan,
                activation_fn=activation_fun,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer=None,
                biases_initializer=tf.constant_initializer(1e-4),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=trainable,
                scope=None
            )

    @classmethod
    def conv_relu(cls, in_tensor, layer_name, kernel_size, out_chan, maxpool=False, stride=1, trainable=True):
        with tf.variable_scope(layer_name):
            tensor = cls.conv(tensor=in_tensor, kernel_size=kernel_size, out_chan=out_chan, stride=stride, name='conv2d', trainable=trainable)
            if maxpool:
                # maxpool(relu(input)) == relu(maxpool(input)); this ordering is a tiny opt which is not general https://github.com/tensorflow/tensorflow/issues/3180#issuecomment-288389772
                tensor = cls.max_pool(tensor=tensor)
                tensor = cls.leaky_relu(tensor=tensor)
            else:
                tensor = cls.leaky_relu(tensor=tensor)
                tensor = cls.batch_normalization(tensor, trainable=trainable)  # https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
                tensor = cls.dropout(tensor=tensor, trainable=trainable)
            return tensor

    @classmethod
    def fc_lru(cls, in_tensor, layer_name, out_chan, droprate=0.5, disable_dropout=False, trainable=True):
        with tf.variable_scope(layer_name):
            tensor = cls.fully_connected(tensor=in_tensor, out_chan=out_chan, activation_fun=tf.nn.leaky_relu, trainable=trainable)
            if not disable_dropout:
                tensor = cls.dropout(tensor=tensor, rate=droprate, trainable=trainable)
            return tensor
