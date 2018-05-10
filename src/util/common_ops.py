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
    def leaky_relu(cls, tensor, name='lReLU'):
        return tf.nn.leaky_relu(tensor, cls.SLOPE_LRU, name=name)

    @classmethod
    def conv(cls, in_tensor, name, kernel_size, out_chan, stride=1, trainable=True):
        return tf.layers.conv2d(
            inputs=in_tensor,
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
    def max_pool(cls, in_tensor, name='pool'):
        return tf.layers.max_pooling2d(
            inputs=in_tensor,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            data_format='channels_first',
            name=name
        )

    @classmethod
    def batch_normalization(cls, in_tensor, name='batch_norm', trainable=True):
        return tf.layers.batch_normalization(
            inputs=in_tensor,
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
    def dropout(cls, in_tensor, name='drop_single', trainable=True):
        return tf.layers.dropout(
            in_tensor,
            rate=cls.DROPOUT,
            noise_shape=None,
            seed=cls.SEED,
            training=trainable,
            name=name
        )

    @classmethod
    def dropout_layer(cls):
        return 0

    @classmethod
    def fully_connected(cls, in_tensor, name, out_chan, activation_fun=None, trainable=True):
        if activation_fun is tf.nn.leaky_relu:
            return tf.nn.leaky_relu(
                tf.contrib.layers.fully_connected(
                    inputs=in_tensor,
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
                inputs=in_tensor,
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
        tensor = cls.conv(in_tensor, layer_name, kernel_size, out_chan, stride, trainable)
        if maxpool:
            name = layer_name + '_pool'
            tensor = cls.max_pool(tensor, name=name)
            name = layer_name + '_relu'
            out_tensor = cls.leaky_relu(tensor, name=name)
        else:
            name = layer_name + '_batch_norm'
            tensor = cls.batch_normalization(in_tensor, name=name, trainable=trainable)
            name = layer_name + '_relu'
            tensor = cls.leaky_relu(tensor, name=name)
            name = layer_name + '_dropout'
            out_tensor = cls.dropout(tensor, name, trainable)
        return out_tensor

    @classmethod
    def fc_lru(cls, in_tensor, layer_name, out_chan, trainable=True):
        tensor = cls.fully_connected(in_tensor, layer_name, out_chan, tf.nn.leaky_relu, trainable)
        return cls.dropout(tensor, layer_name + '_drop', trainable)