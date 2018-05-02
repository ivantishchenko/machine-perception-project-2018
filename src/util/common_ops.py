import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import math

class NetworkOps(object):
    """ Operations that are frequently used within networks. """
    SLOPE_LRU = 0.01

    @classmethod
    def leaky_relu(cls, tensor, name='lReLU'):
        return tf.nn.leaky_relu(tensor, cls.SLOPE_LRU, name=name)

    @classmethod
    def conv(cls, in_tensor, layer_name, kernel_size, out_chan, stride=1, trainable=True):
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
            name=layer_name,
            reuse=None
        )

    @classmethod
    def conv_relu(cls, in_tensor, layer_name, kernel_size, out_chan, stride=1, trainable=True):
        tensor = cls.conv(in_tensor, layer_name, kernel_size, out_chan, stride, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

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
    def fully_connected(cls, in_tensor, layer_name, out_chan, activation_fun=None, trainable=True):
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
                name=layer_name
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
    def fc_lru(cls, in_tensor, layer_name, out_chan, trainable=True):
        return cls.fully_connected(in_tensor, layer_name, out_chan, tf.nn.leaky_relu, trainable)