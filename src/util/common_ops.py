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
            strides=stride,
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
    def upconv(cls, tensor, kernel_size, out_chan, stride=1, name='conv2d_t', trainable=True):
        return tf.layers.conv2d_transpose(
            inputs=tensor,
            filters=out_chan,
            kernel_size=kernel_size,
            strides=stride,
            padding='valid',
            data_format='channels_first',
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
    def max_pool(cls, tensor, pool=2, stride=2, name='max_pool'):
        return tf.layers.max_pooling2d(
            inputs=tensor,
            pool_size=pool,
            strides=stride,
            padding='same',
            data_format='channels_first',
            name=name
        )

    @classmethod
    def batch_normalization(cls, tensor, name='batch_norm', trainable=True):
        return tf.layers.batch_normalization(
            inputs=tensor,
            axis=cls.CHANNEL_AXIS,
            momentum=0.95,
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
    def dropout(cls, tensor, rate=-1., name='dropout', trainable=True):
        if rate == -1.:
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
    def conv_relu(cls, in_tensor, layer_name, kernel_size, out_chan, stride=1, maxpool=False, enable_scale=False, disable_dropout=True, trainable=True):
        with tf.variable_scope(layer_name):
            tensor = cls.conv(tensor=in_tensor, kernel_size=kernel_size, out_chan=out_chan, stride=stride, name='conv2d', trainable=trainable)
            if maxpool:
                # maxpool(relu(input)) == relu(maxpool(input)); this ordering is a tiny opt which is not general https://github.com/tensorflow/tensorflow/issues/3180#issuecomment-288389772
                tensor = cls.max_pool(tensor=tensor)
                tensor = cls.leaky_relu(tensor=tensor)
                tensor = cls.batch_normalization(tensor, trainable=trainable)  # https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
                if not disable_dropout:
                    tensor = cls.dropout(tensor=tensor, trainable=trainable)
            else:
                tensor = cls.leaky_relu(tensor=tensor)
                tensor = cls.batch_normalization(tensor, trainable=trainable)  # https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
                if not disable_dropout:
                    tensor = cls.dropout(tensor=tensor, trainable=trainable)
            return tensor

    @classmethod
    def fc_relu(cls, in_tensor, layer_name, out_chan, droprate=0.5, disable_dropout=False, trainable=True):
        with tf.variable_scope(layer_name):
            tensor = cls.fully_connected(tensor=in_tensor, out_chan=out_chan, activation_fun=tf.nn.leaky_relu, trainable=trainable)
            if not disable_dropout:
                tensor = cls.dropout(tensor=tensor, rate=droprate, trainable=trainable)
            return tensor


class ResNetOps(object):
    MINIMAL_LAYERS = 64
    KEYPOINTS = 21

    @classmethod
    def init_block(cls, in_tensor, trainable=True):
        with tf.variable_scope('resnet_init'):
            tensor = NetworkOps.conv(tensor=in_tensor, kernel_size=7, out_chan=64, stride=2, name='conv2d',
                              trainable=trainable)
            tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm')
            tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu')
            tensor = NetworkOps.max_pool(tensor=tensor, pool=3, stride=2)
            return tensor

    @classmethod
    def last_layer(cls, in_tensor, use_4k=False, trainable=True):
        with tf.variable_scope('resnet_last'):
            if use_4k:
                tensor = NetworkOps.conv(in_tensor, kernel_size=1, out_chan=2048, stride=1, name='conv2d_1',
                                         trainable=trainable)
                tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_1')
                tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_1')
                tensor = tf.layers.average_pooling2d(tensor, pool_size=4, strides=1, data_format='channels_first',
                                                     padding='same', name='average_pool')
                tensor = NetworkOps.conv(tensor, kernel_size=1, out_chan=4096, stride=1, name='conv2d_2',
                                         trainable=trainable)
                tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_2')
                tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')
                return NetworkOps.dropout(tensor=tensor, trainable=trainable, name='batch_norm_2')
            else:
                return tf.layers.average_pooling2d(in_tensor, pool_size=4, strides=1, data_format='channels_first',
                                                   padding='same', name='average_pool')

    @classmethod
    def output_layer(cls, in_tensor, use_4k=False, trainable=True):
        with tf.variable_scope('resnet_out'):
            if use_4k:
                result = NetworkOps.conv(in_tensor, kernel_size=1, out_chan=cls.KEYPOINTS * 2, stride=1, name='conv2d_1',
                                         trainable=trainable)
                result = NetworkOps.batch_normalization(tensor=result, name='batch_norm_1')
                return NetworkOps.leaky_relu(tensor=result, name='leaky_relu_1')
            else:
                result = tf.contrib.layers.flatten(in_tensor)
                return NetworkOps.fc_relu(result, 'fc_relu', out_chan=cls.KEYPOINTS * 2, disable_dropout=True, trainable=True)


    @classmethod
    def _vanilla_residual(cls, in_tensor, out_chan, strides=1, trainable=True):
        tensor = NetworkOps.conv(in_tensor, kernel_size=3, out_chan=out_chan, stride=strides, name='conv2d_1',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_1')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_1')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=3, out_chan=out_chan, stride=1, name='conv2d_2',
                                 trainable=trainable)
        return NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_2')

    @classmethod
    def _bottleneck_residual(cls, in_tensor, out_chan, strides=1, trainable=True):
        tensor = NetworkOps.conv(in_tensor, kernel_size=1, out_chan=out_chan, stride=strides, name='conv2d_1',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_1')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_1')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=3, out_chan=out_chan, stride=1, name='conv2d_2',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_2')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=1, out_chan=4 * out_chan, stride=1, name='conv2d_3',
                                 trainable=trainable)
        return NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_3')

    @classmethod
    def _inception_residual(cls, in_tensor, out_chan, strides=1, trainable=True):
        tensor = NetworkOps.conv(tensor=in_tensor, kernel_size=(1, 3), out_chan=out_chan, stride=(1, strides), name='conv2d_1',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_1')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_1')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, stride=(strides, 1), name='conv2d_2',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_2')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, stride=1, name='conv2d_3',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_3')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_3')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, stride=1, name='conv2d_4',
                                 trainable=trainable)
        return NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_4')

    @classmethod
    def _bottleneck_inception_residual(cls, in_tensor, out_chan, strides=1, trainable=True):
        tensor = NetworkOps.conv(in_tensor, kernel_size=1, out_chan=out_chan, stride=strides, name='conv2d_1',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_1')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_1')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, stride=1, name='conv2d_2',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_2')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, stride=1, name='conv2d_3',
                                 trainable=trainable)
        tensor = NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_3')
        tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_3')

        tensor = NetworkOps.conv(tensor=tensor, kernel_size=1, out_chan=4 * out_chan, stride=1, name='conv2d_4',
                                 trainable=trainable)
        return NetworkOps.batch_normalization(tensor=tensor, name='batch_norm_4')


    @classmethod
    def _shortcut(cls, in_tensor, out_chan, is_bottleneck, strides=1, trainable=True):
        if is_bottleneck:
            channels = 4 * out_chan
        else:
            channels = out_chan
        shortcut = NetworkOps.conv(in_tensor, kernel_size=1, out_chan=channels, stride=strides, name='conv2d',
                                   trainable=trainable)
        return NetworkOps.batch_normalization(tensor=shortcut, name='batch_norm', trainable=trainable)

    @classmethod
    def vanilla(cls, in_tensor, layer_name, out_chan, first_layer, trainable=True):
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == cls.MINIMAL_LAYERS:
                    strides = 1
                else:
                    strides = 2
                shortcut = cls._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=False, strides=strides, trainable=trainable)
                residual = cls._vanilla_residual(in_tensor, out_chan=out_chan, strides=strides, trainable=trainable)
            else:
                shortcut = in_tensor
                residual = cls._vanilla_residual(in_tensor, out_chan=out_chan, trainable=trainable)
            return tf.add(residual, shortcut)
            # tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')  # http://torch.ch/blog/2016/02/04/resnets.html

    @classmethod
    def inception(cls, in_tensor, layer_name, out_chan, first_layer, trainable=True):
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == cls.MINIMAL_LAYERS:
                    strides = 1
                else:
                    strides = 2
                shortcut = cls._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=False, strides=strides, trainable=trainable)
                residual = cls._inception_residual(in_tensor, out_chan=out_chan, strides=strides, trainable=trainable)
            else:
                shortcut = in_tensor
                residual = cls._inception_residual(in_tensor, out_chan=out_chan, trainable=trainable)
            return tf.add(residual, shortcut)
            # tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')  # http://torch.ch/blog/2016/02/04/resnets.html

    @classmethod
    def bottleneck(cls, in_tensor, layer_name, out_chan, first_layer, trainable=True):
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == cls.MINIMAL_LAYERS:
                    strides = 1
                else:
                    strides = 2
                shortcut = cls._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=True, strides=strides, trainable=trainable)
                residual = cls._bottleneck_residual(in_tensor, out_chan=out_chan, strides=strides, trainable=trainable)
            else:
                shortcut = in_tensor
                residual = cls._bottleneck_residual(in_tensor, out_chan=out_chan, trainable=trainable)
            return tf.add(residual, shortcut)
            # tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')  # http://torch.ch/blog/2016/02/04/resnets.html

    @classmethod
    def bottleneck_inception(cls, in_tensor, layer_name, out_chan, first_layer, trainable=True):
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == cls.MINIMAL_LAYERS // 2:
                    strides = 1
                else:
                    strides = 2
                shortcut = cls._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=True, strides=strides, trainable=trainable)
                residual = cls._bottleneck_inception_residual(in_tensor, out_chan=out_chan, strides=strides, trainable=trainable)
            else:
                shortcut = in_tensor
                residual = cls._bottleneck_inception_residual(in_tensor, out_chan=out_chan, trainable=trainable)
            return tf.add(residual, shortcut)
            # tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')  # http://torch.ch/blog/2016/02/04/resnets.html



class ImageOps(object):
    @classmethod
    def make_gaussian(cls, size, sigma=3, centre=None, normalized=False):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        # sigma = tf.mult(fwhm, tf.reciprocal(tf.sqrt(tf.mul(8, tf.log(2))))) # fwhm * 1/(sqrt(8*ln(2))) = sigma
        if normalized:
            norm_factor = 1 / 2*math.pi * sigma
        else:
            norm_factor = 1
        if centre is None:
            x0 = y0 = size // 2
        else:
            x0 = centre[0]
            y0 = centre[1]
        return norm_factor * tf.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))

    @classmethod
    def get_single_heatmap(cls, hand_joints, heatmap_size, gaussian_variance, scale_factor, normalized=False):
        gt_heatmap_np = []
        # invert_heatmap_np = tf.ones(shape=(heatmap_size, heatmap_size))  # See comment below
        for j in range(hand_joints.shape[0]):
            cur_joint_heatmap = cls.make_gaussian(heatmap_size,
                                                  gaussian_variance,
                                                  centre=(hand_joints[j] // scale_factor),
                                                  normalized=normalized)
            gt_heatmap_np.append(cur_joint_heatmap)
            # invert_heatmap_np -= cur_joint_heatmap  # Maybe we should include that but I don't see why; maybe background?
        return gt_heatmap_np, 0
