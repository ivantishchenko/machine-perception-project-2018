import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import math

class BasicLayers(object):
    """ Operations that are frequently used within networks. """
    SLOPE_LRU = 0.01
    CHANNEL_AXIS = 1
    DROPRATE = 0.2
    SEED = 42
    summary = None
    visualize = False

    def __init__(self, summary, visualize=False):
        self.summary = summary
        self.visualize = visualize

    def _leaky_relu(self, tensor, name='leaky_relu'):
        layer = tf.nn.leaky_relu(tensor, self.SLOPE_LRU, name=name)
        # print(name)
        # print('\n'.join(str(e) for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        # print('==========')
        if self.visualize:
            # self.summary.feature_maps(name, layer)
            self.summary.histogram(name + '/layer', layer)
        return layer

    def _conv(self, tensor, kernel_size, out_chan, is_training, stride=1, name='conv2d'):
        layer = tf.layers.conv2d(
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
            # trainable=is_training,
            name=name,
            reuse=None)
        # print(name)
        # print('\n'.join(str(e) for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        # print('==========')
        if self.visualize:
            self.summary.filters(name, layer)
            self.summary.feature_maps(name, layer)
            self.summary.histogram(name + '/layer', layer)
            with tf.variable_scope(name, reuse=True):
                kernel = tf.get_variable('kernel')
                bias = tf.get_variable('bias')
                self.summary.histogram(name + '/kernel', kernel)
                self.summary.histogram(name + '/bias', bias)
        return layer

    def _upconv(self, tensor, kernel_size, out_chan, is_training, stride=1, name='conv2d_t'):
        '''
        Note: Play around with the stride, then blowing it up, but not kernel sizes
        :param tensor:
        :param kernel_size:
        :param out_chan:
        :param stride:
        :param name:
        :return:
        '''
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
            # trainable=is_training,
            name=name,
            reuse=None
        )

    def _max_pool(self, tensor, pool=2, stride=2, name='max_pool'):
        layer = tf.layers.max_pooling2d(
            inputs=tensor,
            pool_size=pool,
            strides=stride,
            padding='same',
            data_format='channels_first',
            name=name
        )
        if self.visualize:
            self.summary.feature_maps(name, layer)
            self.summary.histogram(name + '/layer', layer)
        return layer

    def _batch_normalization(self, tensor, is_training, use_batch_stats=True, name='batch_norm'):
        layer = tf.layers.batch_normalization(
            inputs=tensor,
            axis=self.CHANNEL_AXIS,
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
            training=is_training,  # use batch stats?
            name=name,
            reuse=None,
            renorm=False,
            renorm_clipping=None,
            renorm_momentum=0.99,
            fused=None,
            virtual_batch_size=None,
            adjustment=None
        )
        if self.visualize:
            self.summary.feature_maps(name, layer)
            self.summary.histogram(name + '/layer', layer)
            with tf.variable_scope(name, reuse=True):
                gamma = tf.get_variable('gamma')
                beta = tf.get_variable('beta')
                self.summary.histogram(name + '/gamma', gamma)
                self.summary.histogram(name + '/beta', beta)
                # The following variables are not of interest really, they stay constant from the looks
                # moving_mean = tf.get_variable('moving_mean')
                # moving_variance = tf.get_variable('moving_variance')
                # self.summary.histogram(name + '/moving_mean', moving_mean)
                # self.summary.histogram(name + '/moving_variance', moving_variance)
        return layer
        # return tf.contrib.layers.batch_norm(
        #     inputs=tensor,
        #     decay=0.95,  # 0.999
        #     center=True,
        #     scale=True,  # False
        #     epsilon=0.001,
        #     activation_fn=None,
        #     param_initializers=None,
        #     param_regularizers=None,
        #     updates_collections=tf.GraphKeys.UPDATE_OPS,
        #     is_training=True,  # True
        #     reuse=None,
        #     variables_collections=None,
        #     outputs_collections=None,
        #     trainable=True,
        #     batch_weights=None,
        #     fused=None,
        #     data_format='NCHW',  #DATA_FORMAT_NHWC
        #     zero_debias_moving_mean=False,
        #     scope=None,
        #     renorm=False,
        #     renorm_clipping=None,
        #     renorm_decay=0.99,
        #     adjustment=None
        # )

    def _dropout(self, tensor, is_training, rate=-1., name='dropout'):
        if rate == -1.:
            droprate = self.DROPRATE
        else:
            droprate = rate
        layer = tf.layers.dropout(
            tensor,
            rate=droprate,
            noise_shape=None,
            seed=self.SEED,
            training=is_training,
            name=name
        )
        print(name)
        print('\n'.join(str(e) for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        print('==========')
        if self.visualize:
            self.summary.feature_maps(name, layer)
            self.summary.histogram(name + '/layer', layer)
        return layer

    def _fully_connected(self, tensor, out_chan, is_training, activation_fun=None, name='full_conn'):
        '''
        Method used to generate a fully connected layer with either our leaky_relu function or vanilla tf functions
        :param tensor: Input tensor on which to apply a fully connected layer on
        :param out_chan: The number of output neurons
        :param activation_fun: The activation function (leaky_relu leads to this class' defined leaky relu usage)
        :param name: Name of this layer (not used)
        :param trainable: Is this layer trainable (i.e. adjust weights and biases)
        :return: The application of a fully connected layer with the respective activation function on the input
        '''
        if activation_fun is tf.nn.leaky_relu:
            layer = tf.contrib.layers.fully_connected(
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
                    # trainable=is_training,
                    scope=None
            )
            # print(name)
            # print('\n'.join(str(e) for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
            # print('==========')
            if self.visualize:
                # self.summary.feature_maps(name, layer)
                self.summary.histogram(name + '/layer', layer)
                with tf.variable_scope('fully_connected', reuse=True):
                    weights = tf.get_variable('weights')
                    biases = tf.get_variable('biases')
                    self.summary.histogram(name + '/weights', weights)
                    self.summary.histogram(name + '/biases', biases)
            layer = self._leaky_relu(layer)
        else:
            layer = tf.contrib.layers.fully_connected(
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
                # trainable=is_training,
                scope=None
            )
            print(name)
            print('\n'.join(str(e) for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
            print('==========')
        return layer

    def conv_relu(self, in_tensor, layer_name, kernel_size, out_chan, is_training, stride=1, maxpool=False, disable_dropout=True):
        '''
        Method to generate a 2d convolutional layer with various extra components (such as max_pool, leaky_relu, batch_norm and dropout)
        :param tensor: Input tensor on which to apply a 2d convolutional layer
        :param layer_name: Name of this layer and the scope of it
        :param kernel_size: Size of the convolutional kernel (accepts tuple or single int)
        :param out_chan: The number of output channels
        :param stride: Stride size of the kernel on the convolutional layer
        :param maxpool: Enable max_pooling?
        :param disable_dropout: Disable dropout (usually turned on)
        :return: The application of a convolutional block with the respective extraneous operations on the input
        '''
        with tf.variable_scope(layer_name):
            tensor = self._conv(tensor=in_tensor, kernel_size=kernel_size, out_chan=out_chan, is_training=is_training, stride=stride)
            if maxpool:
                # maxpool(relu(input)) == relu(maxpool(input)); this ordering is a tiny opt which is not general https://github.com/tensorflow/tensorflow/issues/3180#issuecomment-288389772
                tensor = self._max_pool(tensor=tensor)
            tensor = self._leaky_relu(tensor=tensor)
            tensor = self._batch_normalization(tensor=tensor, is_training=is_training)  # https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
            if not disable_dropout:
                tensor = self._dropout(tensor=tensor, is_training=is_training)
            return tensor

    def fc_relu(self, in_tensor, layer_name, out_chan, is_training, droprate=0.5, disable_dropout=False):
        '''
        Method used to generate a fully connected layer with leaky_relu and, if not otherwise requested, dropout of 0.5
        :param in_tensor: Input tensor on which to apply a fully connected layer on
        :param layer_name: Name of this layer and the scope of it
        :param out_chan: The number of output neurons
        :param droprate: The droprate of the dropout layer
        :param disable_dropout: Disable dropout (usually turned on)
        :param trainable: Is this trainable (i.e. adjust weights and biases)
        :return: The application of a fully connected leaky relu layer with potential dropout
        '''
        with tf.variable_scope(layer_name):
            tensor = self._fully_connected(tensor=in_tensor, out_chan=out_chan, is_training=is_training, activation_fun=tf.nn.leaky_relu)
            if not disable_dropout:
                tensor = self._dropout(tensor=tensor, is_training=is_training, rate=droprate)
            return tensor


class ResNetLayers(BasicLayers):
    MINIMAL_LAYERS = 64
    KEYPOINTS = 21

    def __init__(self, summary, visualize=False):
        super().__init__(summary, visualize)

    def init_block(self, in_tensor, is_training):
        with tf.variable_scope('resnet_init'):
            tensor = self._conv(tensor=in_tensor, kernel_size=7, out_chan=64, is_training=is_training, stride=2)
            tensor = self._batch_normalization(tensor=tensor, is_training=is_training)
            tensor = self._leaky_relu(tensor=tensor)
            tensor = self._max_pool(tensor=tensor, pool=3, stride=2)
            return tensor

    def last_layer(self, in_tensor, is_training, use_4k=False):
        with tf.variable_scope('resnet_last'):
            if use_4k:
                tensor = self._conv(in_tensor, kernel_size=1, out_chan=2048, is_training=is_training, stride=1, name='conv2d-1')
                tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
                tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')
                tensor = tf.layers.average_pooling2d(tensor, pool_size=4, strides=1, data_format='channels_first',
                                                     padding='same', name='average_pool')
                tensor = self._conv(tensor, kernel_size=1, out_chan=4096, is_training=is_training, stride=1, name='conv2d-2')
                tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
                tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')
                return self._dropout(tensor=tensor, is_training=is_training, name='dropout-2')
            else:
                return tf.layers.average_pooling2d(in_tensor, pool_size=4, strides=1, data_format='channels_first',
                                                   padding='same', name='average_pool')

    def output_layer(self, in_tensor, is_training, use_4k=False):
        with tf.variable_scope('resnet_out'):
            if use_4k:
                result = self._conv(in_tensor, kernel_size=1, out_chan=ResNetLayers.KEYPOINTS * 2, is_training=is_training, stride=1)
                result = self._batch_normalization(tensor=result, is_training=is_training)
                return self._leaky_relu(tensor=result)
            else:
                result = tf.contrib.layers.flatten(in_tensor)
                return self.fc_relu(result, 'fc_relu', out_chan=ResNetLayers.KEYPOINTS * 2, is_training=is_training, disable_dropout=True)

    def _vanilla_residual(self, in_tensor, out_chan, is_training, strides=1):
        tensor = self._conv(in_tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=strides, name='conv2d-1')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')

        tensor = self._conv(tensor=tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=1, name='conv2d-2')
        return self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')

    def _bottleneck_residual(self, in_tensor, out_chan, is_training, strides=1):
        tensor = self._conv(in_tensor, kernel_size=1, out_chan=out_chan, stride=strides, is_training=is_training, name='conv2d-1')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')

        tensor = self._conv(tensor=tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=1, name='conv2d-2')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')

        tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=4 * out_chan, is_training=is_training, stride=1, name='conv2d-3')
        return self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-3')

    def _inception_residual(self, in_tensor, out_chan, is_training, strides=1):
        tensor = self._conv(tensor=in_tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training, stride=(1, strides), name='conv2d-1')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')

        tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training, stride=(strides, 1), name='conv2d-2')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')

        tensor = self._conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training, stride=1, name='conv2d-3')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-3')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-3')

        tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training, stride=1, name='conv2d-4')
        return self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-4')

    def _bottleneck_inception_residual(self, in_tensor, out_chan, is_training, strides=1):
        tensor = self._conv(in_tensor, kernel_size=1, out_chan=out_chan, is_training=is_training, stride=strides, name='conv2d-1')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')

        tensor = self._conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training, stride=1, name='conv2d-2')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')

        tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training, stride=1, name='conv2d-3')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-3')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-3')

        tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=4 * out_chan, is_training=is_training, stride=1, name='conv2d-4')
        return self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-4')

    def _shortcut(self, in_tensor, out_chan, is_bottleneck, is_training, strides=1):
        if is_bottleneck:
            channels = 4 * out_chan
        else:
            channels = out_chan
        shortcut = self._conv(in_tensor, kernel_size=1, out_chan=channels, is_training=is_training, stride=strides)
        return self._batch_normalization(tensor=shortcut, is_training=is_training)

    def vanilla(self, in_tensor, layer_name, out_chan, first_layer, is_training):
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == ResNetLayers.MINIMAL_LAYERS:
                    strides = 1
                else:
                    strides = 2
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=False, is_training=is_training, strides=strides)
                residual = self._vanilla_residual(in_tensor, out_chan=out_chan, is_training=is_training, strides=strides)
            else:
                shortcut = in_tensor
                residual = self._vanilla_residual(in_tensor, out_chan=out_chan, is_training=is_training)
            return tf.add(residual, shortcut)
            # tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')  # http://torch.ch/blog/2016/02/04/resnets.html

    def inception(self, in_tensor, layer_name, out_chan, first_layer, is_training):
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == ResNetLayers.MINIMAL_LAYERS:
                    strides = 1
                else:
                    strides = 2
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=False, is_training=is_training, strides=strides)
                residual = self._inception_residual(in_tensor, out_chan=out_chan, is_training=is_training, strides=strides)
            else:
                shortcut = in_tensor
                residual = self._inception_residual(in_tensor, out_chan=out_chan, is_training=is_training)
            return tf.add(residual, shortcut)
            # tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')  # http://torch.ch/blog/2016/02/04/resnets.html

    def bottleneck(self, in_tensor, layer_name, out_chan, first_layer, is_training):
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == ResNetLayers.MINIMAL_LAYERS:
                    strides = 1
                else:
                    strides = 2
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=True, is_training=is_training, strides=strides)
                residual = self._bottleneck_residual(in_tensor, out_chan=out_chan, is_training=is_training, strides=strides)
            else:
                shortcut = in_tensor
                residual = self._bottleneck_residual(in_tensor, out_chan=out_chan, is_training=is_training)
            return tf.add(residual, shortcut)
            # tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')  # http://torch.ch/blog/2016/02/04/resnets.html

    def bottleneck_inception(self, in_tensor, layer_name, out_chan, first_layer, is_training):
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == ResNetLayers.MINIMAL_LAYERS // 2:
                    strides = 1
                else:
                    strides = 2
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=True, is_training=is_training, strides=strides)
                residual = self._bottleneck_inception_residual(in_tensor, out_chan=out_chan, is_training=is_training, strides=strides)
            else:
                shortcut = in_tensor
                residual = self._bottleneck_inception_residual(in_tensor, out_chan=out_chan, is_training=is_training)
            return tf.add(residual, shortcut)
            # tensor = NetworkOps.leaky_relu(tensor=tensor, name='leaky_relu_2')  # http://torch.ch/blog/2016/02/04/resnets.html


class LossFuncs(object):

    @classmethod
    def flat_mse(cls, ground_truth, predictions, layer_name):
        with tf.variable_scope(layer_name):
            return tf.losses.mean_squared_error(ground_truth, predictions)

    @classmethod
    def kp_mse(cls, ground_truth, predictions, layer_name):
        with tf.variable_scope(layer_name):
            return tf.losses.mean_squared_error(ground_truth, predictions, axis=1)

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
