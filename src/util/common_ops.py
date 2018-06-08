import tensorflow as tf
import numpy as np
import math


class BasicLayers(object):
    """
    Helper class to quickly and consistently call frequently used operations within neural/convolutional networks.
    """
    SLOPE_LRU = 0.01
    CHANNEL_AXIS = 1
    DROPRATE = 0.2
    SEED = 42
    KEYPOINTS = 21
    summary = None
    visualize = False

    def __init__(self, summary, visualize=False):
        self.summary = summary
        self.visualize = visualize

    def _leaky_relu(self, tensor, name='leaky_relu'):
        """
        Apply a leaky-relu activation layer to the input tensor.
        :param tensor: Tensor which to apply leaky-relu on
        :param name: Name of this layer
        :return: Application of leaky-relu on the layer

        """
        layer = tf.nn.leaky_relu(tensor, self.SLOPE_LRU, name=name)
        if self.visualize:
            # self.summary.feature_maps(name, layer)
            self.summary.histogram(name + '/layer', layer)
        return layer

    def _conv(self, tensor, kernel_size, out_chan, is_training, stride=1, dilation=1, name='conv2d'):
        """
        Apply a convolutional layer to the input tensor.
        :param tensor: Tensor which to convolve
        :param kernel_size: Size of the kernel (Accepts int and tuple)
        :param out_chan: Number of features to extract
        :param is_training: Is this layer in training mode?
        :param stride: Stride of the convolution (Accepts int and tuple)
        :param dilation: Dilation of the kernel (Accepts int and tuple)
        :param name: Name of this layer
        :return: Application of convolution on the layer
        """
        def __conv(bool_training=True):
            return tf.layers.conv2d(
                inputs=tensor,
                filters=out_chan,
                kernel_size=kernel_size,
                strides=stride,
                padding='same',
                data_format='channels_first',
                dilation_rate=dilation,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                bias_initializer=tf.constant_initializer(1e-4),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=bool_training,
                name=name,
                reuse=tf.AUTO_REUSE
            )

        layer = __conv()
        if self.visualize:
            # self.summary.filters(name, layer)
            # self.summary.feature_maps(name, layer)
            # self.summary.histogram(name + '/layer', layer)
            with tf.variable_scope(name, reuse=True):
                kernel = tf.get_variable('kernel', [kernel_size, kernel_size, tensor.shape[1], out_chan])
                bias = tf.get_variable('bias', [out_chan])
                self.summary.histogram(name + '/kernel', kernel)
                self.summary.histogram(name + '/bias', bias)
        return layer

    def _upconv(self, tensor, kernel_size, out_chan, is_training, stride=1, name='conv2d_t'):
        """
        Apply a transpose-convolution layer to the input tensor.
        Note: Play around with the stride, then blowing it up, but not kernel sizes
        :param tensor: Tensor which to transpose-convolve
        :param kernel_size: Size of the kernel (Accepts int and tuple)
        :param out_chan: Number of features to extract
        :param is_training: Is this layer in training mode?
        :param stride: Stride of the transpose-convolution (Accepts int and tuple)
        :param name: Name of this layer
        :return: Application of transpose-convolution on the layer
        """
        def __upconv(bool_training=True):
            return tf.layers.conv2d_transpose(
                inputs=tensor,
                filters=out_chan,
                kernel_size=kernel_size,
                strides=stride,
                padding='same',
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
                trainable=bool_training,
                name=name,
                reuse=tf.AUTO_REUSE
            )

        layer = __upconv(True)
        if self.visualize:
            # self.summary.filters(name, layer)
            # self.summary.feature_maps(name, layer)
            # self.summary.histogram(name + '/layer', layer)
            with tf.variable_scope(name, reuse=True):
                kernel = tf.get_variable('kernel')
                bias = tf.get_variable('bias')
                self.summary.histogram(name + '/kernel', kernel)
                self.summary.histogram(name + '/bias', bias)
        return layer

    def _max_pool(self, tensor, pool=2, stride=2, name='max_pool'):
        """
        Apply a max-pooling on the input tensor.
        :param tensor: Tensor on which to apply max-pooling on
        :param pool: Size of pooling (Accepts int and tuple)
        :param stride: Stride size of pooling (Accepts int and tuple)
        :param name: Name of this layer
        :return: Application of max-pooling on the layer
        """
        layer = tf.layers.max_pooling2d(
            inputs=tensor,
            pool_size=pool,
            strides=stride,
            padding='same',
            data_format='channels_first',
            name=name
        )
        if self.visualize:
            # self.summary.feature_maps(name, layer)
            self.summary.histogram(name + '/layer', layer)
        return layer

    def _batch_normalization(self, tensor, is_training, use_batch_stats=True, name='batch_norm'):
        """
        Apply batch-normalization to the input tensor.
        :param tensor: Tensor on which to apply batch-normalization on
        :param is_training: Is this layer in training mode?
        :param use_batch_stats: Non-used variable for the moment
        :return: Application of batch-normalization on the layer
        """
        __momentum = 0.97
        __scale = True

        def __official_bn():
            return tf.layers.batch_normalization(
                inputs=tensor,
                axis=self.CHANNEL_AXIS,
                momentum=__momentum,
                epsilon=0.001,
                center=True,
                scale=__scale,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                training=True,
                name=name,
                reuse=None,
                renorm=False,
                renorm_clipping=None,
                renorm_momentum=0.99,
                fused=True,
                virtual_batch_size=None,
                adjustment=None
            )

        def __contrib_bn():
            return tf.contrib.layers.batch_norm(
                inputs=tensor,
                decay=__momentum,  # 0.999
                center=True,
                scale=__scale,  # False
                epsilon=0.001,
                activation_fn=None,
                param_initializers=None,
                param_regularizers=None,
                updates_collections=None,  # tf.GraphKeys.UPDATE_OPS,
                is_training=True,  # is_training would be more correct but validation performance is strange with it
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                batch_weights=None,
                fused=None,
                data_format='NCHW',  # DATA_FORMAT_NHWC
                zero_debias_moving_mean=False,
                scope=None,
                renorm=False,
                renorm_clipping=None,
                renorm_decay=0.99,
                adjustment=None
            )

        layer = __contrib_bn()
        if self.visualize:
            name = 'BatchNorm'
            # self.summary.feature_maps(name, layer)
            # self.summary.histogram(name + '/layer', layer)
            with tf.variable_scope(name, reuse=True):
                beta = tf.get_variable('beta')
                self.summary.histogram(name + '/beta', beta)
                gamma = tf.get_variable('gamma')
                self.summary.histogram(name + '/gamma', gamma)
                # The following variables are not of interest really, they stay constant from the looks
                moving_mean = tf.get_variable('moving_mean')
                moving_variance = tf.get_variable('moving_variance')
                self.summary.histogram(name + '/moving_mean', moving_mean)
                self.summary.histogram(name + '/moving_variance', moving_variance)
        return layer

    def _dropout(self, tensor, is_training, rate=-1., name='dropout'):
        """
        Apply dropout to the input tensor.
        :param tensor: Tensor on which to apply dropout on
        :param is_training: Is this layer in training mode?
        :param rate: Set the amount of drop of this layer (between [0.0, 1.0[)
        :param name: Name of this layer
        :return: Application of dropout on the layer
        """
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
            # self.summary.feature_maps(name, layer)
            self.summary.histogram(name + '/layer', layer)
        return layer

    def _fully_connected(self, tensor, out_chan, is_training, activation_fun=None, name='full_conn'):
        """
        Apply a fully connected layer with either custom leaky_relu or requested activation_fn on the input tensor.
        :param tensor: Tensor on which to apply a fully connected layer on
        :param out_chan: Number of output neurons
        :param activation_fun: Activation function (leaky_relu uses custom leaky relu)
        :param name: Name of this layer (not used)
        :param is_training: Is this layer in training mode?
        :return: Application of a fully connected layer on the input
        """

        def __fc(bool_training=True, act_fun=None):
            return tf.contrib.layers.fully_connected(
                inputs=tensor,
                num_outputs=out_chan,
                activation_fn=act_fun,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer=None,
                biases_initializer=tf.constant_initializer(1e-4),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=bool_training,
                scope=None
            )

        if activation_fun is tf.nn.leaky_relu:
            layer = __fc(act_fun=None)
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
            layer = __fc(act_fun=activation_fun)
            print(name)
            print('\n'.join(str(e) for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
            print('==========')
        return layer

    def conv_layer(self, in_tensor, layer_name, kernel_size, out_chan, is_training, stride=1, max_pool=False,
                   disable_dropout=True):
        """
        Wrapper to apply 2d-convolutions with various extra ops (max_pool, leaky_relu, batch_norm, dropout).
        :param in_tensor: Tensor on which to apply this wrapper on
        :param layer_name: Name of this layer and related scope
        :param kernel_size: Size of the convolutional kernel (accepts tuple or single int)
        :param out_chan: Number of features to extract
        :param is_training: Is this layer in training mode?
        :param stride: Stride size of the kernel on the convolutional layer
        :param max_pool: Enable max_pooling?
        :param disable_dropout: Disable dropout (usually turned on)
        :return: Application of this wrapper with requested ops on the input
        """
        with tf.variable_scope(layer_name):
            tensor = self._conv(tensor=in_tensor, kernel_size=kernel_size, out_chan=out_chan, is_training=is_training,
                                stride=stride)
            if max_pool:
                # maxpool(relu(input)) == relu(maxpool(input)); this ordering is a tiny opt which is not general
                # https://github.com/tensorflow/tensorflow/issues/3180#issuecomment-288389772
                tensor = self._max_pool(tensor=tensor)
            tensor = self._leaky_relu(tensor=tensor)
            # https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
            tensor = self._batch_normalization(tensor=tensor, is_training=is_training)
            if not disable_dropout:
                tensor = self._dropout(tensor=tensor, is_training=is_training)
            return tensor

    def conv_relu(self, in_tensor, layer_name, kernel_size, out_chan, is_training, stride=1, max_pool=False):
        """
        Wrapper to apply 2d-convolutions with a select few ops (leaky_relu, max_pool).
        :param in_tensor: Tensor on which to apply this wrapper on
        :param layer_name: Name of this layer and related scope
        :param kernel_size: Size of the convolutional kernel (accepts tuple or single int)
        :param out_chan: Number of features to extract
        :param is_training: Is this layer in training mode?
        :param stride: Stride size of the kernel on the convolutional layer
        :param max_pool: Enable max_pooling?
        :return: Application of this wrapper with requested ops on the input
        """
        with tf.variable_scope(layer_name):
            tensor = self._conv(tensor=in_tensor, kernel_size=kernel_size, out_chan=out_chan, is_training=is_training,
                                stride=stride)
            tensor = self._leaky_relu(tensor=tensor)
            if max_pool:
                tensor = self._max_pool(tensor=tensor)
            return tensor

    def fc_relu(self, in_tensor, layer_name, out_chan, is_training, droprate=0.5, disable_dropout=False):
        """
        Wrapper for a fully connected layer with leaky_relu and default dropout of 0.5
        :param in_tensor: Input tensor on which to apply this wrapper on
        :param layer_name: Name of this layer and related scope
        :param out_chan: Number of output neurons
        :param droprate: Droprate of the dropout layer
        :param disable_dropout: Disable dropout (usually turned on)
        :param is_training: Is this layer in training mode?
        :return: Application of a fully connected leaky relu layer with potential dropout
        """
        with tf.variable_scope(layer_name):
            tensor = self._fully_connected(tensor=in_tensor, out_chan=out_chan, is_training=is_training,
                                           activation_fun=tf.nn.leaky_relu)
            if not disable_dropout:
                tensor = self._dropout(tensor=tensor, is_training=is_training, rate=droprate)
            return tensor

    def pred_layer(self, in_tensor, is_training):
        with tf.variable_scope('pred'):
            result = tf.contrib.layers.flatten(in_tensor)
            result = self.fc_relu(result, 'fc_relu', out_chan=BasicLayers.KEYPOINTS * 2, is_training=is_training,
                                  disable_dropout=True)
            return tf.reshape(result, (-1, 21, 2))


class ResNetLayers(BasicLayers):
    """
    A helper class to quickly generate different residual networks. The blocks follow the full pre-activation principle.
    Additionally stochastic depth can be enabled for each of the building blocks when called using public functions.
    """
    # http://torch.ch/blog/2016/02/04/resnets.html
    MINIMAL_FEATURES = 64
    FULL_PREACTIVATION = False

    def __init__(self, summary, visualize=False, minimal_features=64, full_preactivation=False):
        """
        Default constructor.
        :param summary: Summary object to allow logging.
        :param visualize: Enable logging?
        """
        super().__init__(summary, visualize)
        self.MINIMAL_FEATURES = minimal_features
        self.FULL_PREACTIVATION = full_preactivation

    @staticmethod
    def _get_survival_rate(depth, survival_last_block=0.5):
        """
        Helper function to calculate the survival rate of a block for stochastic depth.
        https://arxiv.org/abs/1603.09382
        :param depth: Tuple of (current_depth, total_depth)
        :param survival_last_block: Survival rate of the last block
        :return: Survival rate of the current layer
        """
        return 1 - (depth[0] / depth[1]) * (1 - survival_last_block)

    def init_block(self, in_tensor, is_training):
        """
        Default initial layer for all residual networks.
        :param in_tensor: Tensor on which to apply this block on
        :param is_training: Is this block in training mode?
        :return: Application of the initial layer block on the input
        """
        with tf.variable_scope('resnet_init'):
            tensor = self._conv(tensor=in_tensor, kernel_size=7, out_chan=64, is_training=is_training, stride=2)
            tensor = self._batch_normalization(tensor=tensor, is_training=is_training)
            tensor = self._leaky_relu(tensor=tensor)
            tensor = self._max_pool(tensor=tensor, pool=3, stride=2)
            return tensor

    def last_layer(self, in_tensor, is_training, use_4k=False, use_upconv=False):
        """
        Default last layer for all residual networks.
        :param in_tensor: Tensor on which to apply this block on
        :param is_training: Is this layer in training mode?
        :param use_4k: Use an especially wide last layer with 2048 features prior and 4096 post average pooling
        :param use_upconv: Have a transpose convolution upsampling before average pooling
        :return: Application of the last layer block on the input
        """
        with tf.variable_scope('resnet_last'):
            tensor = in_tensor
            if use_4k:
                if not self.FULL_PREACTIVATION:
                    tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=2048, is_training=is_training, stride=1,
                                        name='conv2d-1')
                tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
                tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')
                if self.FULL_PREACTIVATION:
                    tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=2048, is_training=is_training, stride=1,
                                        name='conv2d-1')
                tensor = tf.layers.average_pooling2d(tensor, pool_size=4, strides=1, data_format='channels_first',
                                                     padding='valid', name='average_pool')
                if not self.FULL_PREACTIVATION:
                    tensor = self._conv(tensor, kernel_size=1, out_chan=4096, is_training=is_training, stride=1,
                                        name='conv2d-2')
                tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
                tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')
                if self.FULL_PREACTIVATION:
                    tensor = self._conv(tensor, kernel_size=1, out_chan=4096, is_training=is_training, stride=1,
                                        name='conv2d-2')
                return self._dropout(tensor=tensor, is_training=is_training, name='dropout-2')
            elif use_upconv:
                if not self.FULL_PREACTIVATION:
                    tensor = self._upconv(tensor=tensor, kernel_size=3, out_chan=in_tensor.shape[1],
                                          is_training=is_training, stride=2, name='conv2d_t')
                tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm')
                tensor = self._leaky_relu(tensor=tensor, name='leaky_relu')
                if self.FULL_PREACTIVATION:
                    tensor = self._upconv(tensor=tensor, kernel_size=3, out_chan=in_tensor.shape[1],
                                          is_training=is_training, stride=2, name='conv2d_t')
                return tf.layers.average_pooling2d(tensor, pool_size=4, strides=1, data_format='channels_first',
                                                   padding='valid', name='average_pool')
            else:
                return tf.layers.average_pooling2d(tensor, pool_size=4, strides=1, data_format='channels_first',
                                                   padding='valid', name='average_pool')

    def prediction_layer(self, in_tensor, is_training, use_4k=False):

        with tf.variable_scope('resnet_pred'):
            result = in_tensor
            if use_4k:
                if not self.FULL_PREACTIVATION:
                    result = self._conv(tensor=result, kernel_size=1, out_chan=BasicLayers.KEYPOINTS * 2,
                                        is_training=is_training, stride=1)
                result = self._batch_normalization(tensor=result, is_training=is_training)
                result = self._leaky_relu(tensor=result)
                if self.FULL_PREACTIVATION:
                    result = self._conv(tensor=result, kernel_size=1, out_chan=BasicLayers.KEYPOINTS * 2,
                                        is_training=is_training, stride=1)
                result = tf.reshape(result, (-1, 21, 2))
            else:
                result = self.pred_layer(result, is_training)
            return result

    def _vanilla_branch(self, in_tensor, out_chan, is_training, strides=1, dilation=1):
        """
        Default building block branch of a residual network using full pre-activation.
        :param in_tensor: Tensor on which to apply this block on
        :param out_chan: Number of features to extract
        :param is_training: Is this block in training mode?
        :param strides: The stride size for the first convolutional layer
        :param dilation: The dilation rate for both convolutional layers
        :return: Application of a default residual building block branch on the input
        """
        tensor = in_tensor
        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=strides,
                            dilation=dilation, name='conv2d-1')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=strides,
                            dilation=dilation, name='conv2d-1')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=dilation, name='conv2d-2')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
        if not self.FULL_PREACTIVATION:
            return tensor
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=dilation, name='conv2d-2')
        return tensor

    def _bottleneck_branch(self, in_tensor, out_chan, is_training, strides=1, dilation=1):
        """
        Bottleneck building block branch of a residual network using full pre-activation.
        :param in_tensor: Tensor on which to apply this block on
        :param out_chan: Number of features to extract
        :param is_training: Is this block in training mode?
        :param strides: The stride size for the center convolutional layer
        :param dilation: The dilation rate for the center convolutional layer
        :return: Application of a bottleneck residual building block branch on the input
        """
        tensor = in_tensor
        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=out_chan, stride=strides,
                                is_training=is_training, name='conv2d-1')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=out_chan, stride=strides,
                                is_training=is_training, name='conv2d-1')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=dilation, name='conv2d-2')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=3, out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=dilation, name='conv2d-2')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=4 * out_chan, is_training=is_training, stride=1,
                                name='conv2d-3')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-3')
        if not self.FULL_PREACTIVATION:
            return tensor
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=4 * out_chan, is_training=is_training, stride=1,
                                name='conv2d-3')
        return tensor

    def _inception_branch(self, in_tensor, out_chan, is_training, strides=1, dilation=1):
        """
        Normal building block branch of a residual network using full pre-activation and an InceptionNet approach to
        kernels.
        :param in_tensor: Tensor on which to apply this block on
        :param out_chan: Number of features to extract
        :param is_training: Is this block in training mode?
        :param strides: The stride size for the first two convolutional layer
        :param dilation: The dilation rate for all convolutional layers
        :return: Application of an InceptionNet based approach to a normal residual building block branch on the input
        """
        tensor = in_tensor
        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training,
                                stride=(1, strides), dilation=(1, dilation), name='conv2d-1')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training,
                                stride=(1, strides), dilation=(1, dilation), name='conv2d-1')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training,
                                stride=(strides, 1), dilation=(dilation, 1), name='conv2d-2')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training,
                                stride=(strides, 1), dilation=(dilation, 1), name='conv2d-2')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=dilation, name='conv2d-3')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-3')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-3')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=dilation, name='conv2d-3')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=dilation, name='conv2d-4')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-4')
        if not self.FULL_PREACTIVATION:
            return tensor
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-4')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=dilation, name='conv2d-4')
        return tensor

    def _bottleneck_inception_branch(self, in_tensor, out_chan, is_training, strides=1, dilation=1):
        """
        Bottleneck building block branch of a residual network using full pre-activation and an InceptionNet approach to
        kernels.
        :param in_tensor: Tensor on which to apply this block on
        :param out_chan: Number of features to extract
        :param is_training: Is this block in training mode?
        :param strides: The stride size for the center two convolutional layer
        :param dilation: The dilation rate for the center convolutional layers
        :return: Application of an InceptionNet based approach to a bottleneck residual building block branch on the
                 input
        """
        tensor = in_tensor
        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=out_chan, is_training=is_training,
                                stride=strides, name='conv2d-1')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-1')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-1')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=out_chan, is_training=is_training,
                                stride=strides, name='conv2d-1')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=(1, dilation), name='conv2d-2')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-2')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-2')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(1, 3), out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=(1, dilation), name='conv2d-2')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=(dilation, 1), name='conv2d-3')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-3')
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-3')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=(3, 1), out_chan=out_chan, is_training=is_training, stride=1,
                                dilation=(dilation, 1), name='conv2d-3')

        if not self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=4 * out_chan, is_training=is_training, stride=1,
                                name='conv2d-4')
        tensor = self._batch_normalization(tensor=tensor, is_training=is_training, name='batch_norm-4')
        if not self.FULL_PREACTIVATION:
            return tensor
        tensor = self._leaky_relu(tensor=tensor, name='leaky_relu-4')
        if self.FULL_PREACTIVATION:
            tensor = self._conv(tensor=tensor, kernel_size=1, out_chan=4 * out_chan, is_training=is_training, stride=1,
                                name='conv2d-4')
        return tensor

    def _shortcut(self, in_tensor, out_chan, is_bottleneck, is_training, strides=1):
        """
        Wrapper for a shortcut building block using full pre-activation
        :param in_tensor: Tensor on which to apply this block on
        :param out_chan: Number of features to extract
        :param is_bottleneck: Is this used in a bottleneck block?
        :param is_training: Is this block in training mode?
        :param strides: The stride size for the convolutional layer
        :return:
        """
        if is_bottleneck:
            channels = 4 * out_chan
        else:
            channels = out_chan
        shortcut=in_tensor
        if not self.FULL_PREACTIVATION:
            shortcut = self._conv(tensor=shortcut, kernel_size=1, out_chan=channels, is_training=is_training,
                                  stride=strides)
        shortcut = self._batch_normalization(tensor=shortcut, is_training=is_training)
        if not self.FULL_PREACTIVATION:
            return shortcut
        shortcut = self._leaky_relu(tensor=shortcut)
        if self.FULL_PREACTIVATION:
            shortcut = self._conv(tensor=shortcut, kernel_size=1, out_chan=channels, is_training=is_training,
                                  stride=strides)
        return shortcut

    def vanilla(self, in_tensor, layer_name, out_chan, first_layer, is_training, depth=None):
        """
        Default building block of a residual network using full pre-activation.
        :param in_tensor: Tensor on which to apply this block on
        :param layer_name: Name of this block and related scope
        :param out_chan: Number of features to extract
        :param first_layer: Is this the first invocation of the block? (increases stride to downsample)
        :param is_training: Is this block in training mode?
        :param depth: Tuple of (current_depth, total_depth) used for stochastic depth survival rate calculations
        :return: Application of a default residual building block on the input
        """
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == self.MINIMAL_FEATURES:
                    strides = 1
                else:
                    strides = 2
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=False, is_training=is_training,
                                          strides=strides)
                residual = self._vanilla_branch(in_tensor, out_chan=out_chan, is_training=is_training, strides=strides)
            else:
                shortcut = in_tensor

                def branch():
                    return self._vanilla_branch(in_tensor, out_chan=out_chan, is_training=is_training)

                if depth is None:
                    residual = branch()
                else:
                    survival_rate = ResNetLayers._get_survival_rate(depth)
                    threshold = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float64)

                    def training():
                        dims = in_tensor.shape
                        return tf.cond(tf.greater_equal(survival_rate, threshold), lambda: branch(),
                                       lambda: tf.zeros([dims[0], out_chan, dims[2], dims[2]]))

                    def inference():
                        return tf.multiply(branch(), survival_rate)

                    residual = tf.cond(is_training, lambda: training(), lambda: inference())
            return tf.add(residual, shortcut)

    def bottleneck(self, in_tensor, layer_name, out_chan, first_layer, is_training, depth=None):
        """
        Bottleneck building block of a residual network using full pre-activation.
        :param in_tensor: Tensor on which to apply this block on
        :param layer_name: Name of this block and related scope
        :param out_chan: Number of features to extract
        :param first_layer: Is this the first invocation of the block? (increases stride to downsample)
        :param is_training: Is this block in training mode?
        :param depth: Tuple of (current_depth, total_depth) used for stochastic depth survival rate calculations
        :return: Application of a bottleneck residual building block on the input
        """
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == self.MINIMAL_FEATURES:
                    strides = 1
                else:
                    strides = 2
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=True, is_training=is_training,
                                          strides=strides)
                residual = self._bottleneck_branch(in_tensor, out_chan=out_chan, is_training=is_training,
                                                   strides=strides)
            else:
                shortcut = in_tensor

                def branch():
                    return self._bottleneck_branch(in_tensor, out_chan=out_chan, is_training=is_training)

                if depth is None:
                    residual = branch()
                else:
                    survival_rate = ResNetLayers._get_survival_rate(depth)
                    threshold = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float64)

                    def training():
                        dims = in_tensor.shape
                        return tf.cond(tf.greater_equal(survival_rate, threshold), lambda: branch(),
                                       lambda: tf.zeros([dims[0], out_chan, dims[2], dims[2]]))

                    def inference():
                        return tf.multiply(branch(), survival_rate)

                    residual = tf.cond(is_training, lambda: training(), lambda: inference())
            return tf.add(residual, shortcut)

    def inception(self, in_tensor, layer_name, out_chan, first_layer, is_training, depth=None):
        """
        Normal building block of a residual network using full pre-activation and an InceptionNet approach to kernels.
        :param in_tensor: Tensor on which to apply this block on
        :param layer_name: Name of this block and related scope
        :param out_chan: Number of features to extract
        :param first_layer: Is this the first invocation of the block? (increases stride to downsample)
        :param is_training: Is this block in training mode?
        :param depth: Tuple of (current_depth, total_depth) used for stochastic depth survival rate calculations
        :return: Application of an InceptionNet based approach to a normal residual building block on the input
        """
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == self.MINIMAL_FEATURES:
                    strides = 1
                else:
                    strides = 2
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=False, is_training=is_training,
                                          strides=strides)
                residual = self._inception_branch(in_tensor, out_chan=out_chan, is_training=is_training,
                                                  strides=strides)
            else:
                shortcut = in_tensor

                def branch():
                    return self._inception_branch(in_tensor, out_chan=out_chan, is_training=is_training)

                if depth is None:
                    residual = branch()
                else:
                    survival_rate = ResNetLayers._get_survival_rate(depth)
                    threshold = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float64)

                    def training():
                        dims = in_tensor.shape
                        return tf.cond(tf.greater_equal(survival_rate, threshold), lambda: branch(),
                                       lambda: tf.zeros([dims[0], out_chan, dims[2], dims[2]]))

                    def inference():
                        return tf.multiply(branch(), survival_rate)

                    residual = tf.cond(is_training, lambda: training(), lambda: inference())
            return tf.add(residual, shortcut)

    def bottleneck_inception(self, in_tensor, layer_name, out_chan, first_layer, is_training, depth=None):
        """
        Bottleneck building block of a residual network using full pre-activation and an InceptionNet approach to
        kernels.
        :param in_tensor: Tensor on which to apply this block on
        :param layer_name: Name of this block and related scope
        :param out_chan: Number of features to extract
        :param first_layer: Is this the first invocation of the block? (increases stride to downsample)
        :param is_training: Is this block in training mode?
        :param depth: Tuple of (current_depth, total_depth) used for stochastic depth survival rate calculations
        :return: Application of an InceptionNet based approach to a bottleneck residual building block on the input
        """
        with tf.variable_scope(layer_name):
            if first_layer:
                if out_chan == self.MINIMAL_FEATURES // 2:
                    strides = 1
                else:
                    strides = 2
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=True, is_training=is_training,
                                          strides=strides)
                residual = self._bottleneck_inception_branch(in_tensor, out_chan=out_chan, is_training=is_training,
                                                             strides=strides)
            else:
                shortcut = in_tensor

                def branch():
                    return self._bottleneck_inception_branch(in_tensor, out_chan=out_chan, is_training=is_training)

                if depth is None:
                    residual = branch()
                else:
                    survival_rate = ResNetLayers._get_survival_rate(depth)
                    threshold = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float64)

                    def training():
                        dims = in_tensor.shape
                        return tf.cond(tf.greater_equal(survival_rate, threshold), lambda: branch(),
                                       lambda: tf.zeros([dims[0], out_chan, dims[2], dims[2]]))

                    def inference():
                        return tf.multiply(branch(), survival_rate)

                    residual = tf.cond(is_training, lambda: training(), lambda: inference())
            return tf.add(residual, shortcut)

    def vanilla_dilation(self, in_tensor, layer_name, out_chan, first_layer, is_training, depth=None):
        """
        Modified default building block of a residual network using full pre-activation that doesn't downsample on the
        first invocation of this block and successive blocks using a dilated 3x3 kernel.
        :param in_tensor: Tensor on which to apply this block on
        :param layer_name: Name of this block and related scope
        :param out_chan: Number of features to extract
        :param first_layer: Is this the first invocation of the block? (increases stride to downsample)
        :param is_training: Is this block in training mode?
        :param depth: Tuple of (current_depth, total_depth) used for stochastic depth survival rate calculations
        :return: Application of the modified default residual building block on the input
        """
        with tf.variable_scope(layer_name):
            if first_layer:
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=False, is_training=is_training)
                residual = self._vanilla_branch(in_tensor, out_chan=out_chan, is_training=is_training, dilation=2)
            else:
                shortcut = in_tensor

                def branch():
                    return self._vanilla_branch(in_tensor, out_chan=out_chan, is_training=is_training, dilation=2)

                if depth is None:
                    residual = branch()
                else:
                    survival_rate = ResNetLayers._get_survival_rate(depth)
                    threshold = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float64)

                    def training():
                        dims = in_tensor.shape
                        return tf.cond(tf.greater_equal(survival_rate, threshold), lambda: branch(),
                                       lambda: tf.zeros([dims[0], out_chan, dims[2], dims[2]]))

                    def inference():
                        return tf.multiply(branch(), survival_rate)

                    residual = tf.cond(is_training, lambda: training(), lambda: inference())
            return tf.add(residual, shortcut)

    def bottleneck_dilation(self, in_tensor, layer_name, out_chan, first_layer, is_training, depth=None):
        """
        Modified bottleneck building block of a residual network using full pre-activation that doesn't downsample on
        the first invocation of this block and successive blocks using a dilated 3x3 kernel.
        :param in_tensor: Tensor on which to apply this block on
        :param layer_name: Name of this block and related scope
        :param out_chan: Number of features to extract
        :param first_layer: Is this the first invocation of the block? (increases stride to downsample)
        :param is_training: Is this block in training mode?
        :param depth: Tuple of (current_depth, total_depth) used for stochastic depth survival rate calculations
        :return: Application of the modified bottleneck residual building block on the input
        """
        with tf.variable_scope(layer_name):
            if first_layer:
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=True, is_training=is_training)
                residual = self._bottleneck_branch(in_tensor, out_chan=out_chan, is_training=is_training, dilation=2)
            else:
                shortcut = in_tensor

                def branch():
                    return self._bottleneck_branch(in_tensor, out_chan=out_chan, is_training=is_training, dilation=2)

                if depth is None:
                    residual = branch()
                else:
                    survival_rate = ResNetLayers._get_survival_rate(depth)
                    threshold = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float64)

                    def training():
                        dims = in_tensor.shape
                        return tf.cond(tf.greater_equal(survival_rate, threshold), lambda: branch(),
                                       lambda: tf.zeros([dims[0], out_chan, dims[2], dims[2]]))

                    def inference():
                        return tf.multiply(branch(), survival_rate)

                    residual = tf.cond(is_training, lambda: training(), lambda: inference())
            return tf.add(residual, shortcut)

    def inception_dilation(self, in_tensor, layer_name, out_chan, first_layer, is_training, depth=None):
        """
        Modified default building block of a residual network using full pre-activation and an InceptionNet approach
        to kernels that doesn't downsample on the first invocation of this block and successive blocks using a dilated
        3x3 kernel.
        :param in_tensor: Tensor on which to apply this block on
        :param layer_name: Name of this block and related scope
        :param out_chan: Number of features to extract
        :param first_layer: Is this the first invocation of the block? (increases stride to downsample)
        :param is_training: Is this block in training mode?
        :param depth: Tuple of (current_depth, total_depth) used for stochastic depth survival rate calculations
        :return: Application of the modified default residual building block based on InceptionNet on the input
        """
        with tf.variable_scope(layer_name):
            if first_layer:
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=False, is_training=is_training)
                residual = self._inception_branch(in_tensor, out_chan=out_chan, is_training=is_training, dilation=2)
            else:
                shortcut = in_tensor

                def branch():
                    return self._inception_branch(in_tensor, out_chan=out_chan, is_training=is_training, dilation=2)

                if depth is None:
                    residual = branch()
                else:
                    survival_rate = ResNetLayers._get_survival_rate(depth)
                    threshold = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float64)

                    def training():
                        dims = in_tensor.shape
                        return tf.cond(tf.greater_equal(survival_rate, threshold), lambda: branch(),
                                       lambda: tf.zeros([dims[0], out_chan, dims[2], dims[2]]))

                    def inference():
                        return tf.multiply(branch(), survival_rate)

                    residual = tf.cond(is_training, lambda: training(), lambda: inference())
            return tf.add(residual, shortcut)

    def bottleneck_inception_dilation(self, in_tensor, layer_name, out_chan, first_layer, is_training, depth=None):
        """
        Modified bottleneck building block of a residual network using full pre-activation and an InceptionNet approach
        to kernels that doesn't downsample on the first invocation of this block and successive blocks using a dilated
        3x3 kernel.
        :param in_tensor: Tensor on which to apply this block on
        :param layer_name: Name of this block and related scope
        :param out_chan: Number of features to extract
        :param first_layer: Is this the first invocation of the block? (increases stride to downsample)
        :param is_training: Is this block in training mode?
        :param depth: Tuple of (current_depth, total_depth) used for stochastic depth survival rate calculations
        :return: Application of the modified bottleneck residual building block based on InceptionNet on the input
        """
        with tf.variable_scope(layer_name):
            if first_layer:
                shortcut = self._shortcut(in_tensor, out_chan=out_chan, is_bottleneck=True, is_training=is_training)
                residual = self._bottleneck_inception_branch(in_tensor, out_chan=out_chan, is_training=is_training,
                                                             dilation=2)
            else:
                shortcut = in_tensor

                def branch():
                    return self._bottleneck_inception_branch(in_tensor, out_chan=out_chan, is_training=is_training,
                                                             dilation=2)
                if depth is None:
                    residual = branch()
                else:
                    survival_rate = ResNetLayers._get_survival_rate(depth)
                    threshold = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float64)

                    def training():
                        dims = in_tensor.shape
                        return tf.cond(tf.greater_equal(survival_rate, threshold), lambda: branch(),
                                       lambda: tf.zeros([dims[0], out_chan, dims[2], dims[2]]))

                    def inference():
                        return tf.multiply(branch(), survival_rate)

                    residual = tf.cond(is_training, lambda: training(), lambda: inference())
            return tf.add(residual, shortcut)


class ImageOps(object):
    @classmethod
    def make_gaussian(cls, size, sigma=3, centre=None, normalized=False):
        """
        Make a square gaussian kernel.
        :param size: Length of a side of the square
        :param sigma: Effective radius of the kernel
        :param centre: Center gaussian on these coordinates (if None then gaussian is centered at size/2)
        :param normalized: Normalize the gaussian
        :return Square gaussian kernel
        """
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        # fwhm is full-width-half-maximum, which can be thought of as an effective radius.
        # sigma = tf.mult(fwhm, tf.reciprocal(tf.sqrt(tf.mul(8, tf.log(2))))) # fwhm * 1/(sqrt(8*ln(2))) = sigma
        if normalized:
            norm_factor = 1 / 2 * math.pi * sigma
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
            # invert_heatmap_np -= cur_joint_heatmap  # Maybe we should include that but I don't see why; background?
        return gt_heatmap_np, 0
