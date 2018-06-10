from typing import Dict
import tensorflow as tf
from core import BaseDataSource, BaseModel

ACCURACY_DISTANCE = 2


class MvitNetV2(BaseModel):
    stddev = 5e-2
    wd = 5e-4
    use_fp16 = False
    moving_average_decay = 0.999

    data_format = 'NCHW'
    bn_axis = 1
    batch_size = 8
    points_num = 21

    def _variable_on_gpu(self, name, shape, initializer, trainable):
        # with tf.device('/gpu:0'):
        #     dtype = tf.float16 if self.use_fp16 else tf.float32
        #     # trainable: If `True` also add the variable to the graph collection
        #     var = tf.get_variable(name, shape, initializer=initializer,
        #             dtype=dtype, trainable=trainable)
        dtype = tf.float16 if self.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd, trainable):
        dtype = tf.float16 if self.use_fp16 else tf.float32
        var = self._variable_on_gpu(name, shape,
                                    tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
                                    trainable)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                name='weights_loss')
            tf.add_to_collection("losses", weight_decay)
        return var

    def conv_layer(self, bottom, kernel_size, out_channel, name, is_BN, trainable):
        with tf.variable_scope(name) as scope:
            kernel = self._variable_with_weight_decay(
                    "weights",
                    shape = [kernel_size, kernel_size, bottom.get_shape()[1],
                      out_channel],
                    stddev = self.stddev,
                    wd = self.wd,
                    trainable=trainable)
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding="SAME", data_format=self.data_format)

            biases = self._variable_on_gpu('biases', [out_channel], tf.constant_initializer(0.0), trainable)

            pre_activation = tf.nn.bias_add(conv, biases, data_format=self.data_format)
            if is_BN:
                bn_activation = tf.layers.batch_normalization(pre_activation, axis=self.bn_axis)
                top = tf.nn.relu(bn_activation, name=scope.name)
            else:
                top = tf.nn.relu(pre_activation, name=scope.name)
        return top

    def fc_layer(self, bottom, out_num, name, is_BN, trainable):
        flatten_bottom = tf.reshape(bottom, [self.batch_size, -1])
        with tf.variable_scope(name) as scope:
            weights = self._variable_with_weight_decay(
                    "weights",
                    shape = [flatten_bottom.get_shape()[-1], out_num],
                    stddev = self.stddev,
                    wd = self.wd,
                    trainable=trainable)
            mul = tf.matmul(flatten_bottom, weights)
            biases = self._variable_on_gpu('biases', [out_num],
                                           tf.constant_initializer(0.0), trainable)
            pre_activation = tf.nn.bias_add(mul, biases)
            if is_BN:
                bn_activation = tf.layers.batch_normalization(pre_activation)
                top = tf.nn.relu(bn_activation, name=scope.name)
            else:
                top = tf.nn.relu(pre_activation, name=scope.name)
        return top

    def final_fc_layer(self, bottom, out_num, name, trainable):
        flatten_bottom = tf.reshape(bottom, [self.batch_size, -1])
        with tf.variable_scope(name) as scope:
            weights = self._variable_with_weight_decay(
                    "weights",
                    shape = [flatten_bottom.get_shape()[-1], out_num],
                    stddev = self.stddev,
                    wd = self.wd,
                    trainable=trainable)
            mul = tf.matmul(flatten_bottom, weights) # Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
            biases = self._variable_on_gpu('biases', [out_num],
                                           tf.constant_initializer(0.0), trainable)
            top = tf.nn.bias_add(mul, biases) # Returns: A `Tensor` with the same type as `value`.
        return top

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['img']
        y = input_tensors['kp_2D']
        is_BN = False
        trainable = True

        with tf.variable_scope('mvitNet'):
            conv1_1 = self.conv_layer(x, 3, 64, 'conv1_1', is_BN, trainable)
            conv1_2 = self.conv_layer(conv1_1, 3, 64, 'conv1_2', is_BN, trainable)
            pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME", name="pool1", data_format=self.data_format)

            conv2_1 = self.conv_layer(pool1, 3, 128, 'conv2_1', is_BN, trainable)
            conv2_2 = self.conv_layer(conv2_1, 3, 128, 'conv2_2', is_BN, trainable)
            pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME", name="pool2", data_format=self.data_format)

            conv3_1 = self.conv_layer(pool2, 3, 256, 'conv3_1', is_BN, trainable)
            conv3_2 = self.conv_layer(conv3_1, 3, 256, 'conv3_2', is_BN, trainable)
            conv3_3 = self.conv_layer(conv3_2, 3, 256, 'conv3_3', is_BN, trainable)
            pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME", name="pool3", data_format=self.data_format)

            conv4_1 = self.conv_layer(pool3, 3, 512, 'conv4_1', is_BN, trainable)
            conv4_2 = self.conv_layer(conv4_1, 3, 512, 'conv4_2', is_BN, trainable)
            conv4_3 = self.conv_layer(conv4_2, 3, 512, 'conv4_3', is_BN, trainable)
            pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME", name="pool4", data_format=self.data_format)

            conv5_1 = self.conv_layer(pool4, 3, 512, 'conv5_1', is_BN, trainable)
            conv5_2 = self.conv_layer(conv5_1, 3, 512, 'conv5_2', is_BN, trainable)
            conv5_3 = self.conv_layer(conv5_2, 3, 512, 'conv5_3', is_BN, trainable)
            pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME", name="pool5", data_format=self.data_format)

            fc6 = self.fc_layer(pool5, 4096, 'fc6', is_BN, trainable)
            fc6 = tf.nn.dropout(fc6, 0.5)

            fc7 = self.fc_layer(fc6, 4096, 'fc7', is_BN, trainable)
            fc7 = tf.nn.dropout(fc7, 0.5)
            fc8 = self.final_fc_layer(fc7, self.points_num * 2, 'fc8', trainable)

            predictions = tf.reshape(fc8, (-1, 21, 2))

        with tf.variable_scope('loss_calculation'):
            loss_mse = tf.reduce_mean(tf.squared_difference(predictions, y))
            corr = tf.count_nonzero(tf.less_equal(tf.squared_difference(predictions, y), ACCURACY_DISTANCE))
            precision = corr / (y.shape[0] * y.shape[1] * y.shape[2])


        # Define outputs
        loss_terms = {  # To optimize
            'kp_loss_mse': loss_mse,
            'kp_accuracy': precision,
        }
        return {'kp_2D': x}, loss_terms, {}
