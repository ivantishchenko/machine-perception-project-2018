from typing import Dict
import tensorflow as tf
from core import BaseDataSource, BaseModel

ACCURACY_DISTANCE = 2

class MvitNet(BaseModel):
    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        filters1, filters2, filters3 = filters

        x = tf.layers.conv2d(input_tensor, filters1, (1, 1), name=conv_name_base + '2a', data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=bn_axis, name=bn_name_base + '2a')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filters2, kernel_size, padding='same', name=conv_name_base + '2b', data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=bn_axis, name=bn_name_base + '2b')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filters3, (1, 1), name=conv_name_base + '2c', data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=bn_axis, name=bn_name_base + '2c')

        x = tf.add(x, input_tensor)
        x = tf.nn.relu(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv_block is the block that has a conv layer at shortcut

        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.

        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        filters1, filters2, filters3 = filters
        bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides=strides, name=conv_name_base + '2a', data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=bn_axis, name=bn_name_base + '2a')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filters2, kernel_size, padding='same', name=conv_name_base + '2b', data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=bn_axis, name=bn_name_base + '2b')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filters3, (1, 1), name=conv_name_base + '2c', data_format='channels_first')
        x = tf.layers.batch_normalization(x, axis=bn_axis, name=bn_name_base + '2c')

        shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), strides=strides, name=conv_name_base + '1', data_format='channels_first')
        shortcut = tf.layers.batch_normalization(shortcut, axis=bn_axis, name=bn_name_base + '1')

        x = tf.add(x, shortcut)
        x = tf.nn.relu(x)
        return x

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['img']
        y = input_tensors['kp_2D']
        bn_axis = 1

        with tf.variable_scope('resnet50'):
            # Optional padding
            # expand_size = 200
            # x = tf.reshape(x, [-1, 128, 128, 3])
            # x = tf.image.resize_image_with_crop_or_pad(x, expand_size, expand_size)
            # x = tf.reshape(x, [-1, 3, expand_size, expand_size])
            # architechture
            x = tf.layers.conv2d(x, 64, (7, 7), strides=(2, 2), name='conv1', data_format='channels_first')
            x = tf.layers.batch_normalization(x, axis=bn_axis, name='bn_conv1')
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), data_format='channels_first')

            x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
            x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
            x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

            x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
            x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
            x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
            x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

            x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
            x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

            x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
            x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
            x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

            x = tf.layers.average_pooling2d(x, (4, 4), strides=1, name='avg_pool', data_format='channels_first')
        with tf.variable_scope('fc'):
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=42, name='out')
            x = tf.reshape(x, (-1, 21, 2))

        with tf.variable_scope('loss_calculation'):
            loss_mse = tf.reduce_mean(tf.squared_difference(x, y))
            corr = tf.count_nonzero(tf.less_equal(tf.squared_difference(x, y), ACCURACY_DISTANCE))
            precision = corr / (y.shape[0] * y.shape[1] * y.shape[2])


        # Define outputs
        loss_terms = {  # To optimize
            'kp_loss_mse': loss_mse,
            'kp_accuracy': precision,
        }
        return {'kp_2D': x}, loss_terms, {}
