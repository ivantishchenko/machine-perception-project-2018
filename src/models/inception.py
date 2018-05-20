"""MnistNet architecture."""
from typing import Dict

import tensorflow as tf

from core import BaseDataSource, BaseModel

class InceptionNet(BaseModel):

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def conv2d_bn(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
        """Utility function to apply conv + BN.
        Arguments:
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.
        Returns:
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        
        # Channels first
        bn_axis = 1
        # bn_axis = 3

        # port
        x = tf.layers.conv2d(x, 
            filters=filters, 
            kernel_size=(num_row, num_col), 
            strides=strides, 
            padding=padding,
            use_bias=False,
            name=conv_name,
            data_format='channels_first')
        
        x = tf.layers.batch_normalization(x,
            axis=bn_axis,
            scale=False,
            name=bn_name)

        x = tf.nn.relu(x, name=name)
        return x


    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['img']
        y = input_tensors['kp_2D']

        # OUTPUTS

        print("X shape = {}".format(x.shape))
        print("Y shape = {}".format(y.shape))

        # CONSTANTS

        channel_axis = 1 

        with tf.variable_scope('conv'):
            # INCEPTION
            x = self.conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding='valid')
            x = self.conv2d_bn(x, 32, 3, 3, padding='valid')
            x = self.conv2d_bn(x, 64, 3, 3)
            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_first')

            x = self.conv2d_bn(x, 80, 1, 1, padding='valid')
            x = self.conv2d_bn(x, 192, 3, 3, padding='valid')
            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_first')

            # mixed 0, 1, 2: 35 x 35 x 256
            branch1x1 = self.conv2d_bn(x, 64, 1, 1)

            branch5x5 = self.conv2d_bn(x, 48, 1, 1)
            branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

            branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

            branch_pool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')
            branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1)
            x = tf.concat(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed0')

            # mixed 1: 35 x 35 x 256
            branch1x1 = self.conv2d_bn(x, 64, 1, 1)

            branch5x5 = self.conv2d_bn(x, 48, 1, 1)
            branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

            branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

            branch_pool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')
            branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
            x = tf.concat(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed1')

            # mixed 2: 35 x 35 x 256
            branch1x1 = self.conv2d_bn(x, 64, 1, 1)

            branch5x5 = self.conv2d_bn(x, 48, 1, 1)
            branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

            branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

            branch_pool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')
            branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
            x = tf.concat(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed2')

            # mixed 3: 17 x 17 x 768
            branch3x3 = self.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

            branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

            branch_pool  = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_first')
            x = tf.concat([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

            # mixed 4: 17 x 17 x 768
            branch1x1 = self.conv2d_bn(x, 192, 1, 1)

            branch7x7 = self.conv2d_bn(x, 128, 1, 1)
            branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self.conv2d_bn(x, 128, 1, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = tf.concat(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed4')

            # mixed 5, 6: 17 x 17 x 768
            for i in range(2):
                branch1x1 = self.conv2d_bn(x, 192, 1, 1)

                branch7x7 = self.conv2d_bn(x, 160, 1, 1)
                branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7)
                branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

                branch7x7dbl = self.conv2d_bn(x, 160, 1, 1)
                branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
                branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 1, 7)
                branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
                branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)
                
                branch_pool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')
                branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
                x = tf.concat(
                    [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                    axis=channel_axis,
                    name='mixed' + str(5 + i))
            
            # mixed 7: 17 x 17 x 768
            branch1x1 = self.conv2d_bn(x, 192, 1, 1)

            branch7x7 = self.conv2d_bn(x, 192, 1, 1)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self.conv2d_bn(x, 192, 1, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = tf.concat(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed7')

            # mixed 8: 8 x 8 x 1280
            branch3x3 = self.conv2d_bn(x, 192, 1, 1)
            branch3x3 = self.conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

            branch7x7x3 = self.conv2d_bn(x, 192, 1, 1)
            branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7)
            branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1)
            branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
            
            branch_pool  = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_first')
            x =  tf.concat([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

            # mixed 9: 8 x 8 x 2048
            for i in range(2):
                branch1x1 = self.conv2d_bn(x, 320, 1, 1)

                branch3x3 = self.conv2d_bn(x, 384, 1, 1)
                branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3)
                branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1)
                branch3x3 = tf.concat([branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

                branch3x3dbl = self.conv2d_bn(x, 448, 1, 1)
                branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3)
                branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3)
                branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1)
                branch3x3dbl = tf.concat([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

                branch_pool = tf.layers.average_pooling2d(x, pool_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')
                branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
                x = tf.concat(
                    [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                    axis=channel_axis,
                    name='mixed' + str(9 + i))

            # Perform global AVG Pooling
            x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=(1, 1), padding='valid', data_format='channels_first')
        

        # FC
        with tf.variable_scope('fc'):
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=42, name='out')
            x = tf.reshape(x, (-1, 21, 2))


        # Define outputs
        loss_terms = {  # To optimize
            'kp_2D_mse': tf.reduce_mean(tf.squared_difference(x, y)),
        }
        return {'kp_2D': x}, loss_terms, {}
