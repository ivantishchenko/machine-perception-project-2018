"""Development architecture."""
from typing import Dict
import tensorflow as tf
import numpy as np
from core import BaseDataSource, BaseModel
from util.common_ops import NetworkOps as ops

# HYPER PARAMETERS
CROPSIZE = 128
KEYPOINT_COUNT = 21
ACCURACY_BOX = 3
TRAIN = True

resnet_channels = [64, 128, 256, 512]
resnet_repetitions_18 = [2, 2, 2, 2]
resnet_repetitions_34 = [3, 4, 6, 3]
resnet_repetitions_50 = [3, 4, 6, 3]


class ResNet(BaseModel):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        # resnet_18 = tf.Graph()
        # with resnet_18.as_default():
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        rgb_image = input_tensors['img']
        keypoints = input_tensors['kp_2D']

        with tf.variable_scope('resnet18-vanilla'):
            image = rgb_image
            image = ops.resnet_init_block(image, trainable=TRAIN)
            for i, layers in enumerate(resnet_repetitions_18):
                for j in range(layers):
                    if j == 0:
                        image = ops.resnet_vanilla_first(image, layer_name='conv%d_%d' % (i + 2, j + 1),
                                                         out_chan=resnet_channels[i], trainable=TRAIN)
                    else:
                        image = ops.resnet_vanilla(image, layer_name='conv%d_%d' % (i + 2, j + 1),
                                                   out_chan=resnet_channels[i], trainable=TRAIN)
            image = ops.max_pool(image, pool=4, name='max_pool')
            image = tf.layers.average_pooling2d(image, pool_size=4, strides=1, data_format='channels_first',
                                                padding='same', name='average_pool')

        with tf.variable_scope('flatten'):
            result = image
            result = tf.contrib.layers.flatten(result)
            result = ops.fc_relu(result, 'fc_relu', out_chan=42, disable_dropout=True, trainable=True)
            result = tf.reshape(result, (-1, 21, 2))

        with tf.variable_scope('loss_calculation'):
            loss_mse = tf.losses.mean_squared_error(keypoints, result)
            corr_pred = tf.count_nonzero(tf.less_equal(tf.abs(tf.subtract(keypoints, result)), ACCURACY_BOX))
            precision = corr_pred / (keypoints.shape[0] * keypoints.shape[1] * keypoints.shape[2])

        loss_terms = {  # To optimize
            'kp_loss_mse': loss_mse,
            'kp_accuracy': precision,
        }
        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': result}, loss_terms, {}


class DenseNet(BaseModel):
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        return