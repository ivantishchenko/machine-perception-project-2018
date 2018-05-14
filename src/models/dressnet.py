"""Development architecture."""
from typing import Dict
import tensorflow as tf
import numpy as np
from core import BaseDataSource, BaseModel
from util.common_ops import ResNetOps as rop
from util.common_ops import NetworkOps as nop


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
            image = rop.init_block(rgb_image, trainable=TRAIN)
            for i, layers in enumerate(resnet_repetitions_18):
                for j in range(layers):
                    image = rop.vanilla(image, layer_name='conv%d_%d' % (i + 2, j + 1), first_layer=(j == 0),
                                        out_chan=resnet_channels[i], trainable=TRAIN)
            image = rop.last_layer(image)

        with tf.variable_scope('flatten'):
            result = rop.output_layer(image)
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