"""Development architecture."""
from typing import Dict
import tensorflow as tf
import numpy as np
from core import BaseDataSource, BaseModel
from util.common_ops import ResNetLayers as rnl


# HYPER PARAMETERS
CROPSIZE = 128
ACCURACY_BOX = 3
USE_4K = False

resnet_channels = [64, 128, 256, 512]
resnet_repetitions_small = [2, 2, 2, 2]
resnet_repetitions_normal = [3, 4, 6, 3]
resnet_repetitions_large = [3, 4, 23, 3]
resnet_repetitions_extra = [3, 8, 36, 3]


class ResNet(BaseModel):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        rgb_image = input_tensors['img']
        keypoints = input_tensors['kp_2D']
        resnet = rnl(self.summary, True)

        with tf.variable_scope('resnet18'):
            image = resnet.init_block(rgb_image, self.is_training)
            for i, layers in enumerate(resnet_repetitions_small):
                for j in range(layers):
                    image = resnet.vanilla(image, layer_name='conv%d_%d' % (i + 2, j + 1),
                                              first_layer=(j == 0), out_chan=resnet_channels[i],
                                              is_training=self.is_training)
            image = resnet._max_pool(image, pool=4)
            image = resnet.last_layer(image, is_training=self.is_training, use_4k=False)
            self.summary.histogram('last_layer', image)
            self.summary.feature_maps('last_layer', image)

        with tf.variable_scope('flatten'):
            image = resnet.output_layer(image, is_training=self.is_training, use_4k=False)
            result = tf.reshape(image, (-1, 21, 2))

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
