"""Development architecture."""
from typing import Dict
import tensorflow as tf
import numpy as np
from core import BaseDataSource, BaseModel
from util.common_ops import ResNetLayers as rnl

# CONSTANTS
resnet_features = [64, 128, 256, 512]
resnet_features_experimental = [16, 32, 64, 128, 256, 512]
resnet_repetitions_small = [2, 2, 2, 2]
resnet_repetitions_normal = [3, 4, 6, 3]
resnet_repetitions_large = [3, 4, 23, 3]
resnet_repetitions_extra = [3, 8, 36, 3]
resnet_repetitions_experimental = [3, 4, 5, 6]


# HYPER PARAMETERS
CROPSIZE = 128
ACCURACY_DISTANCE = 2
FULL_PREACTIVATION = False
USE_4K = False
USE_UPCONVOLUTION = False
FCNN = False
RESNET_FEATURES = resnet_features
RESNET_REPETITIONS = resnet_repetitions_normal
RESNET_IDENTITY_LAYERS = np.sum(RESNET_REPETITIONS) - 4


class ResNet(BaseModel):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        rgb_image = input_tensors['img']
        keypoints = input_tensors['kp_2D']
        is_visible = input_tensors['vis_2D']
        resnet = rnl(self.summary, visualize=True,
                     minimal_features=RESNET_FEATURES[0], full_preactivation=FULL_PREACTIVATION)
        image = rgb_image

        with tf.variable_scope('resnet-experimental'):
            image = resnet.init_block(image, self.is_training)
            depth_count = 0
            for i, layers in enumerate(RESNET_REPETITIONS):
                for j in range(layers):
                    # if j != 0:
                    #     depth_count = depth_count + 1
                    # if i == 2 and j == 0:
                    #     image_snapshot = image
                    # if layers == RESNET_REPETITIONS[-1]:
                    #     image = resnet.bottleneck_dilation(image, layer_name='conv%d_%d' % (i + 2, j + 1),
                    #                                        first_layer=(j == 0), out_chan=RESNET_FEATURES[i],
                    #                                        is_training=self.is_training,
                    #                                        depth=(depth_count, RESNET_IDENTITY_LAYERS))
                    # else:
                    image = resnet.vanilla(image, layer_name='conv%d_%d' % (i + 2, j + 1),
                                           first_layer=(j == 0), out_chan=RESNET_FEATURES[i],
                                           is_training=self.is_training)
                                           #depth=(depth_count, RESNET_IDENTITY_LAYERS))

            # image = resnet._max_pool(image, pool=(4, 4))
            image = resnet.last_layer(image, is_training=self.is_training, use_4k=USE_4K, use_upconv=USE_UPCONVOLUTION)
            self.summary.histogram('last_layer', image)

        with tf.variable_scope('flatten'):
            result = resnet.prediction_layer(image, is_training=self.is_training, use_4k=USE_4K, fcnn=FCNN)

        with tf.variable_scope('loss_calculation'):
            # Include all keypoints for metrics. These are rougher scores.
            loss_mse = tf.reduce_mean(tf.squared_difference(keypoints, result))
            corr = tf.count_nonzero(tf.less_equal(tf.squared_difference(keypoints, result), ACCURACY_DISTANCE))
            precision = corr / (keypoints.shape[0] * keypoints.shape[1] * keypoints.shape[2])

            # Only include visible keypoints for metrics. These are nicer scores overall.
            count_vis = tf.count_nonzero(tf.multiply(keypoints, is_visible))
            loss_mse_vis = tf.multiply(tf.squared_difference(keypoints, result), is_visible)
            loss_mse_vis = tf.reduce_sum(tf.truediv(loss_mse_vis, tf.cast(count_vis, dtype=tf.float32)))
            corr_vis = tf.count_nonzero(tf.less_equal(tf.multiply(tf.squared_difference(keypoints, result), is_visible),
                                                      ACCURACY_DISTANCE))
            precision_visible = tf.divide(corr_vis, count_vis)

        loss_terms = {  # To optimize
            'kp_loss_mse': loss_mse,
            'kp_accuracy': precision,
            'kp_loss_mse_vis': loss_mse_vis,
            'kp_accuracy_vis': precision_visible
        }
        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': result}, loss_terms, {}
