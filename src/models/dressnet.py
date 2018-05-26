"""Development architecture."""
from typing import Dict
import tensorflow as tf
from core import BaseDataSource, BaseModel
from util.common_ops import ResNetLayers as rnl


# HYPER PARAMETERS
CROPSIZE = 128
ACCURACY_DISTANCE = 2
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
        is_visible = input_tensors['vis_2D']
        resnet = rnl(self.summary, True)

        with tf.variable_scope('resnet34'):
            image = resnet.init_block(rgb_image, self.is_training)
            for i, layers in enumerate(resnet_repetitions_normal):
                for j in range(layers):
                    image = resnet.vanilla(image, layer_name='conv%d_%d' % (i + 2, j + 1),
                                           first_layer=(j == 0), out_chan=resnet_channels[i],
                                           is_training=self.is_training)
            # image = resnet._max_pool(image, pool=4)
            image = resnet.last_layer(image, is_training=self.is_training, use_4k=False)
            self.summary.histogram('last_layer', image)

        with tf.variable_scope('flatten'):
            result = resnet.prediction_layer(image, is_training=self.is_training, use_4k=False)

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
