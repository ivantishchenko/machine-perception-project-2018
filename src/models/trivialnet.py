from typing import Dict
import tensorflow as tf
from core import BaseDataSource, BaseModel
from util.common_ops import BasicLayers as bl

ACCURACY_BOX = 3

class TrivialNet(BaseModel):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        rgb_image = input_tensors['img']
        keypoints = input_tensors['kp_2D']
        layers = bl(self.summary)

        with tf.variable_scope('keypoints'):
            image = rgb_image
            image = layers.conv_layer(image, "layer1", kernel_size=3, out_chan=64, is_training=self.is_training,
                                      disable_dropout=False)
            image = layers.conv_layer(image, "layer2", kernel_size=3, out_chan=128, is_training=self.is_training,
                                      disable_dropout=False)
            image = layers.conv_layer(image, "layer3", kernel_size=3, out_chan=128, is_training=self.is_training,
                                      disable_dropout=False)
            image = layers.conv_layer(image, "layer4", kernel_size=3, out_chan=128, is_training=self.is_training,
                                      max_pool=True, disable_dropout=False)
            image = layers.conv_layer(image, "layer5", kernel_size=3, out_chan=128, is_training=self.is_training,
                                      disable_dropout=False)
            image = layers.conv_layer(image, "layer6", kernel_size=3, out_chan=128, is_training=self.is_training,
                                      disable_dropout=False)
            image = layers.conv_layer(image, "layer7", kernel_size=3, out_chan=256, is_training=self.is_training,
                                      max_pool=True, disable_dropout=False)
            image = layers.conv_layer(image, "layer8", kernel_size=3, out_chan=256, is_training=self.is_training,
                                      disable_dropout=False)
            image = layers.conv_layer(image, "layer9", kernel_size=3, out_chan=256, is_training=self.is_training,
                                      max_pool=True, disable_dropout=False)
            image = layers.conv_layer(image, "layer10", kernel_size=3, out_chan=512, is_training=self.is_training,
                                      disable_dropout=False)
            image = layers.conv_layer(image, "layer11", kernel_size=1, out_chan=2048, is_training=self.is_training,
                                      disable_dropout=False)
            image = layers.conv_layer(image, "layer12", kernel_size=1, out_chan=256, is_training=self.is_training,
                                      max_pool=True, disable_dropout=False)
            image = layers.conv_layer(image, "layer13", kernel_size=3, out_chan=256, is_training=self.is_training,
                                      max_pool=True, disable_dropout=False)

        with tf.variable_scope('flatten'):
            result = tf.contrib.layers.flatten(image)
            result = tf.layers.dense(result, units=42, name='dense')
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