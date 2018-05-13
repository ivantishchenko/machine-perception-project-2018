from typing import Dict
import tensorflow as tf
from core import BaseDataSource, BaseModel
from util.common_ops import NetworkOps as nop

TRAIN = True

class TrivialNet(BaseModel):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        rgb_image = input_tensors['img']
        keypoints = input_tensors['kp_2D']

        with tf.variable_scope('keypoints'):
            image = rgb_image
            image = nop.conv_relu(image, "layer1", kernel_size=3, out_chan=64, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer2", kernel_size=3, out_chan=128, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer3", kernel_size=3, out_chan=128, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer4", kernel_size=3, out_chan=128, maxpool=True, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer5", kernel_size=3, out_chan=128, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer6", kernel_size=3, out_chan=128, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer7", kernel_size=3, out_chan=256, maxpool=True, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer8", kernel_size=3, out_chan=256, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer9", kernel_size=3, out_chan=256, maxpool=True, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer10", kernel_size=3, out_chan=512, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer11", kernel_size=1, out_chan=2048, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer12", kernel_size=1, out_chan=256, maxpool=True, disable_dropout=False, enable_scale=True, trainable=TRAIN)
            image = nop.conv_relu(image, "layer13", kernel_size=3, out_chan=256, maxpool=True, disable_dropout=False, enable_scale=True, trainable=TRAIN)

        with tf.variable_scope('flatten'):
            result = tf.contrib.layers.flatten(image)
            result = tf.layers.dense(result, units=42, name='dense')
            result = tf.reshape(result, (-1, 21, 2))

        with tf.variable_scope('loss_calculation'):
            loss_mse = tf.losses.mean_squared_error(keypoints, result)

        loss_terms = {  # To optimize
            'kp_loss_mse': loss_mse,
        }
        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': result}, loss_terms, {}