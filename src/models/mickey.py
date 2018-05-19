"""Development architecture."""
from typing import Dict
import tensorflow as tf
import numpy as np
from core import BaseDataSource, BaseModel
from util.common_ops import BasicLayers as bl
from util.common_ops import ImageOps as iop

# HYPER PARAMETERS
CROP_SIZE = 128
KEYPOINT_COUNT = 21
HEATMAP_SIZE = 16


class Glover(BaseModel):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        print(data_source.output_tensors)
        rgb_image = input_tensors['img']
        print('rgb_image_dims={}'.format(rgb_image.get_shape()))
        keypoints = input_tensors['kp_2D']
        print('keypoint_dims={}'.format(keypoints.get_shape()))
        hm_gt, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, HEATMAP_SIZE, 1.0, CROP_SIZE // HEATMAP_SIZE, True),
                             tf.nn.embedding_lookup(keypoints, np.array(range(keypoints.shape[0]))),
                             dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                    ], tf.float32),
                             back_prop=False)
        hm_gt = tf.stack(hm_gt)
        hm_gt = tf.transpose(hm_gt, perm=[1, 0, 2, 3])
        print('heatmap_dims={}'.format(hm_gt))
        # hm_gt_fs, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, CROP_SIZE, 1.0, CROP_SIZE // CROP_SIZE),
        #                         tf.nn.embedding_lookup(keypoints, np.array(range(keypoints.shape[0]))),
        #                         dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
        #                                 tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
        #                                 tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
        #                                 ], tf.float32),
        #                         back_prop=False)
        # hm_gt_fs = tf.stack(hm_gt_fs)
        # hm_gt_fs = tf.transpose(hm_gt_fs, perm=[1, 0, 2, 3])
        layers = bl(self.summary)

        with tf.variable_scope('keypoints'):
            layers_per_block = [2, 2, 4, 5]
            out_chan_list = [32, 64, 128, 256]
            pool_list = [True, True, True, False]
            scoremap_list = []

            image = rgb_image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    image = layers.conv_relu(image, 'conv%d_%d' % (block_id, layer_id + 1), kernel_size=3,
                                             out_chan=chan_num, is_training=self.is_training,
                                             maxpool=(True if layer_id == range(layer_num)[-1] and pool else False))

            downsampled_map = layers.conv_relu(image, 'conv4_D', kernel_size=3, out_chan=64, is_training=self.is_training)
            image = layers.conv_relu(downsampled_map, 'conv5_1', kernel_size=1, out_chan=256, is_training=self.is_training)
            scoremap = layers._conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, is_training=self.is_training, name='conv5_P')
            scoremap_list.append(scoremap)

            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 4
            num_recurrent_units = 2
            offset = 6
            for pass_id in range(num_recurrent_units):
                image = tf.concat([scoremap_list[-1], downsampled_map], 1)
                for rec_id in range(layers_per_recurrent_unit):
                    image = layers.conv_relu(image, 'conv%d_%d' % (pass_id + offset, rec_id + 1), kernel_size=5, out_chan=64, is_training=self.is_training)
                image = layers.conv_relu(image, 'conv%d_%d' % (pass_id + offset, layers_per_recurrent_unit + 1), kernel_size=1, out_chan=64, is_training=self.is_training)
                scoremap = layers._conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, is_training=self.is_training, name='conv%d_P' % (pass_id + offset))
                scoremap_list.append(scoremap)

        with tf.variable_scope('upscale_pred'):
            pred_upscale = tf.transpose(scoremap_list[-1], [0, 2, 3, 1])
            pred_upscale = tf.image.resize_images(pred_upscale, (CROP_SIZE, CROP_SIZE), method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
            pred_upscale = tf.transpose(pred_upscale, [0, 3, 1, 2])

        with tf.variable_scope('point_pred'):
            input = pred_upscale

            def image_max(t):
                t = tf.reshape(t, [-1])
                idx = tf.argmax(t)
                x = tf.cast(idx % CROP_SIZE, dtype=tf.float32)
                y = tf.cast(idx // CROP_SIZE, dtype=tf.float32)
                return x, y

            def outer(t):
                res = tf.map_fn(lambda j: image_max(j), tf.nn.embedding_lookup(t, np.array(range(input.shape[1]))),
                                dtype=(tf.float32, tf.float32), back_prop=False)
                return tf.stack([res[0], res[1]], axis=1)

            pred_points = tf.map_fn(lambda i: outer(i), tf.nn.embedding_lookup(input, np.array(range(input.shape[0]))),
                                    dtype=tf.float32, back_prop=False)

        with tf.variable_scope('loss_calculation'):
            loss_filter = (HEATMAP_SIZE ** 2) * tf.losses.mean_squared_error(hm_gt, scoremap_list[-1])
            loss_scaled = tf.losses.mean_squared_error(keypoints, pred_points)

        loss_terms = {  # To optimize
            'kp_mse_filter': loss_filter,
            'kp_mse_scaled': loss_scaled
        }
        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': pred_points}, loss_terms, {}