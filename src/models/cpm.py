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
ACCURACY_DISTANCE = 2


class Glover(BaseModel):
    """
    Network performing 3D pose estimation of a human hand from a single color image.
    Inspired by:
    https://github.com/shihenw/convolutional-pose-machines-release/tree/master/model/_trained_MPI
        no downscaling layer from this one

    Additional interesting resources:
    https://github.com/ildoonet/tf-pose-estimation/blob/master/src/network_cmu.py
    """
    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        print(data_source.output_tensors)
        rgb_image = input_tensors['img']
        keypoints = input_tensors['kp_2D']
        is_visible = input_tensors['vis_2D']

        def generate_heatmaps(kp, length, dev = 1.0):
            heatmap, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, length, dev, CROP_SIZE // length, True),
                                   tf.nn.embedding_lookup(kp, np.array(range(kp.shape[0]))),
                                   dtype=(
                                   [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                   tf.float32),
                                   back_prop=False)
            return tf.transpose(tf.stack(heatmap), perm=[1, 0, 2, 3])

        # hm_gt = generate_heatmaps(keypoints, HEATMAP_SIZE)
        # hm_gt_fs = generate_heatmaps(keypoints, CROP_SIZE)

        layers = bl(self.summary, True)

        with tf.variable_scope('posenet'):
            layers_per_block = [2, 2, 4, 2]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]
            scoremap_list = []

            image = rgb_image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    image = layers.conv_relu(image, 'conv%d_%d' % (block_id, layer_id + 1), kernel_size=3,
                                             out_chan=chan_num, is_training=self.is_training,
                                             max_pool=(True if layer_id == range(layer_num)[-1] and pool else False))

            image = layers.conv_relu(image, 'conv4_3_CPM', kernel_size=3, out_chan=256, is_training=self.is_training)
            image = layers.conv_relu(image, 'conv4_4_CPM', kernel_size=3, out_chan=256, is_training=self.is_training)
            image = layers.conv_relu(image, 'conv4_5_CPM', kernel_size=3, out_chan=256, is_training=self.is_training)
            image = layers.conv_relu(image, 'conv4_6_CPM', kernel_size=3, out_chan=256, is_training=self.is_training)
            downsampled_map = layers.conv_relu(image, 'conv4_7_CPM', kernel_size=3, out_chan=128,
                                               is_training=self.is_training)

            image = layers.conv_relu(downsampled_map, 'conv5_1_CPM', kernel_size=1, out_chan=512,
                                     is_training=self.is_training)
            scoremap = layers._conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, is_training=self.is_training,
                                    name='conv5_P')
            scoremap_list.append(scoremap)

            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 5
            num_recurrent_units = 5
            offset = 6
            for pass_id in range(num_recurrent_units):
                image = tf.concat([scoremap_list[-1], downsampled_map], 1)
                for rec_id in range(layers_per_recurrent_unit):
                    image = layers.conv_relu(image, 'conv%d_%d_CPM' % (pass_id + offset, rec_id + 1), kernel_size=5,
                                             out_chan=128, is_training=self.is_training)
                image = layers.conv_relu(image, 'conv%d_%d_CPM' % (pass_id + offset, layers_per_recurrent_unit + 1),
                                         kernel_size=1, out_chan=128, is_training=self.is_training)
                scoremap = layers._conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, is_training=self.is_training,
                                        name='conv%d_P' % (pass_id + offset))
                scoremap_list.append(scoremap)

        with tf.variable_scope('flatten'):
            result = layers.pred_layer(scoremap_list[-1], is_training=self.is_training)

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

        # with tf.variable_scope('upscale_pred'):
        #     pred_upscale = tf.transpose(scoremap_list[-1], [0, 2, 3, 1])
        #     pred_upscale = tf.image.resize_images(pred_upscale, (CROP_SIZE, CROP_SIZE), method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
        #     pred_upscale = tf.transpose(pred_upscale, [0, 3, 1, 2])
        #
        # with tf.variable_scope('point_pred'):
        #     input = pred_upscale
        #
        #     def image_max(t):
        #         t = tf.reshape(t, [-1])
        #         idx = tf.argmax(t)
        #         x = tf.cast(idx % CROP_SIZE, dtype=tf.float32)
        #         y = tf.cast(idx // CROP_SIZE, dtype=tf.float32)
        #         return x, y
        #
        #     def outer(t):
        #         res = tf.map_fn(lambda j: image_max(j), tf.nn.embedding_lookup(t, np.array(range(input.shape[1]))),
        #                         dtype=(tf.float32, tf.float32), back_prop=False)
        #         return tf.stack([res[0], res[1]], axis=1)
        #
        #     pred_points = tf.map_fn(lambda i: outer(i), tf.nn.embedding_lookup(input, np.array(range(input.shape[0]))),
        #                             dtype=tf.float32, back_prop=False)
        #
        # with tf.variable_scope('loss_calculation'):
        #     loss_filter = (HEATMAP_SIZE ** 2) * tf.losses.mean_squared_error(hm_gt, scoremap_list[-1])
        #     loss_scaled = tf.losses.mean_squared_error(keypoints, pred_points)
        #
        # loss_terms = {  # To optimize
        #     'kp_mse_filter': loss_filter,
        #     'kp_mse_scaled': loss_scaled
        # }

        loss_terms_flattened = {  # To optimize
            'kp_loss_mse': loss_mse,
            'kp_accuracy': precision,
            'kp_loss_mse_vis': loss_mse_vis,
            'kp_accuracy_vis': precision_visible
        }

        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': result}, loss_terms_flattened, {}