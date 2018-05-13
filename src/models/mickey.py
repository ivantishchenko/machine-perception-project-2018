"""Development architecture."""
from typing import Dict
import tensorflow as tf
import numpy as np
from core import BaseDataSource, BaseModel
from util.common_ops import NetworkOps as nop
from util.common_ops import ImageOps as iop

# HYPER PARAMETERS
CROPSIZE = 128
KEYPOINT_COUNT = 21
HEATMAP_SIZE = 16
TRAIN = True


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
        heatmap_keypoints, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, HEATMAP_SIZE, 1.0, CROPSIZE // HEATMAP_SIZE), tf.nn.embedding_lookup(keypoints, np.array(range(keypoints.shape[0]))),
                                      dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                                      back_prop=False)
        heatmap_keypoints = tf.stack(heatmap_keypoints)
        heatmap_keypoints = tf.transpose(heatmap_keypoints, perm=[1, 0, 2, 3])
        print('heatmap_dims={}'.format(heatmap_keypoints))

        with tf.variable_scope('keypoints'):
            layers_per_block = [2, 2, 4, 5]
            out_chan_list = [32, 64, 128, 256]
            pool_list = [True, True, True, False]
            scoremap_list = []

            image = rgb_image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    image = nop.conv_relu(image, 'conv%d_%d' % (block_id, layer_id + 1), kernel_size=3, out_chan=chan_num, maxpool=(True if layer_id == range(layer_num)[-1] and pool else False), trainable=TRAIN)
                    print('conv{}_{}: output_dims={}'.format(block_id, layer_id+1, image.get_shape()))

            downsampled_map = nop.conv_relu(image, 'conv4_D', kernel_size=3, out_chan=64, trainable=TRAIN)
            print('conv4_D: output_dims={}'.format(downsampled_map.get_shape()))
            image = nop.conv_relu(downsampled_map, 'conv5_1', kernel_size=1, out_chan=256, trainable=TRAIN)
            print('conv5_1: output_dims={}'.format(image.get_shape()))
            scoremap = nop.conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, name='conv5_P', trainable=TRAIN)
            print('conv5_P: output_dims={}'.format(scoremap.get_shape()))
            scoremap_list.append(scoremap)

            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 4
            num_recurrent_units = 2
            offset = 6
            for pass_id in range(num_recurrent_units):
                image = tf.concat([scoremap_list[-1], downsampled_map], 1)
                for rec_id in range(layers_per_recurrent_unit):
                    print('conv{}_{}: input_dims={}'.format(pass_id + offset, rec_id+1, image.get_shape()))
                    image = nop.conv_relu(image, 'conv%d_%d' % (pass_id + offset, rec_id + 1), kernel_size=5, out_chan=64, trainable=TRAIN)
                    print('conv{}_{}: output_dims={}'.format(pass_id + offset, rec_id+1, image.get_shape()))
                # print('conv{}_6: input_dims={}'.format(pass_id + offset, image.get_shape()))
                image = nop.conv_relu(image, 'conv%d_%d' % (pass_id + offset, layers_per_recurrent_unit + 1), kernel_size=1, out_chan=64, trainable=TRAIN)
                print('conv{}_{}: output_dims={}'.format(pass_id + offset, layers_per_recurrent_unit+1, image.get_shape()))
                # print('conv{}_7: input_dims={}'.format(pass_id + offset, image.get_shape()))
                scoremap = nop.conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, name='conv%d_P' % (pass_id + offset), trainable=TRAIN)
                print('conv{}_P: output_dims={}'.format(pass_id + offset, scoremap.get_shape()))
                scoremap_list.append(scoremap)
            print('final: {}'.format(scoremap_list))

        # TODO: Fix this; returns values too small imho
        with tf.variable_scope('loss_calculation'):
            print(scoremap_list[-1])
            print(heatmap_keypoints)
            loss = tf.losses.mean_squared_error(heatmap_keypoints, scoremap_list[-1])
            print(loss)

        with tf.variable_scope('upscale_pred'):
            pred_upscale = scoremap_list[-1]
            pred_upscale = tf.transpose(pred_upscale, [0, 2, 3, 1])
            pred_upscale = tf.image.resize_images(pred_upscale, (CROPSIZE, CROPSIZE), method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
            pred_upscale = tf.transpose(pred_upscale, [0, 3, 1, 2])

        with tf.variable_scope('point_pred'):
            input = pred_upscale

            def image_max(t):
                t = tf.reshape(t, [-1])
                idx = tf.argmax(t)
                x = tf.cast(idx % CROPSIZE, dtype=tf.float32)
                y = tf.cast(idx // CROPSIZE, dtype=tf.float32)
                return x, y

            def outer(t):
                res = tf.map_fn(lambda j: image_max(j), tf.nn.embedding_lookup(t, np.array(range(input.shape[1]))),
                                dtype=(tf.float32, tf.float32), back_prop=TRAIN)
                return tf.stack([res[0], res[1]], axis=1)

            pred_points = tf.map_fn(lambda i: outer(i), tf.nn.embedding_lookup(input, np.array(range(input.shape[0]))),
                                    dtype=tf.float32, back_prop=TRAIN)

        loss_terms = {  # To optimize
            'kp_2D_mse': tf.reduce_mean(tf.squared_difference(scoremap_list[-1], heatmap_keypoints)),
        }
        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': pred_points}, loss_terms, {}