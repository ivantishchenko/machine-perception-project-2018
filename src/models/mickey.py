"""Development architecture."""
from typing import Dict
import tensorflow as tf
import numpy as np
from core import BaseDataSource, BaseModel
from util.common_ops import NetworkOps as ops
import cv2

# HYPER PARAMETERS
CROPSIZE = 128
KEYPOINT_COUNT = 21
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
        heatmap_keypoints = []
        heatmap_keypoints = tf.map_fn(lambda i: self.get_single_heatmap(i, 16, 1.0, CROPSIZE // 16), tf.nn.embedding_lookup(keypoints, np.array(range(keypoints.shape[0]))),
                                dtype=(tf.float32, tf.float32), back_prop=False)
        # for i in keypoints.shape[0]:
        #     heatmap_keypoints.append(self.get_single_heatmap(keypoints[i], 16, 1.0, CROPSIZE // 16))
        scoremap_list = list()

        with tf.variable_scope('keypoints'):
            layers_per_block = [2, 2, 4, 5]
            out_chan_list = [32, 64, 128, 256]
            pool_list = [True, True, True, False]

            image = rgb_image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    # print('conv{}_{}: input_dims={}'.format(block_id, layer_id+1, image.get_shape()))
                    image = ops.conv_relu(image, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, out_chan=chan_num, maxpool=(True if layer_id == range(layer_num)[-1] and pool else False), trainable=TRAIN)
                    print('conv{}_{}: output_dims={}'.format(block_id, layer_id+1, image.get_shape()))

            downsampled_map = ops.conv_relu(image, 'conv4_D', kernel_size=3, out_chan=64, trainable=TRAIN)
            print('conv4_D: output_dims={}'.format(downsampled_map.get_shape()))
            image = ops.conv_relu(downsampled_map, 'conv5_1', kernel_size=1, out_chan=256, trainable=TRAIN)
            print('conv5_1: output_dims={}'.format(image.get_shape()))
            scoremap = ops.conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, name='conv5_P', trainable=TRAIN)
            print('conv5_P: output_dims={}'.format(scoremap.get_shape()))
            scoremap_list.append(scoremap)

            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 5
            num_recurrent_units = 2
            offset = 6
            for pass_id in range(num_recurrent_units):
                image = tf.concat([scoremap_list[-1], downsampled_map], 1)
                for rec_id in range(layers_per_recurrent_unit):
                    print('conv{}_{}: input_dims={}'.format(pass_id + offset, rec_id+1, image.get_shape()))
                    image = ops.conv_relu(image, 'conv%d_%d' % (pass_id + offset, rec_id+1), kernel_size=5, out_chan=64, trainable=TRAIN)
                    print('conv{}_{}: output_dims={}'.format(pass_id + offset, rec_id+1, image.get_shape()))
                # print('conv{}_6: input_dims={}'.format(pass_id + offset, image.get_shape()))
                image = ops.conv_relu(image, 'conv%d_%d' % (pass_id + offset, layers_per_recurrent_unit+1), kernel_size=1, out_chan=64, trainable=TRAIN)
                print('conv{}_{}: output_dims={}'.format(pass_id + offset, layers_per_recurrent_unit+1, image.get_shape()))
                # print('conv{}_7: input_dims={}'.format(pass_id + offset, image.get_shape()))
                scoremap = ops.conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, name='conv%d_P' % (pass_id + offset), trainable=TRAIN)
                print('conv{}_P: output_dims={}'.format(pass_id + offset, scoremap.get_shape()))
                scoremap_list.append(scoremap)

            print('final: {}'.format(scoremap_list))

        with tf.variable_scope('downsized_pred'):
            print(keypoints)
            print(heatmap_keypoints)
            print(scoremap_list[-1])


        with tf.variable_scope('upscale_pred'):
            pred_upscale = scoremap_list[-1]
            pred_upscale = tf.transpose(pred_upscale, [0, 2, 3, 1])
            pred_upscale = tf.image.resize_images(pred_upscale, (CROPSIZE, CROPSIZE))
            pred_upscale = tf.transpose(pred_upscale, [0, 3, 1, 2])
            print(pred_upscale)


        with tf.variable_scope('point_pred'):
            input = pred_upscale

            def image_max(tensor):
                tensor = tf.reshape(tensor, [-1])
                idx = tf.argmax(tensor)
                return idx % CROPSIZE, idx // CROPSIZE

            def outer(tensor):
                return tf.map_fn(lambda j: image_max(j), tf.nn.embedding_lookup(tensor, np.array(range(input.shape[1]))), dtype=(tf.float32, tf.float32), back_prop=TRAIN)

            pred_points = tf.get_variable("empty", [32, 42])
            pred_points = tf.map_fn(lambda i: outer(i), tf.nn.embedding_lookup(input, np.array(range(input.shape[0]))), dtype=(tf.float32, tf.float32), back_prop=TRAIN)
            pred_points = tf.reshape(pred_points, [input.shape[0], input.shape[1], 2])
            # print(tf.trainable_variables())

        loss_terms = {  # To optimize
            'kp_2D_mse': tf.reduce_mean(tf.squared_difference(pred_points, keypoints)),
        }
        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': pred_points}, loss_terms, {}

    def make_gaussian(self, size, fwhm=3, centre=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        if centre is None:
            x0 = y0 = size // 2
        else:
            x0 = centre[0]
            y0 = centre[1]
        return tf.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)

    def get_single_heatmap(self, hand_joints, heatmap_size, gaussian_variance, scale_factor):
        gt_heatmap_np = []
        invert_heatmap_np = tf.ones(shape=(16, 16))
        cur_joint_heatmap = []
        for j in range(hand_joints.shape[0]):
            cur_joint_heatmap = self.make_gaussian(heatmap_size,
                                                   gaussian_variance,
                                                   centre=(hand_joints[j] // scale_factor))
            print(cur_joint_heatmap)
            gt_heatmap_np.append(cur_joint_heatmap)
            invert_heatmap_np -= cur_joint_heatmap
        gt_heatmap_np.append(invert_heatmap_np)
        print(gt_heatmap_np)
        return gt_heatmap_np