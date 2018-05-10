"""Development architecture."""
from typing import Dict
import tensorflow as tf
from core import BaseDataSource, BaseModel
from util.common_ops import NetworkOps as ops

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
        rgb_image = input_tensors['img']
        print('rgb_image_dims={}'.format(rgb_image.get_shape()))
        keypoints = input_tensors['kp_2D']
        print('keypoint_dims={}'.format(keypoints.get_shape()))

        with tf.variable_scope('keypoints'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 5]
            out_chan_list = [32, 64, 128, 256]
            pool_list = [True, True, True, False]

            image = rgb_image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    # print('conv{}_{}: input_dims={}'.format(block_id, layer_id+1, image.get_shape()))
                    image = ops.conv_relu(image, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, out_chan=chan_num, maxpool=(True if layer_id == range(layer_num)[-1] and pool else False), trainable=TRAIN)
                    print('conv{}_{}: output_dims={}'.format(block_id, layer_id+1, image.get_shape()))

            copy_image = image

            scoremap = ops.conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, name='conv4_6', trainable=TRAIN)
            print('conv4_6: output_dims={}'.format(scoremap.get_shape()))
            scoremap_list.append(scoremap)

            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 5
            num_recurrent_units = 2
            offset = 5
            for pass_id in range(num_recurrent_units):
                image = tf.concat([scoremap_list[0], copy_image], 1)
                if len(scoremap_list) > 1:
                    image = tf.concat([scoremap_list[1], image], 1)
                for rec_id in range(layers_per_recurrent_unit):
                    # print('conv{}_{}: input_dims={}'.format(pass_id + offset, rec_id+1, image.get_shape()))
                    image = ops.conv_relu(image, 'conv%d_%d' % (pass_id + offset, rec_id+1), kernel_size=5, out_chan=64, trainable=TRAIN)
                    print('conv{}_{}: output_dims={}'.format(pass_id + offset, rec_id+1, image.get_shape()))
                # print('conv{}_6: input_dims={}'.format(pass_id + offset, image.get_shape()))
                image = ops.conv_relu(image, 'conv%d_6' % (pass_id + offset), kernel_size=1, out_chan=64, trainable=TRAIN)
                print('conv{}_6: output_dims={}'.format(pass_id + offset, image.get_shape()))
                # print('conv{}_7: input_dims={}'.format(pass_id + offset, image.get_shape()))
                scoremap = ops.conv(image, kernel_size=1, out_chan=KEYPOINT_COUNT, name='conv%d_7' % (pass_id + offset), trainable=TRAIN)
                print('conv{}_7: output_dims={}'.format(pass_id + offset, scoremap.get_shape()))
                scoremap_list.append(scoremap)

            image = tf.concat([scoremap_list[0], scoremap_list[1], scoremap_list[2]], 1)
            print('final: output_dims={}'.format(image.get_shape()))

        with tf.variable_scope('fully_connected'):
            predictions = tf.contrib.layers.flatten(image)
            print('pred: flat_dims={}'.format(predictions.get_shape()))

            predictions = ops.fc_lru(predictions, layer_name='full_conn1', out_chan=1024, trainable=TRAIN)
            print('pred: fc_dims={}'.format(predictions.get_shape()))

            predictions = ops.fc_lru(predictions, layer_name='full_conn2', out_chan=512, trainable=TRAIN)
            print('pred: fc_dims={}'.format(predictions.get_shape()))

            predictions = ops.fc_lru(predictions, layer_name='prediction', out_chan=2*KEYPOINT_COUNT, disable_dropout=True, trainable=TRAIN)
            print('pred: fc_dims={}'.format(predictions.get_shape()))

            predictions = tf.reshape(predictions, (-1, KEYPOINT_COUNT, 2))
            print('pred: reshape_dims={}'.format(predictions.get_shape()))

        loss_terms = {  # To optimize
            'kp_2D_mse': tf.reduce_mean(tf.squared_difference(predictions, keypoints)),
        }
        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': predictions}, loss_terms, {}