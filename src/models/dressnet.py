"""Development architecture."""
from typing import Dict
import tensorflow as tf
import numpy as np
from core import BaseDataSource, BaseModel
from util.common_ops import ResNetLayers as rnl


# HYPER PARAMETERS
CROPSIZE = 128
ACCURACY_BOX = 2
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

        with tf.variable_scope('resnet18'):
            image = resnet.init_block(rgb_image, self.is_training)
            for i, layers in enumerate(resnet_repetitions_small):
                for j in range(layers):
                    image = resnet.vanilla(image, layer_name='conv%d_%d' % (i + 2, j + 1),
                                              first_layer=(j == 0), out_chan=resnet_channels[i],
                                              is_training=self.is_training)
            # image = resnet._max_pool(image, pool=4)
            image = resnet.last_layer(image, is_training=self.is_training, use_4k=False)
            self.summary.histogram('last_layer', image)
            self.summary.feature_maps('last_layer', image)

        with tf.variable_scope('flatten'):
            image = resnet.output_layer(image, is_training=self.is_training, use_4k=False)
            result = tf.reshape(image, (-1, 21, 2))

        with tf.variable_scope('loss_calculation'):
            # def inner_loss_mse(t):
            #     loss_image = tf.map_fn(lambda e: lf.rmse((e[0], e[1]), (e[2], e[3]), 0),
            #                            tf.nn.embedding_lookup(t, np.array(range(t.shape[0]))))
            #     return loss_image
            #
            # def outer_loss_mse(t):
            #     loss_image = tf.reduce_mean(tf.map_fn(lambda kp: inner_loss_mse(kp),
            #                                           tf.nn.embedding_lookup(t, np.array(range(t.shape[0])))))
            #     return loss_image
            #
            # def inner_loss_mse_vis(t):
            #     loss_image = tf.map_fn(lambda e: lf.rmse((e[0], e[1]), (e[2], e[3]), e[4]),
            #                            tf.nn.embedding_lookup(t, np.array(range(t.shape[0]))))
            #     return loss_image
            #
            # def outer_loss_mse_vis(t):
            #     loss_image = tf.reduce_mean(tf.map_fn(lambda kp: inner_loss_mse_vis(kp),
            #                                           tf.nn.embedding_lookup(t, np.array(range(t.shape[0])))))
            #     return loss_image
            #
            # def inner_correct(t):
            #     accuracy_image = tf.map_fn(lambda e: lf.accuracy_euclidean_distance((e[0], e[1]), (e[2], e[3]), ACCURACY_BOX),
            #                                tf.nn.embedding_lookup(t, np.array(range(t.shape[0]))))
            #     return accuracy_image
            #
            # def outer_correct(t):
            #     accuracy_image = tf.reduce_sum(tf.map_fn(lambda image: inner_correct(image),
            #                                    tf.nn.embedding_lookup(t, np.array(range(t.shape[0])))))
            #     return accuracy_image
            #
            # def inner_correct_vis(t):
            #     accuracy_image = tf.map_fn(lambda e: lf.accuracy_euclidean_distance_vis((e[0], e[1]), (e[2], e[3]), ACCURACY_BOX, e[4]),
            #                                tf.nn.embedding_lookup(t, np.array(range(t.shape[0]))))
            #     return accuracy_image
            #
            # def outer_correct_vis(t):
            #     accuracy_image = tf.reduce_sum(tf.map_fn(lambda image: inner_correct_vis(image),
            #                                    tf.nn.embedding_lookup(t, np.array(range(t.shape[0])))))
            #     return accuracy_image
            #
            # visible_tuples = tf.concat([keypoints, result, is_visible], axis=2)
            # hard_tuples = tf.concat([keypoints, result], axis=2)
            # loss_mse_vis = tf.reduce_mean(tf.map_fn(lambda img: outer_loss_mse_vis(img),
            #                                         tf.nn.embedding_lookup(visible_tuples,
            #                                                                np.array(range(visible_tuples.shape[0]))),
            #                                         dtype=tf.float32))
            # loss_mse = tf.reduce_mean(tf.map_fn(lambda img: outer_loss_mse(img),
            #                                     tf.nn.embedding_lookup(hard_tuples,
            #                                                            np.array(range(hard_tuples.shape[0]))),
            #                                     dtype=tf.float32))
            # corr_vis = tf.reduce_sum(tf.map_fn(lambda img: outer_correct_vis(img),
            #                                    tf.nn.embedding_lookup(visible_tuples,
            #                                                           np.array(range(visible_tuples.shape[0]))),
            #                                    dtype=tf.float32))
            # corr = tf.reduce_sum(tf.map_fn(lambda img: outer_correct(img),
            #                                tf.nn.embedding_lookup(hard_tuples,
            #                                                       np.array(range(hard_tuples.shape[0]))),
            #                                dtype=tf.float32))
            # precision_visible = corr_vis / (keypoints.shape[0] * keypoints.shape[1] * keypoints.shape[2])

            # Include all keypoints for metrics. These are rougher scores.
            loss_mse = tf.reduce_mean(tf.squared_difference(keypoints, result))
            corr = tf.count_nonzero(tf.less_equal(tf.squared_difference(keypoints, result), ACCURACY_BOX))
            precision = corr / (keypoints.shape[0] * keypoints.shape[1] * keypoints.shape[2])

            # Only include visible keypoints for metrics. These are nicer scores overall.
            count_vis = tf.count_nonzero(tf.multiply(keypoints, is_visible))
            loss_mse_vis = tf.multiply(tf.squared_difference(keypoints, result), is_visible)
            loss_mse_vis = tf.reduce_sum(tf.truediv(loss_mse_vis, tf.cast(count_vis, dtype=tf.float32)))
            corr_vis = tf.count_nonzero(tf.less_equal(tf.multiply(tf.squared_difference(keypoints, result), is_visible), ACCURACY_BOX))
            precision_visible = tf.divide(corr_vis, count_vis)

        loss_terms = {  # To optimize
            'kp_loss_mse': loss_mse,
            'kp_accuracy': precision,
            'kp_loss_mse_vis': loss_mse_vis,
            'kp_accuracy_vis': precision_visible
        }
        # Return output_tensor, loss_tensor and metrics (not used)
        return {'kp_2D': result}, loss_terms, {}
