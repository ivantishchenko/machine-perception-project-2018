# import os
# import sys
#
# src_dir, _ = os.path.split(os.path.dirname(os.path.realpath(sys.argv[0])))
# print(src_dir)
# if not src_dir in sys.path:
#     sys.path.insert(0, src_dir)
# del src_dir

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util.common_ops import ImageOps as iop

CROPSIZE = 128

tensor_np = np.array([[
                        [1, 5], [19, 32.], [104, 44], [90, 11], [55, 22],
                        [88, 128], [32, 65], [91, 38], [73, 40], [99, 10], [21, 1], [5, 19],
                        [32, 104], [44, 90], [11, 55], [22, 88], [128, 32], [65, 91], [38, 73], [40, 99], [10, 21]
                    ]])
tensor = tf.convert_to_tensor(tensor_np, dtype=tf.float32)
gt_np = 128 * np.random.random_sample(size=(1, 21, 2))
gt = tf.convert_to_tensor(gt_np, dtype=tf.float32)
model = tf.global_variables_initializer()

with tf.Session() as session:
    heatmap_keypoints, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, 16, 1.0, CROPSIZE // 16),
                                     tf.nn.embedding_lookup(tensor, np.array(range(tensor.shape[0]))),
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

    normalized_heatmap, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, 16, 1.0, CROPSIZE // 16, True),
                                     tf.nn.embedding_lookup(tensor, np.array(range(tensor.shape[0]))),
                                     dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                                     back_prop=False)
    normalized_heatmap = tf.stack(normalized_heatmap)
    normalized_heatmap = tf.transpose(normalized_heatmap, perm=[1, 0, 2, 3])

    gt_keypoints, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, 16, 1.0, CROPSIZE // 16),
                                     tf.nn.embedding_lookup(gt, np.array(range(gt.shape[0]))),
                                     dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                                     back_prop=False)
    gt_keypoints = tf.stack(gt_keypoints)
    gt_keypoints = tf.transpose(gt_keypoints, perm=[1, 0, 2, 3])

    gt_keypoints_norm, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, 16, 1.0, CROPSIZE // 16, True),
                                     tf.nn.embedding_lookup(gt, np.array(range(gt.shape[0]))),
                                     dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                                     back_prop=False)
    gt_keypoints_norm = tf.stack(gt_keypoints_norm)
    gt_keypoints_norm = tf.transpose(gt_keypoints_norm, perm=[1, 0, 2, 3])

    u_l = tf.reduce_mean(tf.squared_difference(gt_keypoints, heatmap_keypoints))
    u_lm = tf.losses.mean_squared_error(gt_keypoints, heatmap_keypoints)
    u_lmp = tf.losses.mean_pairwise_squared_error(gt_keypoints, heatmap_keypoints)

    n_l = tf.reduce_mean(tf.squared_difference(gt_keypoints_norm, normalized_heatmap))
    n_lm = tf.losses.mean_squared_error(gt_keypoints_norm, normalized_heatmap)
    n_lmp = tf.losses.mean_pairwise_squared_error(gt_keypoints_norm, normalized_heatmap)

    numerical = tf.reduce_mean(tf.squared_difference(tensor, gt))

    heatmap_large, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, CROPSIZE, 1.0, CROPSIZE // CROPSIZE),
                                     tf.nn.embedding_lookup(tensor, np.array(range(tensor.shape[0]))),
                                     dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                                     back_prop=False)
    heatmap_large = tf.stack(heatmap_large)
    heatmap_large = tf.transpose(heatmap_large, perm=[1, 0, 2, 3])


    pred_upscale = heatmap_keypoints
    pred_upscale = tf.transpose(pred_upscale, [0, 2, 3, 1])
    pred_upscale = tf.image.resize_images(pred_upscale, (CROPSIZE, CROPSIZE), method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
    pred_upscale = tf.transpose(pred_upscale, [0, 3, 1, 2])
    # pred_upscale = tf.map_fn(lambda i: tf.image.per_image_standardization(i), tf.nn.embedding_lookup(pred_upscale, np.array(range(pred_upscale.shape[0]))),
    #                          back_prop=False)

    input = pred_upscale

    def image_max(t):
        t = tf.reshape(t, [-1])
        idx = tf.argmax(t)
        x = tf.cast(idx % CROPSIZE, dtype=tf.float32)
        y = tf.cast(idx // CROPSIZE, dtype=tf.float32)
        return x, y

    def outer(t):
        res = tf.map_fn(lambda j: image_max(j), tf.nn.embedding_lookup(t, np.array(range(input.shape[1]))),
                         dtype=(tf.float32, tf.float32), back_prop=False)
        return tf.stack([res[0], res[1]], axis=1)

    pred_points = tf.map_fn(lambda i: outer(i), tf.nn.embedding_lookup(input, np.array(range(input.shape[0]))),
                            dtype=tf.float32, back_prop=False)

    session.run(model)
    [u_p_np, n_p_np, u_gt_np, n_gt_np] = session.run([heatmap_keypoints, normalized_heatmap, gt_keypoints, gt_keypoints_norm])
    [u_l_np, u_lm_np, u_lmp_np, n_l_np, n_lm_np, n_lmp_np] = session.run([u_l, u_lm, u_lmp, n_l, n_lm, n_lmp])
    pred_np = session.run(pred_upscale)
    numerical_np = session.run(numerical)
    heatmap_large_np = session.run(heatmap_large)
    points_np = session.run(pred_points)

u_loss_np = 0
n_loss_np = 0
for i in range(pred_np.shape[1]):
    u_image = np.sum(np.square(np.subtract(u_gt_np[0][i], u_p_np[0][i]))).mean()
    n_image = np.sum(np.square(np.subtract(n_gt_np[0][i], n_p_np[0][i]))).mean()
    u_loss_np += u_image
    n_loss_np += n_image
print(numerical_np)
print("{} {} {}".format(u_l_np, u_lm_np, u_lmp_np))
print("{} {}".format(u_loss_np, u_loss_np / pred_np.shape[1]))
print("{} {} {}".format(n_l_np, n_lm_np, n_lmp_np))
print("{} {}".format(n_loss_np, n_loss_np / pred_np.shape[1]))

scale_loss = 0
for i in range(pred_np.shape[1]):
    s_image = np.sum(np.square(np.subtract(points_np[0][i], tensor_np[0][i]))).mean()
    scale_loss += s_image
print("{} {}".format(scale_loss, scale_loss / pred_np.shape[1]))

# # Print scaled images with verifier
# np_preds = []
# for b in range(pred_np.shape[0]):
#     np_batch = []
#     for c in range(pred_np.shape[1]):
#         temp = np.reshape(pred_np[b][c], -1)
#         idx = np.argmax(temp)
#         np_batch.append([idx % CROPSIZE, idx // CROPSIZE])
#     np_preds.append(np_batch)
#
# for i in range(pred_np.shape[1]):
#     fig = plt.figure()
#
#     a = fig.add_subplot(1,3,1)
#     plt.imshow(pred_np[0][i])
#     a.set_title('upscale: {} ({})'.format(points_np[0][i], np_preds[0][i]))
#     a = fig.add_subplot(1,3,2)
#     plt.imshow(heatmap_large_np[0][i])
#     a.set_title('orig: {}'.format(tensor_np[0][i]))
#     a = fig.add_subplot(1,3,3)
#     plt.imshow(np.square(np.subtract(pred_np[0][i], heatmap_large_np[0][i])))
#     mse = np.sum(np.square(np.subtract(pred_np[0][i], heatmap_large_np[0][i]))).mean()
#     a.set_title('diff = {}'.format(mse))
#
#     plt.show()

# # Print kernel-sized images and diffs and such
# for i in range(pred_np.shape[1]):
#     fig = plt.figure()
#
#     a = fig.add_subplot(4,3,1)
#     plt.imshow(n_p_np[0][i])
#     a.set_title('n_hm')
#     a = fig.add_subplot(4,3,2)
#     plt.imshow(u_p_np[0][i])
#     a.set_title('u_hm')
#     a = fig.add_subplot(4,3,3)
#     plt.imshow(np.square(np.subtract(n_p_np[0][i], u_p_np[0][i])))
#     mse = np.square(np.subtract(n_p_np[0][i], u_p_np[0][i])).mean()
#     a.set_title('Diff = {}'.format(mse))
#
#     a = fig.add_subplot(4,3,4)
#     plt.imshow(np.square(np.subtract(u_gt_np[0][i], n_p_np[0][i])))
#     mse = np.sum(np.square(np.subtract(u_gt_np[0][i], n_p_np[0][i]))).mean()
#     a.set_title('diff_un = {}'.format(mse))
#     a = fig.add_subplot(4,3,5)
#     plt.imshow(np.square(np.subtract(u_gt_np[0][i], u_p_np[0][i])))
#     mse = np.sum(np.square(np.subtract(u_gt_np[0][i], u_p_np[0][i]))).mean()
#     a.set_title('diff_uu = {}'.format(mse))
#     a = fig.add_subplot(4,3,6)
#     plt.imshow(u_gt_np[0][i])
#     a.set_title('u_gt')
#
#     a = fig.add_subplot(4,3,7)
#     plt.imshow(np.square(np.subtract(n_gt_np[0][i], n_p_np[0][i])))
#     mse = np.sum(np.square(np.subtract(n_gt_np[0][i], n_p_np[0][i]))).mean()
#     a.set_title('diff_nn = {}'.format(mse))
#     a = fig.add_subplot(4,3,8)
#     plt.imshow(np.square(np.subtract(n_gt_np[0][i], u_p_np[0][i])))
#     mse = np.sum(np.square(np.subtract(n_gt_np[0][i], u_p_np[0][i]))).mean()
#     a.set_title('diff_nu = {}'.format(mse))
#     a = fig.add_subplot(4,3,9)
#     plt.imshow(n_gt_np[0][i])
#     a.set_title('n_gt')
#
#     a = fig.add_subplot(4,3,10)
#     plt.imshow(pred_np[0][i])
#     a.set_title('upscale')
#     a = fig.add_subplot(4,3,11)
#     plt.imshow(heatmap_large_np[0][i])
#     a.set_title('orig')
#     a = fig.add_subplot(4,3,12)
#     plt.imshow(np.square(np.subtract(pred_np[0][i], heatmap_large_np[0][i])))
#     mse = np.sum(np.square(np.subtract(pred_np[0][i], heatmap_large_np[0][i]))).mean()
#     a.set_title('diff = {}'.format(mse))
#
#     plt.show()
