# import os
# import sys
#
# src_dir, _ = os.path.split(os.path.dirname(os.path.realpath(sys.argv[0])))
# print(src_dir)
# if not src_dir in sys.path:
#     sys.path.insert(0, src_dir)
# del src_dir

import h5py
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util.common_ops import ImageOps as iop

SEED = 42
CROPSIZE = 320
HEATMAPSIZE = 16
BATCHSIZE = 32

# Read in here some random images and corresponding ground truth values
project_root = os.path.dirname(os.path.abspath(os.path.dirname(sys.argv[0])))
train_file = os.path.join(project_root, 'datasets', 'training.h5')
f = h5py.File(train_file, 'r')
n_samples = len(f['train']['img'])
index_array = list(range(n_samples))
# np.random.seed(SEED)
permuted_indexes = np.random.permutation(index_array)
gt_np = np.empty([BATCHSIZE, 21, 2])

for i, idx in enumerate(permuted_indexes[:BATCHSIZE], 0):
    gt_np[i] = f['train']['kp_2D'][idx]
gt = tf.convert_to_tensor(gt_np, dtype=tf.float32)

rand_guesses_np = CROPSIZE * np.random.random_sample(size=(BATCHSIZE, 21, 2))
rand_guesses = tf.convert_to_tensor(rand_guesses_np, dtype=tf.float32)

model = tf.global_variables_initializer()

with tf.Session() as session:
    u_hm_pred, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, HEATMAPSIZE, 1.0, CROPSIZE // HEATMAPSIZE),
                             tf.nn.embedding_lookup(rand_guesses, np.array(range(rand_guesses.shape[0]))),
                             dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                             back_prop=False)
    u_hm_pred = tf.stack(u_hm_pred)
    u_hm_pred = tf.transpose(u_hm_pred, perm=[1, 0, 2, 3])

    n_hm_pred, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, HEATMAPSIZE, 1.0, CROPSIZE // HEATMAPSIZE, True),
                             tf.nn.embedding_lookup(rand_guesses, np.array(range(rand_guesses.shape[0]))),
                             dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                             back_prop=False)
    n_hm_pred = tf.stack(n_hm_pred)
    n_hm_pred = tf.transpose(n_hm_pred, perm=[1, 0, 2, 3])

    u_hm_gt, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, HEATMAPSIZE, 1.0, CROPSIZE // HEATMAPSIZE),
                           tf.nn.embedding_lookup(gt, np.array(range(gt.shape[0]))),
                           dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                           back_prop=False)
    u_hm_gt = tf.stack(u_hm_gt)
    u_hm_gt = tf.transpose(u_hm_gt, perm=[1, 0, 2, 3])

    n_hm_gt, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, HEATMAPSIZE, 1.0, CROPSIZE // HEATMAPSIZE, True),
                           tf.nn.embedding_lookup(gt, np.array(range(gt.shape[0]))),
                           dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                           back_prop=False)
    n_hm_gt = tf.stack(n_hm_gt)
    n_hm_gt = tf.transpose(n_hm_gt, perm=[1, 0, 2, 3])

    u_l = (HEATMAPSIZE ** 2) * tf.reduce_mean(tf.squared_difference(u_hm_gt, u_hm_pred))
    u_lm = (HEATMAPSIZE ** 2) * tf.losses.mean_squared_error(u_hm_gt, u_hm_pred)

    n_l = (HEATMAPSIZE ** 2) * tf.reduce_mean(tf.squared_difference(n_hm_gt, n_hm_pred))
    n_lm = (HEATMAPSIZE ** 2) * tf.losses.mean_squared_error(n_hm_gt, n_hm_pred)

    numerical = tf.reduce_mean(tf.squared_difference(rand_guesses, gt))

    u_hm_pred_fs, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, CROPSIZE, 1.0, CROPSIZE // CROPSIZE),
                                tf.nn.embedding_lookup(rand_guesses, np.array(range(rand_guesses.shape[0]))),
                                dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                                back_prop=False)
    u_hm_pred_fs = tf.stack(u_hm_pred_fs)
    u_hm_pred_fs = tf.transpose(u_hm_pred_fs, perm=[1, 0, 2, 3])

    u_hm_gt_fs, _ = tf.map_fn(lambda i: iop.get_single_heatmap(i, CROPSIZE, 1.0, CROPSIZE // CROPSIZE),
                                tf.nn.embedding_lookup(gt, np.array(range(gt.shape[0]))),
                                dtype=([tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                             tf.float32,
                                             ], tf.float32),
                                back_prop=False)
    u_hm_gt_fs = tf.stack(u_hm_gt_fs)
    u_hm_gt_fs = tf.transpose(u_hm_gt_fs, perm=[1, 0, 2, 3])

    pred_upscale = u_hm_pred
    pred_upscale = tf.transpose(pred_upscale, [0, 2, 3, 1])
    pred_upscale = tf.image.resize_images(pred_upscale, (CROPSIZE, CROPSIZE), method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
    pred_upscale = tf.transpose(pred_upscale, [0, 3, 1, 2])

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
    [u_p_np, n_p_np, u_gt_np, n_gt_np] = session.run([u_hm_pred, n_hm_pred, u_hm_gt, n_hm_gt])
    [u_l_np, u_lm_np, n_l_np, n_lm_np] = session.run([u_l, u_lm, n_l, n_lm])
    numerical_np = session.run(numerical)
    [u_hm_pred_fs_np, u_hm_gt_fs_np] = session.run([u_hm_pred_fs, u_hm_gt_fs])
    pred_np = session.run(pred_upscale)
    points_np = session.run(pred_points)

u_loss_np = 0
n_loss_np = 0
for i in range(pred_np.shape[1]):
    u_image = np.sum(np.square(np.subtract(u_gt_np[0][i], u_p_np[0][i]))).mean()
    n_image = np.sum(np.square(np.subtract(n_gt_np[0][i], n_p_np[0][i]))).mean()
    u_loss_np += u_image
    n_loss_np += n_image
print("Unnormalized losses:")
print("    Manual = %8.3f    Layer  = %8.3f" % (u_l_np, u_lm_np))
print("    MSE    = %8.3f    SE     = %8.3f" % (u_loss_np / pred_np.shape[1], u_loss_np))
print("    Validate:   %3.1f" % ((u_loss_np / pred_np.shape[1]) / u_lm_np))
print("Normalized losses:")
print("    Manual = %8.3f    Layer  = %8.3f" % (n_l_np, n_lm_np))
print("    MSE    = %8.3f    SE     = %8.3f" % (n_loss_np / pred_np.shape[1], n_loss_np))
print("    Validate:   %3.1f" %((n_loss_np / pred_np.shape[1]) / n_lm_np))

scale_loss = 0
for i in range(pred_np.shape[1]):
    s_image = np.sum(np.square(np.subtract(points_np[0][i], rand_guesses_np[0][i]))).mean()
    scale_loss += s_image
print("Coordinate loss:")
print("    %8.3f" % numerical_np)
print("Upscaling loss:")
print("    MSE    = %8.3f    SE     = %8.3f" % (scale_loss / pred_np.shape[1], scale_loss))

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

# Print kernel-sized images and diffs and such
# for i in range(pred_np.shape[1]):
#     fig = plt.figure()
#
#     a = fig.add_subplot(3,5,1)
#     plt.imshow(u_hm_gt_fs_np[0][i])
#     a.set_title('gt_orig {}'.format(gt_np[0][i]))
#     a = fig.add_subplot(3,5,2)
#     plt.imshow(n_p_np[0][i])
#     a.set_title('n_hm')
#     a = fig.add_subplot(3,5,3)
#     plt.imshow(u_p_np[0][i])
#     a.set_title('u_hm')
#     a = fig.add_subplot(3,5,4)
#     plt.imshow(pred_np[0][i])
#     a.set_title('upscale {}'.format(points_np[0][i]))
#     a = fig.add_subplot(3,5,5)
#     plt.imshow(u_hm_pred_fs_np[0][i])
#     a.set_title('guess_orig {}'.format(rand_guesses_np[0][i]))
#     # plt.imshow(np.square(np.subtract(n_p_np[0][i], u_p_np[0][i])))
#     # mse = np.square(np.subtract(n_p_np[0][i], u_p_np[0][i])).mean()
#     # a.set_title('Diff = %.3f' % mse)
#
#     a = fig.add_subplot(3,5,6)
#     plt.imshow(u_gt_np[0][i])
#     a.set_title('u_gt')
#     a = fig.add_subplot(3,5,7)
#     plt.imshow(np.square(np.subtract(u_gt_np[0][i], n_p_np[0][i])))
#     mse = np.sum(np.square(np.subtract(u_gt_np[0][i], n_p_np[0][i]))).mean()
#     a.set_title('diff_un = %.3f' % mse)
#     a = fig.add_subplot(3,5,8)
#     plt.imshow(np.square(np.subtract(u_gt_np[0][i], u_p_np[0][i])))
#     mse = np.sum(np.square(np.subtract(u_gt_np[0][i], u_p_np[0][i]))).mean()
#     a.set_title('diff_uu = %.3f' % mse)
#
#     a = fig.add_subplot(3,5,11)
#     plt.imshow(n_gt_np[0][i])
#     a.set_title('n_gt')
#     a = fig.add_subplot(3,5,12)
#     plt.imshow(np.square(np.subtract(n_gt_np[0][i], n_p_np[0][i])))
#     mse = np.sum(np.square(np.subtract(n_gt_np[0][i], n_p_np[0][i]))).mean()
#     a.set_title('diff_nn = %.3f' % mse)
#     a = fig.add_subplot(3,5,13)
#     plt.imshow(np.square(np.subtract(n_gt_np[0][i], u_p_np[0][i])))
#     mse = np.sum(np.square(np.subtract(n_gt_np[0][i], u_p_np[0][i]))).mean()
#     a.set_title('diff_nu = %.3f' % mse)
#
#     # plt.imshow(np.square(np.subtract(pred_np[0][i], u_hm_pred_fs_np[0][i])))
#     # mse = np.sum(np.square(np.subtract(pred_np[0][i], u_hm_pred_fs_np[0][i]))).mean()
#
#     plt.show()
