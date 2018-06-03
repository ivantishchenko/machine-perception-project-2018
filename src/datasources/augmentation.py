import matplotlib.pyplot as plt
from util.img_transformations import crop_hand, resize

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

NUM_JOINTS = 21
NUM_TRANSFORMATIONS = 8
colours = {0: 'black', 1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'yellow'}


"""TRANSFORMATIONS"""


def salt_pepper(img, val=0.01):
    """Perform salt and pepper"""
    seq = iaa.Sequential([
        iaa.SaltAndPepper(val)
    ])
    image_aug = seq.augment_image(img)
    return image_aug


def dropout(img, val=0.01):
    """Perform dropout"""
    seq = iaa.Sequential([
        iaa.Dropout(val)
    ])
    image_aug = seq.augment_image(img)
    return image_aug


def change_brightness(img, factor=(0.8, 1.2)):
    """Change brightness"""
    seq = iaa.Sequential([
        iaa.Multiply(factor)
    ])
    image_aug = seq.augment_image(img)
    return image_aug


def change_contrast(img, factor=(0.8, 1.2)):
    """Change contrast"""
    seq = iaa.Sequential([
        iaa.ContrastNormalization(factor)
    ])
    image_aug = seq.augment_image(img)
    return image_aug


def shear(img, kp_2D, angle=None, scale=1):
    """Shearing"""
    if angle is None:
        transform = iaa.Affine(shear=(-15, 15), scale=0.78)  # scale = 1 / (tan(shear) + 1)
    else:
        transform = iaa.Affine(shear=angle, scale=scale)
    seq = iaa.Sequential([transform]).to_deterministic()
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    return image_aug, keypoints_aug


def rotate(img, kp_2D, angle=None, scale=1):
    """Rotation by angle"""
    if angle is None:
        transform = iaa.Affine(rotate=(-45, 45), scale=0.70)  # scale = 1 / sqrt(2)
    else:
        transform = iaa.Affine(rotate=angle, scale=scale)
    seq = iaa.Sequential([transform]).to_deterministic()
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    return image_aug, keypoints_aug

def scale(img, kp_2D, scale=1):
    """Scale by amount"""
    seq = iaa.Sequential([iaa.Affine(scale=scale)]).to_deterministic()
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    return image_aug, keypoints_aug


def flip_vertical(img, kp_2D):
    """Do a vertical flip img + kp"""
    seq = iaa.Sequential([
        iaa.Flipud(1.0)  # Vertical vertical
    ])
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    return image_aug, keypoints_aug


def flip_horizontal(img, kp_2D):
    """Do a horizontal flip img + kp"""
    # define an augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(1.0)  # Horizontal flip
    ])
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    # keypoints_aug[:, 0] += 1
    return image_aug, keypoints_aug


def get_vis(keypoints_aug, original_visibility, img_bound=128):
    """Check visibility of the keypoints"""
    vis1 = np.any(keypoints_aug > img_bound + 1, axis=1)
    vis2 = np.any(keypoints_aug < -1, axis=1)
    vis = ~(vis1 + vis2)
    vis = vis.astype(int).reshape(21, 1)
    vis = np.bitwise_and(original_visibility, vis)
    return vis

# def check_vis_inif(keypoints_aug):
#     vis = np.ones((21, 1))
#     for i in range(len(keypoints_aug)):
#         if keypoints_aug[i][0] > 128 or keypoints_aug[i][1] > 128 or keypoints_aug[i][0] < 0 or keypoints_aug[i][1] < 0:
#             vis[i] = 0
#     return vis


"""INTERNAL FUNCTIONS"""
def perform_augmentation_all(seq, img, kp_2D):
    # create keypoints object
    keypoints = create_keypoints_object(kp_2D, img.shape)
    # do augmentation
    image_aug, keypoints_aug = augment(seq, img, keypoints)
    #formating back to numpy
    keypoints_aug = format_nparray(keypoints_aug)
    return image_aug, keypoints_aug

def format_nparray(keypoints):
    keypoints_aug = np.array([[point.x, point.y] for point in keypoints.keypoints])
    return keypoints_aug

def augment(seq, img, keypoints):
    image_aug = seq.augment_image(img)
    keypoints_aug = seq.augment_keypoints([keypoints])[0]
    return image_aug, keypoints_aug

def create_keypoints_object(kp_2D, img_shape):
    keypoints = [ia.Keypoint(x=point[0], y=point[1]) for point in kp_2D]
    keypoints = ia.KeypointsOnImage(keypoints, shape=img_shape)
    return keypoints

"""LOCAL TESTING"""

def show_image(img, kp_2D):
    kpx = kp_2D[:,0]
    kpy = kp_2D[:,1]
    plt.imshow(img)
    plt.scatter(kpx[0], kpy[0], color=colours[0])
    # Attach all knuckles to root of hand, then draw then fingers
    for i in range(1, 6):
        plt.plot([kpx[0], kpx[i*4]], [kpy[0], kpy[i*4]], color=colours[0])
        plt.plot(kpx[i*4-3:(i+1)*4-3], kpy[i*4-3:(i+1)*4-3], marker='o', color=colours[i])
    plt.show()

def default_processing(original_img, original_kp_2D):
    img = original_img.transpose(1, 2, 0)

    img, kp_2D = crop_hand(img, original_kp_2D)
    img, kp_2D = resize(img, kp_2D, (128, 128))

    return img, kp_2D