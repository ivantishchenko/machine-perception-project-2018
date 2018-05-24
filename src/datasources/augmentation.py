import matplotlib.pyplot as plt
from util.img_transformations import crop_hand, resize

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

NUM_TRANSFORMATIONS = 8
colours={0:'black', 1:'blue', 2:'orange', 3:'green', 4:'red', 5:'yellow'}

"""TRANSFORMATIONS"""
"""Perform salt and pepper"""
def salt_pepper(img, kp_2D, val=0.01):
    seq = iaa.Sequential([
        iaa.SaltAndPepper(val)
    ])
    image_aug = seq.augment_image(img)
    return image_aug, kp_2D

"""Perform dropout"""
def dropout(img, kp_2D, val=0.01):
    seq = iaa.Sequential([
        iaa.Dropout(val)
    ])
    image_aug = seq.augment_image(img)
    return image_aug, kp_2D

"""Change brightness"""
def change_brightness(img, kp_2D, factor=(0.6, 1.2)):
    seq = iaa.Sequential([
        iaa.Multiply(factor)
    ])
    image_aug = seq.augment_image(img)
    return image_aug, kp_2D

"""Change contrast"""
def change_contrast(img, kp_2D, factor=(0.999, 1.001)):
    seq = iaa.Sequential([
        iaa.ContrastNormalization(factor)
    ])
    image_aug = seq.augment_image(img)
    return image_aug, kp_2D

"""Shearing"""
def shear(img, kp_2D, angle=30):
    seq = iaa.Sequential([
        iaa.Affine(
            shear=angle,
            scale=0.7
        )
    ])
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    return image_aug, keypoints_aug

"""Rotation by angle"""
# WARNING CHECK VISIBILITY WHEN ROTATING BY NOT 90
def rotate(img, kp_2D, angle=90):
    if abs(angle) == 90 or abs(angle) == 180:
        transofrm = iaa.Affine(rotate=angle)
    else: 
        transofrm = iaa.Affine(rotate=angle, scale=0.7)
        
    seq = iaa.Sequential([transofrm])
    
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    # for entry in keypoints_aug:
    #     if entry[0] > 128 or entry[1] > 128 or entry[0] < 0 or entry[1] < 0:
    #         print('Lucky')
    return image_aug, keypoints_aug

"""Do a vertical flip img + kp"""
def flip_vertical(img, kp_2D):
    seq = iaa.Sequential([
        iaa.Flipud(1.0) # Vertical vertical
    ])
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    return image_aug, keypoints_aug

"""Do a horizontal flip img + kp"""
def flip_horizontal(img, kp_2D):
    # define an augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(1.0) # Horizontal flip
    ])
    image_aug, keypoints_aug = perform_augmentation_all(seq, img, kp_2D)
    return image_aug, keypoints_aug

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
    img = original_img.transpose(1,2,0)
    img = img / 255.0

    img, kp_2D = crop_hand(img, original_kp_2D)
    img, kp_2D = resize(img, kp_2D, (128, 128))

    return img, kp_2D