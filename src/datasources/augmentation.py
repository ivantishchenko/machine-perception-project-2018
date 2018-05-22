import matplotlib.pyplot as plt
from util.img_transformations import crop_hand, resize

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

colours={0:'black', 1:'blue', 2:'orange', 3:'green', 4:'red', 5:'yellow'}

def flipHorizontal(img, kp_2D):
    seq = iaa.Sequential([
        iaa.Fliplr(1.0) # Horizontal flip
    ])
    # create keypoints object
    keypoints = [ia.Keypoint(x=point[0], y=point[1]) for point in kp_2D]
    keypoints = ia.KeypointsOnImage(keypoints, shape=img.shape)

    # do augmentation
    image_aug = seq.augment_image(img)
    keypoints_aug = seq.augment_keypoints([keypoints])[0]

    #formating back to numpy
    keypoints_aug = np.array([[point.x, point.y] for point in keypoints_aug.keypoints])
    return image_aug, keypoints_aug


def showImage(img, kp_2D):
    kpx = kp_2D[:,0]
    kpy = kp_2D[:,1]
    plt.imshow(img)
    plt.scatter(kpx[0], kpy[0], color=colours[0])
    # Attach all knuckles to root of hand, then draw then fingers
    for i in range(1, 6):
        plt.plot([kpx[0], kpx[i*4]], [kpy[0], kpy[i*4]], color=colours[0])
        plt.plot(kpx[i*4-3:(i+1)*4-3], kpy[i*4-3:(i+1)*4-3], marker='o', color=colours[i])
    plt.show()

def defaultProcessingTrain(original_img, original_kp_2D):
    img = original_img.transpose(1,2,0)
    img = img / 255.0

    img, kp_2D = crop_hand(img, original_kp_2D)
    img, kp_2D = resize(img, kp_2D, (128, 128))

    return img, kp_2D