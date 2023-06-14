# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# Directories
images_dir = 'data/processed-databases/deep-keypoint-detection/picture-db/images/anterior'
keypoints_dir = 'data/processed-databases/deep-keypoint-detection/picture-db/keypoints'
keypoints_pred_dir = 'data/processed-databases/deep-keypoint-detection/picture-db/predictions/keypoints'


# Get images list
images_fname_list = [i for i in os.listdir(images_dir) if not i.startswith('.')]

for image_fname in images_fname_list:
    image = Image.open(os.path.join(images_dir, image_fname)).convert('RGB')
    image = np.array(image)

    keypoints_gt = np.load(
        os.path.join(keypoints_dir, image_fname.split('.')[0]+'.npy'),
        allow_pickle=True,
        fix_imports=True
    )

    keypoints_pred = np.load(
        os.path.join(keypoints_pred_dir, image_fname.split('.')[0]+'.npy'),
        allow_pickle=True,
        fix_imports=True
    )

    plt.title(image_fname)
    plt.imshow(image, cmap='gray')
    plt.plot(keypoints_gt[0:74:2], keypoints_gt[1:75:2], 'bo')
    plt.plot(keypoints_pred[0:74:2], keypoints_pred[1:75:2], 'ro')
    plt.show()
