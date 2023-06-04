# Imports
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



# Create the parser for the CLI
parser = argparse.ArgumentParser()


# Images path
parser.add_argument("--images_path", type=str, help="Directory of the images of the data set.")

# Keypoints path
parser.add_argument("--keypoints_path", type=str, help="Directory of the keypoints of the data set.")

# Heatmaps path
parser.add_argument("--heatmaps_path", type=str, help="Directory of the heatmaps data set.")


# Parse the arguments
args = parser.parse_args()



# Images list
images_list = [i for i in os.listdir(args.images_path) if not i.startswith('.')]


for image_fname in images_list:
    image_fpath = os.path.join(args.images_path, image_fname)
    image = Image.open(image_fpath).convert('RGB')
    image = np.array(image)

    keypoints = np.load(
        file=os.path.join(args.keypoints_path, image_fname.split('.')[0]+'.npy'),
        allow_pickle=True,
        fix_imports=True
    )

    heatmap = np.load(
        file=os.path.join(args.heatmaps_path, image_fname.split('.')[0]+'.npy'),
        allow_pickle=True,
        fix_imports=True
    )


    plt.imshow(image, cmap='gray')
    plt.imshow(heatmap, alpha=0.5)
    plt.show()

    plt.imshow(image, cmap='gray')
    plt.plot(keypoints[0:74:2], keypoints[1:75:2], 'o')
    plt.show()
