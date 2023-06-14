# Imports
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Project Imports
from data_utilities import convert_to_keypoints_tupple, generate_heatmap 



# Create CLI
parser = argparse.ArgumentParser()

# Database
parser.add_argument('--database', type=str, required=True, choices=["picture-db", "original_files"], help="Database(s): picture-db, original_files.")

# Parse the arguments
args = parser.parse_args()



# Select database
if args.database == "picture-db":

    # Read directories
    images_dir = "/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/images/anterior"
    keypoints_dir = "/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/keypoints"
    heatmaps_dir = "/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/heatmaps"


    # Create heatmaps directory
    if not os.path.isdir(heatmaps_dir):
        os.makedirs(heatmaps_dir)


    # Load files
    images_list = [i for i in os.listdir(images_dir) if not i.startswith('.')]

    for image_fname in tqdm(images_list):

        # Open image
        image_fpath = os.path.join(images_dir, image_fname)
        image = Image.open(image_fpath).convert('RGB')
        image = np.array(image)

        # Keypoints
        keypoints = np.load(
            file=os.path.join(keypoints_dir, image_fname.split('.')[0]+'.npy'),
            allow_pickle=True,
            fix_imports=True
        )

        # Convert to tupple
        keypoints_tupple = convert_to_keypoints_tupple(keypoints_data=keypoints)

        # Generate heatmap
        heatmap = generate_heatmap(image=image, keypoints_tupple=keypoints_tupple)

        # Save heatmap
        np.save(
            file=os.path.join(heatmaps_dir, image_fname.split('.')[0]+'.npy'),
            arr=heatmap,
            allow_pickle=True
        )


# Select database
elif args.database == "original_files":

    # Read directories
    images_dir = "/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/original_files/images"
    keypoints_dir = "/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/original_files/keypoints"
    heatmaps_dir = "/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/original_files/heatmaps"


    # Create heatmaps directory
    if not os.path.isdir(heatmaps_dir):
        os.makedirs(heatmaps_dir)


    # Load files
    images_list = [i for i in os.listdir(images_dir) if not i.startswith('.')]

    for image_fname in tqdm(images_list):

        # Open image
        image_fpath = os.path.join(images_dir, image_fname)
        image = Image.open(image_fpath).convert('RGB')
        image = np.array(image)

        # Keypoints
        keypoints = np.load(
            file=os.path.join(keypoints_dir, image_fname.split('.')[0]+'.npy'),
            allow_pickle=True,
            fix_imports=True
        )

        # Convert to tupple
        keypoints_tupple = convert_to_keypoints_tupple(keypoints_data=keypoints)

        # Generate heatmap
        heatmap = generate_heatmap(image=image, keypoints_tupple=keypoints_tupple)

        # Save heatmap
        np.save(
            file=os.path.join(heatmaps_dir, image_fname.split('.')[0]+'.npy'),
            arr=heatmap,
            allow_pickle=True
        )

print("Finished.")
