# Imports
import os
import argparse
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import _pickle as Cpickle

# Project Import
from data_utilities import resize_images_keypoints



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Database
parser.add_argument('--database', type=str, required=True, choices=["picture-db", "original_files"], help="Database(s): picture-db.")

# Original directory
parser.add_argument("--original_path", type=str, help="Directory of the original data set.")

# New (resized) directory
parser.add_argument("--new_path", type=str, help="Directory of the resized data set.")

# New width
parser.add_argument("--new_width", type=int, default=512, help="New width of the images (default=512).")

# New height
parser.add_argument("--new_height", type=int, default=512, help="New height of the images (default=512).")

# Parse the arguments
args = parser.parse_args()


# Get the arguments
ORIGINAL_PATH = args.original_path
NEW_PATH = args.new_path
NEW_WIDTH = args.new_width
NEW_HEIGHT = args.new_height


# Create new path if needed
if not os.path.exists(NEW_PATH):
    os.makedirs(NEW_PATH)


if args.database == 'picture-db':

    # Original directories
    or_images_dir = os.path.join(ORIGINAL_PATH, "images", "anterior")
    or_metadata_dir = os.path.join(ORIGINAL_PATH, "metadata")
    
    
    # New directories
    n_images_dir = os.path.join(NEW_PATH, args.database.lower(), "images", "anterior")
    n_keypoints_dir = os.path.join(NEW_PATH, args.database.lower(), "keypoints")
    if not os.path.isdir(n_images_dir):
        os.makedirs(n_images_dir)
    if not os.path.isdir(n_keypoints_dir):
        os.makedirs(n_keypoints_dir)


    # Load files
    bcct_data = pd.read_csv(os.path.join(or_metadata_dir, 'bcct_data.csv'), sep=',')
    keypoints_bcct_data = bcct_data.copy()[
                [
                'file path',
                'left contour x1',
                'left contour y1',
                'left contour x2',
                'left contour y2',
                'left contour x3',
                'left contour y3',
                'left contour x4',
                'left contour y4',
                'left contour x5',
                'left contour y5',
                'left contour x6',
                'left contour y6',
                'left contour x7',
                'left contour y7',
                'left contour x8',
                'left contour y8',
                'left contour x9',
                'left contour y9',
                'left contour x10',
                'left contour y10',
                'left contour x11',
                'left contour y11',
                'left contour x12',
                'left contour y12',
                'left contour x13',
                'left contour y13',
                'left contour x14',
                'left contour y14',
                'left contour x15',
                'left contour y15',
                'left contour x16',
                'left contour y16',
                'left contour x17',
                'left contour y17',
                'right contour x1',
                'right contour y1',
                'right contour x2',
                'right contour y2',
                'right contour x3',
                'right contour y3',
                'right contour x4',
                'right contour y4',
                'right contour x5',
                'right contour y5',
                'right contour x6',
                'right contour y6',
                'right contour x7',
                'right contour y7',
                'right contour x8',
                'right contour y8',
                'right contour x9',
                'right contour y9',
                'right contour x10',
                'right contour y10',
                'right contour x11',
                'right contour y11',
                'right contour x12',
                'right contour y12',
                'right contour x13',
                'right contour y13',
                'right contour x14',
                'right contour y14',
                'right contour x15',
                'right contour y15',
                'right contour x16',
                'right contour y16',
                'right contour x17',
                'right contour y17',
                'sternal notch x',
                'sternal notch y',
                'left nipple x',
                'left nipple y',
                'right nipple x',
                'right nipple y'
                ]
            ]


    # Go through these data
    keypoints_bcct_data = keypoints_bcct_data.values

    for sample in tqdm(keypoints_bcct_data):
        
        # Filename
        filename = sample[0]

        # Open image
        image_fpath = os.path.join(or_images_dir, filename)
        if not os.path.exists(image_fpath):
            image_fpath = os.path.join(or_images_dir, filename.split('.')[0]+'.JPG')
        
        # Read image
        image = cv2.imread(os.path.join(image_fpath))

        # Keypoints
        keypoints = sample[1::]


        # Resize image and keypoints
        resized_image, resized_keypoints = resize_images_keypoints(
            image=image,
            keypoints_array=keypoints,
            new_width=NEW_WIDTH,
            new_height=NEW_HEIGHT
        )


        # Save new image
        cv2.imwrite(os.path.join(n_images_dir, filename), resized_image)


        # Save new keypoints
        np.save(
            file=os.path.join(n_keypoints_dir, filename.split('.')[0]+'.npy'),
            arr=resized_keypoints,
            allow_pickle=True
        )

elif args.database == 'original_files':
        
        # Original directories
        or_images_dir = os.path.join(ORIGINAL_PATH, "images")
        or_metadata_dir = os.path.join(ORIGINAL_PATH, "files")
    
    
        # New directories
        n_images_dir = os.path.join(NEW_PATH, args.database.lower(), "images")
        n_keypoints_dir = os.path.join(NEW_PATH, args.database.lower(), "keypoints")
        if not os.path.isdir(n_images_dir):
            os.makedirs(n_images_dir)
        if not os.path.isdir(n_keypoints_dir):
            os.makedirs(n_keypoints_dir)
        

        # Open pickle with the filenames and the keypoints
        with open(os.path.join(or_metadata_dir, 'filenames.pickle'), 'rb') as fp:
            filenames_list = Cpickle.load(fp)
        
        with open(os.path.join(or_metadata_dir, 'keypoints.pickle'), 'rb') as fp:
            keypoints_list = Cpickle.load(fp)


        for idx in tqdm(range(len(filenames_list))):
        
            # Filename
            filename = filenames_list[idx].split('/')[-1]
            # print(filename)

            # Open image
            image_fpath = os.path.join(or_images_dir, filename)
            
            # Read image
            image = cv2.imread(image_fpath)

            # Keypoints
            keypoints = keypoints_list[idx]


            # Resize image and keypoints
            resized_image, resized_keypoints = resize_images_keypoints(
                image=image,
                keypoints_array=keypoints,
                new_width=NEW_WIDTH,
                new_height=NEW_HEIGHT
            )


            # Save new image
            cv2.imwrite(os.path.join(n_images_dir, filename), resized_image)


            # Save new keypoints
            np.save(
                file=os.path.join(n_keypoints_dir, filename.split('.')[0]+'.npy'),
                arr=resized_keypoints,
                allow_pickle=True
            )
