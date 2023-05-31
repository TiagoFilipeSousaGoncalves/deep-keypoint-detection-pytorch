# Imports
import os
import argparse
import pandas as pd

# Project Imports
from data_utilities import convert_to_keypoints_tupple, heatmap_generation 



# Create CLI
parser = argparse.ArgumentParser()

# Database
parser.add_argument('--database', type=str, required=True, choices=["picture-db"], help="Database(s): picture-db.")

# Parse the arguments
args = parser.parse_args()



# Select database
if args.database == "picture-db":

    # Read directories
    images_dir = "/nas-ctm01/datasets/private/CINDERELLA/picture-db/images"
    metadata_dir = "/nas-ctm01/datasets/private/CINDERELLA/picture-db/metadata"
    heatmaps_dir = f"/nas-ctm01/homes/tgoncalv/deep-keypoint-detection-pytorch/data/{args.database}/heatmaps"

    # Load files
    bcct_data = pd.read_csv(os.path.join(metadata_dir, 'bcct_data.csv'), sep=',')
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

    for sample in keypoints_bcct_data:
        
        # Filename
        filename = sample[0]

        # Keypoints
        keypoints = sample[1::]


        # Convert to tupple
        keypoints_tupple = convert_to_keypoints_tupple()