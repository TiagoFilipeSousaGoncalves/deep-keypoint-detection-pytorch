# Imports
import argparse
import os
import numpy as np
import albumentations as A
import albumentations.pytorch as A_torch
from PIL import Image
from tqdm import tqdm

# PyTorch Imports
import torch

# Project Imports
from model_utilities import DeepKeypointDetectionModel



# CLI Arguments
parser = argparse.ArgumentParser()


# Database(s)
parser.add_argument('--database', type=str, choices=['picture-db'], help="Database(s) to use (i.e., picture-db)")

# Device
parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU.")

# Model directory
parser.add_argument('--model_directory', type=str, required=True, help="Model directory")

# Parse the arguments
args = parser.parse_args()
DATABASE = args.database
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
MODEL_DIR = args.results_directory



if DATABASE == 'picture-db':

    # Ground-truth data directories
    images_dir = '/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/images/anterior'
    keypoints_dir = '/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/keypoints'

    # Predictions directories
    predictions_dir = '/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/predictions/keypoints'

    if not os.path.isdir(predictions_dir):
        os.makedirs(predictions_dir)



# Transforms
transform = A.Compose([A.Normalize(), A_torch.ToTensorV2()])


# Load model
model = DeepKeypointDetectionModel()
model_weights = torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), map_location=DEVICE)
model.load_state_dict(model_weights['model_state_dict'], strict=True)


# Read images
images_fnames_list = [i for i in os.listdir(images_dir) if not i.startswith('.')]

for image_fname in tqdm(images_fnames_list):

    # Load image
    image = Image.open(os.path.join(images_dir, image_fname)).convert('RGB')
    image = np.array(image)
    image = transform(image=image)['image']

    # Get predictions
    _, _, _, keypoints_prediction = model(image)

    # Convert predictions into NumPy
    keypoints_prediction = keypoints_prediction.detach().cpu().numpy()
    keypoints_prediction = keypoints_prediction.flatten()
    keypoints_prediction *= 512

    # Create keypoints fname
    keypoints_prediction_fname = image_fname.split('.')[0] + '.npy'

    # Save into storage
    np.save(
        file=os.path.join(predictions_dir, keypoints_prediction_fname),
        arr=keypoints_prediction,
        allow_pickle=True,
        fix_imports=True
    )
