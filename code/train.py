# Imports
import argparse
import os
import cv2
import albumentations as A

# PyTorch Imports


# Project Imports
from data_utilities import PICTUREBCCTKDetectionDataset
from model_utilities import DeepKeypointDetectionModel


# CLI Arguments


DATABASE = args.database


# Build dataset and transforms
if DATABASE == "picture-db":
    train_transform = A.Compose(
        [
            A.Affine(translate_px={'x':(-50, 50), 'y':(-30, 30)}, rotate=(-10, 10), p=0.1),
            A.HorizontalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.1),
        ],
        keypoint_params=A.KeypointParams(format='xy')
    )