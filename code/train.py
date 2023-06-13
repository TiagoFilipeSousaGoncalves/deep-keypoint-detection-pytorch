# Imports
import argparse
import os
import datetime
import numpy as np
import albumentations as A
import albumentations.pytorch as A_torch
from tqdm import tqdm

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Project Imports
from data_utilities import PICTUREBCCTKDetectionDataset
from model_utilities import DeepKeypointDetectionModel

# Weights and Biases (W&B) Imports
import wandb

# Log in to W&B Account
wandb.login()



# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)



# CLI Arguments
parser = argparse.ArgumentParser()


# Database(s)
parser.add_argument('--database', type=str, choices=['picture-db'], help="Database(s) to use (i.e., picture-db)")

# Epochs
parser.add_argument('--epochs', type=int, default=300, help="Number of epochs to train the model.")

# Batch size
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for dataloader.")

# Number of Workers
parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for dataloader.")

# Device
parser.add_argument('--gpu_id', type=int, default=1, help="The ID of the GPU.")

# Results directory
parser.add_argument('--results_directory', type=str, default='results', help="Results directory")

# Parse the arguments
args = parser.parse_args()
DATABASE = args.database
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = args.results_directory


# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(RESULTS_DIR, timestamp)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)



# Set the W&B project
wandb.init(
    project="cinderella-deep-keypoint-detection", 
    name=timestamp,
    config={
        "database": DATABASE,
        "epochs": EPOCHS,
    }
)



# Build dataset and transforms
if DATABASE == "picture-db":
    transform = A.Compose(
        [
            A.Affine(translate_px={'x':(-50, 50), 'y':(-30, 30)}, rotate=(-10, 10), p=0.1),
            A.HorizontalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A_torch.ToTensorV2(),
            A.Normalize()
        ],
        keypoint_params=A.KeypointParams(format='xy')
    )

    dataset = PICTUREBCCTKDetectionDataset(
        images_dir='/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/images/',
        heatmaps_dir='/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/heatmaps/',
        keypoints_dir='/nas-ctm01/datasets/private/CINDERELLA/processed-databases/deep-keypoint-detection/picture-db/keypoints',
        transform=transform
    )

    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)



# Model & Hyperparameters
model = DeepKeypointDetectionModel()
model.to(DEVICE)

# Watch model using W&B
wandb.watch(model)

# Optimiser
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss functions
loss_stage1 = torch.nn.MSELoss(reduction='mean')
loss_stage2 = torch.nn.MSELoss(reduction='mean')
loss_stage3 = torch.nn.MSELoss(reduction='mean')
loss_stage4 = torch.nn.MSELoss(reduction='mean')


# Train model
best_loss = -np.inf
best_keypoints_loss = -np.inf

for epoch in range(EPOCHS):

    # Running avg loss
    running_avg_loss = 0.

    # Running avg stages losses
    running_avg_stg1_loss = 0.
    running_avg_stg2_loss = 0.
    running_avg_stg3_loss = 0.

    # Running keypoints loss
    running_avg_kpts_loss = 0.

    # Training loop
    for image, keypoints, heatmap in tqdm(dataloader):

        image, keypoints, heatmap = image.to(DEVICE), keypoints.to(DEVICE), heatmap.to(DEVICE)

        # Clear batch gradients
        optimiser.zero_grad()

        # Perform inference
        stage1, stage2, stage3, stage4 = model(image)

        # Compute individual losses (per stage)
        loss1 = loss_stage1(stage1, heatmap)
        loss2 = loss_stage2(stage2, heatmap)
        loss3 = loss_stage3(stage3, heatmap)
        loss4 = loss_stage4(stage4, keypoints)

        # Final loss is give by the weighted sum of these losses
        loss = 1*loss1 + 2*loss2 + 4*loss3 + 10*loss4
        
        # Perform backpropagation
        loss.backward()

        # Updated model parameters
        optimiser.step()

        # Update running avg loss
        running_avg_loss += loss.item()

        # Update running avg stages losses
        running_avg_stg1_loss += loss1.item()
        running_avg_stg2_loss += loss2.item()
        running_avg_stg3_loss += loss3.item()

        # Updated running avg keypoints loss
        running_avg_kpts_loss += loss4.item()


    # Get final average losses (by dividing by the number of batches)
    running_avg_loss /= len(dataloader)
    running_avg_stg1_loss /= len(dataloader)
    running_avg_stg2_loss /= len(dataloader)
    running_avg_stg3_loss /= len(dataloader)
    running_avg_kpts_loss /= len(dataloader)

    # Log to W&B
    wandb_tr_metrics = {
        "loss/stage1":running_avg_stg1_loss,
        "loss/stage2":running_avg_stg2_loss,
        "loss/stage3":running_avg_stg3_loss,
        "loss/keypoints":running_avg_kpts_loss,
        "loss/loss":running_avg_loss,
    }
    wandb.log(wandb_tr_metrics)

    # Update best losses
    if running_avg_loss < best_loss:
        print(f"Loss decreased from {best_loss} to {running_avg_loss}.")
        best_loss = running_avg_loss

    if running_avg_kpts_loss < best_keypoints_loss:
        print(f"Keypoints loss decreased from {best_keypoints_loss} to {running_avg_kpts_loss}.")

        model_path = os.path.join(results_dir, 'best_model.pt')
        print(f"Saving model into: {model_path}")
        
        # Save model
        save_dict = {'model_state_dict': model.state_dict()}
        torch.save(save_dict, model_path)
