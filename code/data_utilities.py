# Imports
import os
import numpy as np 
import pandas as pd
import _pickle as cPickle
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import scipy.misc

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# Function: Resize images and keypoints
def resize_images_keypoints(image, keypoints_array, new_width=512, new_height=512):

    # Get image shape
    rows, columns, _ = np.shape(image)

    # Get ratios
    x1 = rows / new_height
    x2 = columns / new_width

    # Get new images
    resized_image = np.array(image.copy())
    resized_image = cv2.resize(resized_image, (np.ceil(rows / x1), np.ceil(columns / x2)), interpolation=cv2.INTER_AREA)

    # Get new keypoints
    resized_keypoints = keypoints_array.copy()

    for j in range(len(resized_keypoints)):
        if(j % 2 == 0):
            resized_keypoints[j] /= x2
        else: 
            resized_keypoints[j] /= x1


    return resized_image, resized_keypoints



# Function: Generate heatmaps from image and keypoints tupple
def generate_heatmap(image, keypoints_tupple, sigma=400):
    w, h, _ = image.shape
    
    x = np.linspace(0, h-1, h*1)
    y = np.linspace(0, w-1, w*1)
    [XX, YY] =  np.meshgrid(y,x)
    sze = XX.shape[0] * XX.shape[1]
    mvg = np.zeros((sze));    
    std = sigma
    p = 2
    count=0
    for i in range(0,37):
        mu = np.array([keypoints_tupple[i][1], keypoints_tupple[i][0]]).reshape((2,1)) 
        mu = np.tile(mu, (1, sze))
        mcov = np.identity(2) * std
        
        X = np.array([np.ravel(XX.T), np.ravel(YY.T)])
        
        temp0 = 1 / ( math.pow(2*math.pi, p/2) * \
                    math.pow(np.linalg.det(mcov), 0.5) )
        
        temp1 = -0.5*(X-mu).T
        temp2 = np.linalg.inv(mcov).dot(X-mu) 
        
        temp3 = temp0 * np.exp(np.sum(temp1 * temp2.T, axis=1))
        maximum = max(temp3.ravel())
        
        mvg = mvg + temp3
        count += 1
    
        mvg[mvg>maximum] = maximum
        
    mvg = mvg.reshape((XX.shape[1], XX.shape[0]))
    
    mvg = ( mvg - min(mvg.ravel()) ) / ( max(mvg.ravel()) - min(mvg.ravel()) )
    
    mvg = mvg * 255.0
    mvg = cv2.resize(mvg, (h, w), interpolation = cv2.INTER_CUBIC)
    mvg = mvg / 255.0
    mvg[mvg<0] = 0
   
    return mvg



# Function: Convert keypoints to tupple of keypoints
def convert_to_keypoints_tupple(keypoints_data):

    # Ensure that data is from the NumPy array type
    keypoints_arr = np.array(keypoints_data)

    # Create temporary lists
    keypoints_tupple = []
    x = []
    y = []

    # Get xx and yy coordinates
    for j in range(len(keypoints_arr)): 
        if (j % 2 == 0): 
            x.append(int(keypoints_arr[j]))
        else:
            y.append(int(keypoints_arr[j]))    


    # Convert this into tupples of xy 
    for z in range(int(len(keypoints_arr)/2)):
        keypoints_tupple.append((int(x[z]), int(y[z])))


    # Convert keypoints tupple to NumPy array
    keypoints_tupple = np.array(keypoints_tupple)


    return keypoints_tupple



# Class: PICTUREBCCTData
class PICTUREBCCTDataset(Dataset):

    def __init__(self, images_dir, heatmaps_dir, metadata_dir, transform=None):

        # Class variables
        self.images_dir = images_dir
        self.heatmaps_dir = heatmaps_dir
        self.metadata_dir = metadata_dir
        self.bcct_data = None
        self.keypoints_bcct_data = None

        # Load BCCT data
        self.get_bcct_data_df()

        # Load Keypoints Data
        self.get_keypoints_bcct_data_df()

        # Get images, keypoints and heatmaps
        images = list()
        keypoints = list()
        heatmaps = list()
        
        # Convert to array
        keypoints_bcct_data = self.keypoints_bcct_data.values

        for sample in keypoints_bcct_data:
        
            # Image filename
            image_fname = sample[0]

            # Keypoints
            image_keypoints = sample[1::]

            # Heatmap filename
            heatmap_fname = image_fname.split('.')[0]+'.npy'


            # FIXME: We have to fix the annotation of this image
            if image_fname.lower() != '040a.jpg'.lower():
                images.append(image_fname)
                keypoints.append(image_keypoints)
                heatmaps.append(heatmap_fname)


        # Assign variables
        self.images = images
        self.keypoints = keypoints
        self.heatmaps = heatmaps
        self.transform = transform

        return


    # Method: Get the BCCT data DataFrame
    def get_bcct_data_df(self):
        
        # Return this, if available
        if self.bcct_data is not None:
            bcct_data_df = self.bcct_data.copy()
        
        # Read .CSV with BCCT data related to PICTURE
        else:
            bcct_data_df = pd.read_csv(os.path.join(self.metadata_dir, 'bcct_data.csv'), sep=',')
            self.bcct_data = bcct_data_df.copy()
        
        return bcct_data_df
    

    # Method: Get the BCCT data DataFrame w/ keypoints information
    def get_keypoints_bcct_data_df(self):

        # Return this information, if available
        if self.keypoints_bcct_data is not None:
            keypoints_bcct_data = self.keypoints_bcct_data.copy()

        else:
            keypoints_bcct_data = self.bcct_data.copy()[
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

            # Add this to the corresponding class variable
            self.keypoints_bcct_data = keypoints_bcct_data.copy()


        return keypoints_bcct_data
    
    
    # Method: __len__
    def __len__(self):
        return len(self.images)


    # Method: __getitem__
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename
        image_fname = self.images[idx]
        image_fpath = os.path.join(self.images_dir, 'anterior', image_fname)
        if not os.path.exists(image_fpath):
            image_fpath = os.path.join(self.images_dir, 'anterior', image_fname.split('.')+'.JPG')
        
        # Load image
        image = Image.open(image_fpath).convert('RGB')


        # Get keypoints
        keypoints = self.keypoints[idx]

        # Get heatmap
        heatmap = self.heatmaps[idx]

        # Apply transforms
        if self.transform:
            pass
        
        return image, keypoints, heatmap



# Example usage
if __name__ == "__main__":

    # PICTUREBCCTDataset
    picture_dataset = PICTUREBCCTDataset(
        images_dir="/nas-ctm01/datasets/private/CINDERELLA/picture-db/images",
        heatmaps_dir=f"/nas-ctm01/homes/tgoncalv/deep-keypoint-detection-pytorch/data/picture-db/heatmaps",
        metadata_dir="/nas-ctm01/datasets/private/CINDERELLA/picture-db/metadata"
    )
