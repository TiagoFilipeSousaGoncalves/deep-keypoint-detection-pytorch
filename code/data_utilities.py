# Imports
import os
import numpy as np 
import math
import cv2
from PIL import Image

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
    resized_image = cv2.resize(resized_image, (int(np.ceil(rows / x1)), int(np.ceil(columns / x2))), interpolation=cv2.INTER_AREA)

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



# Function: Convert keypoints to albumentations format
def convert_keypoints_to_albumentations(keypoints_array):

    # Convert keypoints to x, y notation    
    x = list()
    y = list()
    for i in range(len(keypoints_array)):
        if i % 2 == 0:
            x.append(keypoints_array[i])
        else:
            y.append(keypoints_array[i])
    

    # Generate list of keypoint tupples
    keypoints_list = list()
    for x_i, y_i in zip(x, y):
        keypoints_list.append((x_i, y_i))


    return keypoints_list



# Class: PICTUREBCCTKDetection
class PICTUREBCCTKDetectionDataset(Dataset):

    def __init__(self, images_dir, heatmaps_dir, keypoints_dir, transform=None):

        # Class variables
        self.images_dir = images_dir
        self.heatmaps_dir = heatmaps_dir
        self.keypoints_dir = keypoints_dir

        # Get images, keypoints and heatmaps (FIXME: We have to fix the annotation of image id '040a')
        images = [i for i in os.listdir(self.images_dir, 'anterior') if not i.startswith('.')]
        images = [i for i in images if i != '040a.jpg']
        
        keypoints = [k for k in os.listdir(self.keypoints_dir) if not k.startswith('.')]
        keypoints = [k for k in keypoints if k != '040a.npy']

        heatmaps = [h for h in os.listdir(self.heatmaps_dir) if not h.startswith('.')]
        heatmaps = [h for h in heatmaps if h != '040a.npy']

        # Assign variables
        self.images = images
        self.keypoints = keypoints
        self.heatmaps = heatmaps
        self.transform = transform

        return
    
    
    # Method: __len__
    def __len__(self):
        return len(self.images)


    # Method: __getitem__
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename and load image
        image_fname = self.images[idx]
        image_fpath = os.path.join(self.images_dir, 'anterior', image_fname)
        image = Image.open(image_fpath).convert('RGB')
        image = np.array(image)

        # Get keypoints filename and load keypoint
        keypoints_fname = self.keypoints[idx]
        keypoints = np.load(os.path.join(self.keypoints_dir, keypoints_fname), allow_pickle=True, fix_imports=True)
        keypoints = convert_keypoints_to_albumentations(keypoints)

        # Get heatmap filename and load heatmap
        heatmap_fname = self.heatmaps[idx]
        heatmap = np.load(os.path.join(self.heatmaps_dir, heatmap_fname), allow_pickle=True, fix_imports=True)

        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                mask=heatmap,
                keypoints=keypoints
            )

            image = transformed["image"]
            keypoints = transformed["keypoints"]
            heatmap = transformed["mask"]


        return image, keypoints, heatmap
