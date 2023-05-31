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



# Function to generate heatmaps
def get_pdf(im, kpts, sigma):
    w, h, channels = im.shape
    
    x = np.linspace(0, h-1, h*1)
    y = np.linspace(0, w-1, w*1)
    [XX, YY] =  np.meshgrid(y,x)
    sze = XX.shape[0] * XX.shape[1]
    mvg = np.zeros((sze));    
    std = sigma
    p = 2
    count=0
    for i in range(0,37):
        mu = np.array([kpts[i][1], kpts[i][0]]).reshape((2,1)) 
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
def convert_to_keypoints_tupple(data):
    
    points = np.array(data)
    
    tupple_keypoints = []
    tupple_aux = [] 
    x = []
    y = []
    
    for i in range(np.shape(points)[0]): 
        for j in range(74): 
            if (j % 2 == 0): 
                x.append(int(points[i][j]))
            else:
                y.append(int(points[i][j]))    
        for z in range(37):
            tupple_aux.append((int(x[z]),int(y[z])))
        tupple_keypoints.append(tupple_aux)
        x = [] 
        y = [] 
        tupple_aux = [] 
    
    for i in range(np.shape(points)[0]):
        tupple_keypoints.append(points[i])
    
    keypoints = np.array(tupple_keypoints)

    return keypoints


# Heatmaps
def heatmap_generation(X, keypoints):

    density_map = []
    
    for i in range(np.shape(X)[0]): 
        oriImg = X[i]
        mapa = get_pdf(oriImg,keypoints[i],400)
        density_map.append(mapa)
        
    density_map = np.array(density_map)
    
    return density_map


""" Data """
with open("processed_files/X_test_221.pickle",'rb') as fp: 
    X = cPickle.load(fp)

with open("processed_files/y_test_221.pickle",'rb') as fp: 
    data = cPickle.load(fp)
    
keypoints = tupple(data)    

density_map = heatmap_generation(X,keypoints)

#print(np.shape(density_map))

#for i in range(X.shape[0]): 
#    plt.imshow(X[i])
#    plt.imshow(density_map[i], interpolation='nearest', alpha=0.6)
#    plt.show()


with open("processed_files/heatmaps_test_221.pickle", "wb") as output_file:
    cPickle.dump(density_map,output_file)







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
                'scale x',
                'scale y',
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


    # Method: Show images and keypoints
    def show_images_and_keypoints(self):

        # Get keypoints data
        keypoints_data = self.get_keypoints_bcct_data_df()

        # Convert into an array
        keypoints_data = keypoints_data.values
        print('SHOW IMAGE AND KEYPOINTS!!')
        # Iterate through this array
        for sample in keypoints_data:
            
            # File name
            filename = sample[0]

            # Keypoints
            keypoints = sample[1::]
            image_path = os.path.join(self.images_dir, 'anterior',filename)

            if not os.path.exists(image_path):
                base, ext = os.path.splitext(filename)
                image_path = os.path.join(self.images_dir, 'anterior',base + ext.upper())
                if not os.path.exists(image_path):
                    print(f"Could not find image file {filename}")
                    continue
            # Open image
            image = Image.open(image_path).convert('RGB')
            image = np.asarray(image)

            # Plot this
            plt.title(filename)
            plt.imshow(image, cmap='gray')
            plt.plot(keypoints[0:len(keypoints):2], keypoints[1:len(keypoints)+1:2], 'bo')
            plt.show()
        
        return
    

    # Method: __getitem__
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        pass


    










# Example usage
if __name__ == "__main__":

    # Global variables
    SHOW_IMGS_KPTS = False

    # Load an instance of the PICTUREBCCTData class
    picture_db = PICTUREBCCTData()


    # Get the BCCT data DataFrame
    bcct_data_df = picture_db.get_bcct_data_df()
    print('BCCT data DataFrame')
    print(bcct_data_df.head())


    # Get the BCCT data DataFrame w/ keypoints information
    keypoints_bcct_data_df = picture_db.get_keypoints_bcct_data_df()
    print('BCCT data DataFrame w/ keypoints information')
    print(keypoints_bcct_data_df.head())


    # Get the BCCT data DataFrame w/ features and classification information
    features_bcct_data_df = picture_db.get_features_bcct_data_df()
    print('BCCT data DataFrame w/ features and classification information')
    print(features_bcct_data_df.head())
    
    
    # Get the 2D features data file
    _2dfeatures_data_df = picture_db.get_2dfeatures_data_df()
    print('2D features data file')
    print(_2dfeatures_data_df.head())


    # Get the 3D features data file
    _3dfeatures_data_df = picture_db.get_3dfeatures_data_df()
    print('3D features data file')
    print(_3dfeatures_data_df.head())


    # Get the Subjective Evaluation 2D data file
    subjeval2d_data_df = picture_db.get_subjeval2d_data_df()
    print('Subjective Evaluation 3D data file')
    print(subjeval2d_data_df.head())

    # Get the Subjective Evaluation 3D data file
    subjeval3d_data_df = picture_db.get_subjeval3d_data_df()
    print('Subjective Evaluation 3D data file')
    print(subjeval3d_data_df.head())


    # Get the patient codes data file
    patient_codes_data_df = picture_db.get_patient_codes_data_df()
    print('Patient codes data file')
    print(patient_codes_data_df.head())


    # Show images and keypoints
    if SHOW_IMGS_KPTS:
        picture_db.show_images_and_keypoints()
