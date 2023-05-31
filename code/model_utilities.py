# Imports
from collections import OrderedDict

# PyTorch Imports
import torch
import torch.nn as nn
import torchvision



# Class: UNet (source: https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )



# Class: VGG16FeatureExtractor
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, channels=3, height=512, width=512):
        super(VGG16FeatureExtractor, self).__init__()

        # Create random vector to compute dimensions
        input_ = torch.rand((1, channels, height, width))

        # Initialise VGG16 using pre-trained weights
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16_features = vgg16.features
        
        # conv1 = conv_base(inputs)
        self.conv_base = vgg16_features


        # Compute the input channels for the first conv layer
        in_channels_ = self.conv_base(input_)
        in_channels_ = in_channels_.size(0) * in_channels_.size(1)

        # conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        self.conv1 = nn.Conv2d(in_channels=in_channels_, out_channels=512, kernel_size=(3, 3), padding='valid')
        self.relu1 = nn.ReLU()
        
        # conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='valid')
        self.relu2 = nn.ReLU()
        
        # conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='valid')
        self.relu3 = nn.ReLU()
        
        # conv1 = Conv2D(512, (1, 1), activation='relu', padding='valid')(conv1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), padding='valid')
        self.relu4 = nn.ReLU()


        # Compute input features for the first linear layer
        in_features_ = self.conv_base(input_)
        in_features_ = self.conv1(in_features_)
        in_features_ = self.conv2(in_features_)
        in_features_ = self.conv3(in_features_)
        in_features_ = self.conv4(in_features_)
        in_features_ = in_features_.size(0) * in_features_.size(1) * in_features_.size(2) * in_features_.size(3)
        
        # dense1 = Dense(256,activation='relu')(flat)
        self.dense1 = nn.Linear(in_features=in_features_, out_features=256)
        self.relu5 = nn.ReLU()

        # dense1 = Dropout(0.2)(dense1)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # dense1 = Dense(128,activation='relu')(dense1)
        self.dense2 = nn.Linear(in_features=256, out_features=128)
        self.relu6 = nn.ReLU()
        
        # reg = Dense(74,activation='sigmoid', name = 'keypoints')(dense1)
        self.dense3 = nn.Linear(in_features=128, out_features=74)
        self.sigmoid = nn.Sigmoid()
    
        return

    def forward(self, x):

        # conv1 = conv_base(inputs)
        conv1 = self.conv_base(x)

        # conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        conv1 = self.relu1(self.conv1(conv1))

        # conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        conv1 = self.relu2(self.conv2(conv1))

        # conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        conv1 = self.relu3(self.conv3(conv1))

        # conv1 = Conv2D(512, (1, 1), activation='relu', padding='valid')(conv1)
        conv1 = self.relu4(self.conv4(conv1))
        
        
        # flat = Flatten()(conv1)
        flat = torch.reshape(conv1, (conv1.size(0), -1))
        
        
        # dense1 = Dense(256,activation='relu')(flat)
        dense1 = self.relu5(self.dense1(flat))
        
        # dense1 = Dropout(0.2)(dense1)
        dense1 = self.dropout1(dense1)

        # dense1 = Dense(128,activation='relu')(dense1)
        dense1 = self.relu6(self.dense2(dense1))
        
        # reg = Dense(74,activation='sigmoid', name = 'keypoints')(dense1)
        reg = self.sigmoid(self.dense3(dense1))


        return reg




# Class: Deep Keypoint Detection Model (Silva et al.) ISBI 2019
class DeepKeypointDetectionModel(nn.Module):

    def __init__(self):
        super(DeepKeypointDetectionModel, self).__init__()

        # UNet Stage 1
        self.unet1 = UNet()

        # UNet Stage 2
        self.unet2 = UNet()

        # UNet Stage 3
        self.unet3 = UNet()

        # Feature Extraction and Keypointe Regression
        self.vgg16_features = VGG16FeatureExtractor()


        return

    def forward(self, x):
        
        # First step is to obtain probability maps
        stage1 = self.unet1(x)
        
        # Concatenate probability maps in order to have an image with the same number of channels as input image
        stage1_concat = torch.cat((stage1, stage1, stage1), dim=1)
        
        # Multiplication between prob maps and input image, to select region of interest
        stage2_in = torch.mul(stage1_concat, x)
        stage2 = self.unet2(stage2_in)
        stage2_concat = torch.cat((stage2, stage2, stage2), dim=1)

        stage3_in = torch.mul(stage2_concat, x)
        stage3 = self.unet3(stage3_in)
        stage3_concat = torch.cat((stage3, stage3, stage3), dim=1)
        
        stage4_in = torch.mul(stage3_concat, x)
        
        # Perform regression
        stage4 = self.vgg16_features(stage4_in)

        return stage1, stage2, stage3, stage4



# Example usage
if __name__ == "__main__":

    # Input
    inputs = torch.rand((1, 3, 512, 512))
    print(f"Input size: {inputs.size()}")
    
    # UNet
    unet = UNet()
    unet_out = unet(inputs)
    print(f"UNet output size: {unet_out.size()}")

    # VGG16F
    vgg16 = VGG16FeatureExtractor()
    vgg16_out = vgg16(inputs)
    print(f"VGG16 Feature Extractor output size: {vgg16_out.size()}")


    # Final Model
    deep_kdetection = DeepKeypointDetectionModel()
    deep_kdetection_out1, deep_kdetection_out2, deep_kdetection_out3, deep_kdetection_out4 = deep_kdetection(inputs)
    print(f"Deep Keypoint Detection Model outputs: {deep_kdetection_out1.size()}, {deep_kdetection_out2.size()}, {deep_kdetection_out3.size()}, {deep_kdetection_out4.size()}")
