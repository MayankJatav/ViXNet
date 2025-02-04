## ViXNet: Vision Transformer with Xception Network for deepfake detection

This repository is an unofficial implementiation for the ViXNet Research Paper. The model architecture is a two channel architecture. It takes the input as the deepfake image and the same image is passed to both the channels. The model tries to classify the image as a deepfake image or not.<br>
https://www.sciencedirect.com/science/article/pii/S0957417422015251

The model consists of four modules:<br>
&nbsp;&nbsp;&nbsp;&nbsp;A. Patch-wise self-attention module<br>
&nbsp;&nbsp;&nbsp;&nbsp;B. Global-self attention module<br>
&nbsp;&nbsp;&nbsp;&nbsp;C. Global image feature extraction module<br>
&nbsp;&nbsp;&nbsp;&nbsp;D. Classification module<br>

![image](https://github.com/user-attachments/assets/1b860170-ed32-4daf-bae9-3a147c644f4f)


#### A. Patch-wise self-attention module

This module is a part of first branch. The image is first passed through this module before passing through the Global-self attention module. Here we convert the image into patches and apply a convolution operation on the patches and then element-wise multiplication is performed on the corresponding original patches and the convolved patches.

#### B. Global-self attention module

This module is also a part of the first branch. It receives input from the Patch-wise self-attention module which is in the form of patches. These patches are passed through VIT-B_16 Vision Transformer which captures the relation between the different patches of the image.

#### C. Global image feature extraction module

This module is a part of the second branch. It receives input as the original deepfake image. It contains a Xception model that extracts the global features from the input image. It is a modification of the Inception model and Xception her stands for Extreme Inception.

#### D. Classification Module

This module takes the input from both the branches and tries to predict the whether input image is a deepfake image or not. The output of both the branches is stacked together and then passed as the input to this module. Here we have fully connected neural network of layers of nodes 512, 256 and 128 respectively.
