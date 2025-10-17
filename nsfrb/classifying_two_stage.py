import os
import argparse
import numpy as np
from PIL import Image
import torch
torch.set_default_device("cpu")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def resize_cube(cube, size=(50, 50)):
    """
    cube: (H, W, T, C) float or int
    size: (width, height) target, e.g. (50, 50)
    
    Returns resized_cube: (size[1], size[0], T, C), i.e. (50, 50, T, C)
    """
    H, W, T, C = cube.shape
    out_w, out_h = size 
    # Prepare empty array for resized result: (out_h, out_w, T, C)
    resized_cube = np.empty((out_h, out_w, T, C), dtype=np.float32)
    # Ensure the cube is float32 so PIL can handle it
    cube = cube.astype(np.float32, copy=False)

    for t in range(T):
        for c in range(C):
            # Extract 2D slice
            slice_2d = cube[:, :, t, c]
            # Create a PIL Image (mode='F' for 32-bit float)
            img = Image.fromarray(slice_2d, mode='F')
            # Resize with bilinear interpolation
            img = img.resize((out_w, out_h), resample=Image.BILINEAR)
            # Convert back to NumPy
            resized_cube[:, :, t, c] = np.array(img, dtype=np.float32)

    return resized_cube




class SpatialCNN(nn.Module):
    """
    Enhanced Convolutional Neural Network (CNN) for image classification.
    *** New implementation: 2D classify freq & time averaged image ***
    """

    def __init__(self):
        super(SpatialCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))  # 64x25x25
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))  # 128x12x12
        print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))  # 256x6x6
        print(x.shape)

        x = x.view(-1, 6 * 6 * 256)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.sigmoid(self.fc3(x))
        print(x.shape)
        return x

dd = 4
class SpectralCNN(nn.Module):
    """
    Enhanced Convolutional Neural Network (CNN) for image classification.
    *** New implemenation: 1D classify peak pixel dynamic spectrum **
    """

    def __init__(self):
        super(SpectralCNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*2*3,16)
        self.fc2 = nn.Linear(16,6)
        self.fc3 = nn.Linear(6,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))  # 64x25x25
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))  # 128x12x12
        print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))  # 256x6x6
        print(x.shape)

        x = x.view(-1, 16*2*3)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.sigmoid(self.fc3(x))
        print(x.shape)
        
        return x

class CombinedCNN(nn.Module):

    def __init__(self):
        super(CombinedCNN, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.speccnn = SpectralCNN()
        self.spatcnn = SpatialCNN()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x1=self.speccnn.forward(torch.nanmean(x,(3,4))[:,np.newaxis,:,:])
        x2=self.spatcnn.forward(torch.nanmean(torch.nanmean(x,1,keepdims=True),2))
        print(x1.shape,x2.shape)
        return x1*x2
        
        



def classify_images_2stage(data_array, model_weights_path, verbose=False):
    """
    Classifies images based on the provided CNN model and data array.

    Parameters:
    - data_array (numpy.ndarray): A preprocessed array of images shaped as
      [num_batches, 25 times, 16 frequencies, image_height, image_width], where each
      image is expected to be resized to 50x50 pixels.
    - model_weights_path (str): Path to the file containing the trained model weights.
    - verbose (bool, optional): If True, prints detailed output during classification.

    Returns:
    - predictions (numpy.ndarray): An array of binary predictions (0 or 1) for each image.
    - probabilities (numpy.ndarray): An array of probabilities corresponding to the
      predictions, indicating the confidence level of each prediction.
    """
    model = CombinedCNN()
    model.load_state_dict(torch.load(model_weights_path,map_location=torch.device('cpu'),weights_only=True))
    model.eval()

    predictions = []  # List to store binary predictions (0s and 1s)
    probabilities = []  # List to store predicted probabilities

    num_batches = data_array.shape[0]
    for i in range(num_batches):
        if verbose:
            print(f"\nClassifying Candidate {i}")
        data = data_array[i,:,:,:,:]
        if verbose:
            print(f"Original shape: {data.shape}")

        # 2) Resize from (H=301, W=301, T=25, C=16) to (50,50,25,16)
        resized_cube = resize_cube(data, size=(50, 50))  # shape => (50, 50, 25, 16)
        if verbose:
            print(f"Resized shape: {resized_cube.shape}")

        # 3) Permute to (16, 25, 50, 50) = (C, T, H, W)
        transposed = np.transpose(resized_cube, (3, 2, 0, 1))  # (C=16, T=25, H=50, W=50)

        # 4) Expand batch dimension => (1,16,25,50,50)
        model_input = np.expand_dims(transposed, axis=0)  # shape => (1,16,25,50,50)

        # 5) Load model & weights
        model = CombinedCNN()
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'),weights_only=True))
        model.eval()

        # 6) Convert to torch.Tensor
        tensor_input = torch.from_numpy(model_input)

        # Normalization
        mean = tensor_input.mean()
        std = tensor_input.std()
        tensor_input = (tensor_input - mean) / (std + 1e-8)

        # 7) Run inference
        with torch.no_grad():
            logits = model(tensor_input)
            logits = logits.squeeze()
            prob = torch.sigmoid(logits).item()
            pred_label = 1 if prob > 0.5 else 0

        predictions.append(pred_label) #1 is RFI, 0 is Source
        probabilities.append(prob) #prob>50% is RFI, <50% is Source
        if verbose:
            label_desc = "RFI" if pred_label == 1 else "Source"

            print("==== Inference Results ====")
            print(f"Logit: {logits.item():.4f}")
            print(f"Sigmoid Probability: {prob:.4f}")
            print(f"Prediction: {label_desc} (binary = {pred_label})")

    return np.array(predictions), np.array(probabilities)  # Return both predictions and probabilities as NumPy arrays

