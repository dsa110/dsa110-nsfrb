import torch
#torch.set_default_dtype(torch.FloatTensor)
torch.set_default_device("cpu")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import argparse
import numpy as np
from torchvision.transforms import functional as TF

# Define the transformation for image resizing
transform = transforms.Compose([
    transforms.Resize((50, 50)),
])
"""
transform = transforms.Compose([
    transforms.RandomRotation(360),
    transforms.RandomResizedCrop(50, scale=(0.8, 1.0)),
    transforms.Resize((50, 50)),
])
"""

class ImageCubeDataset(Dataset):
    """
    Custom dataset class for loading multiple batches of image cubes.; FROM RFI_CLASSIFICATION_PYTORCH.IPYNB NOTEBOOK
    """
    def __init__(self, root_dir, transform=None):
        self.obs_dirs = [os.path.join(root_dir, obs)
                         for obs in sorted(os.listdir(root_dir))
                         if os.path.isdir(os.path.join(root_dir, obs))]
        self.transform = transform

    def __len__(self):
        return len(self.obs_dirs)

    def __getitem__(self, idx):
        obs_dir = self.obs_dirs[idx]
        image_cube = self.load_image_cube(obs_dir)
        if self.transform:
            image_cube = self.transform(image_cube)
        return image_cube

    def load_image_cube(self, observation_dir):
        subband_images = []
        image_dir = os.path.join(observation_dir, 'images')
        filenames = sorted(os.listdir(image_dir))

        for filename in filenames:
            if filename.endswith(".png") and "subband_avg" in filename:
                img_path = os.path.join(image_dir, filename)
                img = Image.open(img_path).convert('L')  # image to grayscale
                subband_images.append(img)



        tensor_stack = torch.stack([transforms.ToTensor()(img)[0] for img in subband_images], dim=0)




        return tensor_stack



class NumpyImageCubeDataset(Dataset):
    """
    Custom dataset class for loading multiple batches of image cubes.
    """
    def __init__(self, data_array, transform=None):
        """
        Args:
            data_array (numpy.ndarray): Array of shape [num_batches, 16 frequencies, image_length, image_width].
            transform (callable, optional): Optional transform to be applied.
        """
        self.data = data_array
        self.transform = transform

    def __len__(self):
        # Return the number of batches
        return self.data.shape[0]

    def __getitem__(self, idx):
        image_cube = self.data[idx]  
        if self.transform:
            transformed_channels = []
            for i in range(image_cube.shape[0]):
                img = TF.to_pil_image(image_cube[i].astype(np.float32),mode='L')
                transformed_img = self.transform(img)
                transformed_channels.append(TF.to_tensor(transformed_img))
            image_cube_tensor = torch.stack(transformed_channels, dim=0)
        else:
            image_cube_tensor = torch.tensor(image_cube, dtype=torch.float32)

        image_cube_tensor = image_cube_tensor.squeeze()

        return image_cube_tensor






class EnhancedCNN(nn.Module):
    """
    Enhanced Convolutional Neural Network (CNN) for image classification.
    """

    def __init__(self):
        super(EnhancedCNN, self).__init__()

        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x25x25
        x = self.pool(F.relu(self.conv2(x)))  # 128x12x12
        x = self.pool(F.relu(self.conv3(x)))  # 256x6x6

        x = x.view(-1, 6 * 6 * 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def classify_images(data_array, model_weights_path, verbose=False):
    """
    Classifies images based on the provided CNN model and data array.

    Parameters:
    - data_array (numpy.ndarray): A preprocessed array of images shaped as
      [num_batches, 16 frequencies, image_height, image_width], where each
      image is expected to be resized to 50x50 pixels.
    - model_weights_path (str): Path to the file containing the trained model weights.
    - verbose (bool, optional): If True, prints detailed output during classification.

    Returns:
    - predictions (numpy.ndarray): An array of binary predictions (0 or 1) for each image.
    - probabilities (numpy.ndarray): An array of probabilities corresponding to the
      predictions, indicating the confidence level of each prediction.
    """
    model = EnhancedCNN()
    model.load_state_dict(torch.load(model_weights_path,map_location=torch.device('cpu')))
    model.eval()

    predictions = []  # List to store binary predictions (0s and 1s)
    probabilities = []  # List to store predicted probabilities

    test_dataset = NumpyImageCubeDataset(data_array, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for inputs in test_loader:
            outputs = model(inputs).squeeze()
            if verbose: print("output prob:",outputs)
            if verbose: print("sigmoid prob:",torch.sigmoid(outputs))
            predicted_prob = outputs #torch.sigmoid(outputs).item()
            predicted_label = 1 if predicted_prob > 0.5 else 0
            predictions.append(predicted_label)  # Store binary prediction
            probabilities.append(predicted_prob)  # Store probability
            
            if verbose:  # Check if verbose output is enabled
                label_description = "RFI" if predicted_label == 1 else "Source"
                print(f"Predicted Label: {label_description}, Probability: {predicted_prob}")

    return np.array(predictions), np.array(probabilities)  # Return both predictions and probabilities as NumPy arrays

def classify_images_dataset(dataset_dir, model_weights_path, verbose=False):
    """
    Classifies images based on the provided CNN model and data array.

    Parameters:
    - dataset_dir (numpy.ndarray): Dataset directory with preprocessed array of images shaped as
      [num_batches, 16 frequencies, image_height, image_width], where each
      image is expected to be resized to 50x50 pixels.
    - model_weights_path (str): Path to the file containing the trained model weights.
    - verbose (bool, optional): If True, prints detailed output during classification.

    Returns:
    - predictions (numpy.ndarray): An array of binary predictions (0 or 1) for each image.
    - probabilities (numpy.ndarray): An array of probabilities corresponding to the
      predictions, indicating the confidence level of each prediction.
    """
    model = EnhancedCNN()
    model.load_state_dict(torch.load(model_weights_path,map_location=torch.device('cpu')))
    model.eval()

    predictions = []  # List to store binary predictions (0s and 1s)
    probabilities = []  # List to store predicted probabilities

    test_dataset = ImageCubeDataset(dataset_dir,transform=transform)#NumpyImageCubeDataset(data_array, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for inputs in test_loader:
            outputs = model(inputs).squeeze()
            if verbose: print("output prob:",outputs)
            if verbose: print("sigmoid prob:",torch.sigmoid(outputs))
            predicted_prob = outputs #torch.sigmoid(outputs).item()
            predicted_label = 1 if predicted_prob > 0.5 else 0
            predictions.append(predicted_label)  # Store binary prediction
            probabilities.append(predicted_prob)  # Store probability

            if verbose:  # Check if verbose output is enabled
                label_description = "RFI" if predicted_label == 1 else "Source"
                print(f"Predicted Label: {label_description}, Probability: {predicted_prob}")

    return np.array(predictions), np.array(probabilities)  # Return both predictions and probabilities as NumPy arrays



            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--npy_file', type=str, required=True, help='Path to the NumPy file containing the images')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Load the NumPy file once, outside the dataset class
    data_array = np.load(args.npy_file)
    # Do all the reshapig so that we have (num_batches, 16 frequencies, image dimensions) shaped array
    data_array = np.nan_to_num(data_array, nan=0.0)
    transposed_array = np.transpose(data_array, (0, 3, 4, 1, 2))
    new_shape = (data_array.shape[0] * data_array.shape[3], data_array.shape[4], data_array.shape[1], data_array.shape[2])
    merged_array = transposed_array.reshape(new_shape) 

    predictions, probabilities = classify_images(merged_array, args.model_weights, verbose=args.verbose)
    
    print(predictions, probabilities) 
