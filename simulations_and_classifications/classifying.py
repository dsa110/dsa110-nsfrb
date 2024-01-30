import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import argparse

# Define the transformation for image resizing
transform = transforms.Compose([
    transforms.Resize((50, 50)),
])

class ImageCubeDataset(Dataset):
    """
    Custom dataset class for loading image cubes.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory of the image cubes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.obs_dirs = [os.path.join(root_dir, obs)
                         for obs in sorted(os.listdir(root_dir))
                         if os.path.isdir(os.path.join(root_dir, obs))]
        self.transform = transform

    def __len__(self):
        """
        Get the number of image cubes in the dataset.

        Returns:
            int: Number of image cubes.
        """
        return len(self.obs_dirs)

    def __getitem__(self, idx):
        """
        Get a specific image cube from the dataset.

        Args:
            idx (int): Index of the image cube.

        Returns:
            tensor: The image cube.
        """
        obs_dir = self.obs_dirs[idx]
        image_cube = self.load_image_cube(obs_dir)
        if self.transform:
            image_cube = self.transform(image_cube)
        return image_cube

    def load_image_cube(self, observation_dir):
        """
        Load an image cube from the specified observation directory.

        Args:
            observation_dir (str): Directory containing the image cube.

        Returns:
            tensor: The loaded image cube.
        """
        subband_images = []
        image_dir = os.path.join(observation_dir, 'images')
        filenames = sorted(os.listdir(image_dir))

        for filename in filenames:
            if filename.endswith(".png") and "subband_avg" in filename:
                img_path = os.path.join(image_dir, filename)
                img = Image.open(img_path).convert('L')  # Convert image to grayscale
                subband_images.append(img)

        tensor_stack = torch.stack([transforms.ToTensor()(img)[0] for img in subband_images], dim=0)
        return tensor_stack


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


def classify_images(root_dir):
    """
    Classify the images in the specified directory using the trained model.

    Args:
        dir (str): Directory containing the images.
    """
    model = EnhancedCNN()
    model.load_state_dict(torch.load('/Users/nikita/dsa110-nsfrb/simulations_and_classifications/model_weights.pth'))
    model.eval()

    test_dataset = ImageCubeDataset(root_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data in test_loader:
            inputs = data
            outputs = model(inputs).squeeze()
            predicted_prob = torch.sigmoid(outputs).item()
            predicted_label = 1 if predicted_prob > 0.5 else 0
            label_description = "RFI" if predicted_label == 1 else "Source"
            print(f"Predicted Label: {label_description}, Probability: {predicted_prob}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--dir', type=str, help='Directory of the images')
    args = parser.parse_args()

    classify_images(args.dir)
