# Simulations and Classifications

This folder contains scripts for simulating Radio Frequency Interference (RFI) images, as well as source images, with DSA-110 core antennas and classifying between them.

## Structure

The folder is structured as follows:

- `generate_rfi_images.py`: This script uses functions from the `simulating.py` and `imaging.py` modules to generate RFI images. For each observation, the script randomizes the HA and Dec, calculates the u,v,w coordinates, adds complex Gaussian noise to the visibilities, and applies RFI based on the selected RFI types (far or near). It then generates dirty images and saves the images along with metadata (in `rfi_examples` folder).

- `generate_source_images.py`: This script uses functions from the same modules to generate images of a source. For each observation, the script randomizes the Hour Angle (HA) and Declination (Dec), calculates the u,v,w coordinates, applies spectral index to the visibilities, adds complex Gaussian noise, and generates dirty images. The script saves the images along with metadata in the specified dataset directory (in `src_examples` folder).

- `rfi_classification_pytorch.ipynb`: This Jupyter notebook contains an example of the implementation of a Convolutional Neural Network (CNN) for classifying the generated RFI and Source images. It includes:
    - Data loading and preprocessing: Utilizes the custom `ImageCubeDataset` class for efficient data handling and transformations.
    - Model architecture: Features both `SimpleCNN` and `EnhancedCNN` models, designed to suit different complexity levels.
    - Training and validation process: Iterative model training with real-time display of loss and accuracy metrics for both training and validation datasets.
    - Visualization tools: Functions for displaying training images, plotting loss and accuracy graphs, and showing wrongly classified images for model evaluation.
    - Model saving and loading mechanisms: Demonstrates how to save trained model weights for later use and how to load them for inference. The `model_weights.pth` can be found in the current directory.

- `classify.py`: This script contains the implementation of the `ImageCubeDataset` class for loading image cubes and the `EnhancedCNN` model for image classification. It also includes the `classify_images` function for classifying images in a specified directory using the trained model.

## Usage

To generate images, run the `generate_rfi_images.py` or `generate_source_images.py` script. You can customize the simulation process by passing arguments to the script. For example:

```bash
python generate_rfi_images.py --num_observations 10 --dist_low 1000 --dist_high 100000 --zoom_pix 50