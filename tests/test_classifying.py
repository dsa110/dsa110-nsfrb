import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from nsfrb.classifying import NumpyImageCubeDataset, EnhancedCNN, classify_images

def test_numpy_image_cube_dataset():
    # Create a dummy data array
    data_array = np.random.rand(10, 16, 50, 50)  # (num_images, channels, height, width)
    dataset = NumpyImageCubeDataset(data_array)

    # Check dataset length
    assert len(dataset) == 10

    # Check individual item shape
    item = dataset[0]
    assert item.shape == (16, 50, 50)

def test_enhanced_cnn():
    # Create a dummy input tensor
    input_tensor = torch.rand(1, 16, 50, 50)

    model = EnhancedCNN()
    output = model(input_tensor)

    # Check output shape
    assert output.shape == (1, 1)

def test_classify_images(tmpdir):
    # Load the generated NumPy arrays
    rfi_data_array = np.load('../tests/data/rfi_image_data.npy')

    # Check the shape of the loaded data
    assert rfi_data_array.shape == (16, 50, 50)

    # Reshape the data to match the model's expected input shape
    rfi_data_array = np.expand_dims(rfi_data_array, axis=0)  # (1, channels, height, width)

    # Check the final shape
    assert rfi_data_array.shape == (1, 16, 50, 50)

    # Create a dummy model and save its state_dict
    model = EnhancedCNN()
    model_path = str(tmpdir.join("dummy_model.pth"))  # Convert LocalPath to string
    torch.save(model.state_dict(), model_path)

    # Classify images using the dummy model
    rfi_predictions, rfi_probabilities = classify_images(rfi_data_array, model_path)

    # Check predictions and probabilities shapes
    assert rfi_predictions.shape == (rfi_data_array.shape[0],)
    assert rfi_probabilities.shape == (rfi_data_array.shape[0],)
    assert np.all((rfi_predictions == 0) | (rfi_predictions == 1))
    assert np.all((rfi_probabilities >= 0) & (rfi_probabilities <= 1))

def test_classify_known_images(tmpdir):
    # Load known data
    rfi_data = np.load('../tests/data/rfi_image_data.npy')

    # Check the shape of the loaded data
    assert rfi_data.shape == (16, 50, 50)

    # Reshape the data to match the model's expected input shape
    rfi_data = np.expand_dims(rfi_data, axis=0)  # (1, channels, height, width)

    # Check the final shape
    assert rfi_data.shape == (1, 16, 50, 50)

    # Dummy labels for testing
    rfi_labels = np.random.randint(0, 2, size=rfi_data.shape[0])

    # Create a dummy model and save its state_dict
    model = EnhancedCNN()
    # load pre-trained weights
    model_path = str(tmpdir.join("known_model.pth"))
    torch.save(model.state_dict(), model_path)

    # Classify images using the model
    rfi_predictions, rfi_probabilities = classify_images(rfi_data, model_path)

    # Check predictions against known labels
    expected_rfi_label = 1  

    # Check rfi predictions
    assert rfi_predictions.shape == rfi_labels.shape
    assert np.all((rfi_predictions == 0) | (rfi_predictions == 1))
    rfi_accuracy = np.mean(rfi_predictions == expected_rfi_label)
    print(f"RFI classification accuracy: {rfi_accuracy * 100:.2f}%")
    assert rfi_accuracy >= 0.5  # Set a reasonable threshold for the accuracy

    # Check rfi probabilities
    assert rfi_probabilities.shape == rfi_labels.shape
    assert np.all((rfi_probabilities >= 0) & (rfi_probabilities <= 1))
    avg_probability = np.mean(rfi_probabilities)
    print(f"Average RFI prediction probability: {avg_probability:.2f}")
    assert avg_probability >= 0.5  # Set a reasonable threshold for the average probability

if __name__ == "__main__":
    pytest.main()

