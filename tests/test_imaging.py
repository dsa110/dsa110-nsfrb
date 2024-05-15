import pytest
import numpy as np
from nsfrb.imaging import briggs_weighting, robust_image, uniform_image
from nsfrb.config import IMAGE_SIZE

def test_briggs_weighting():
    u = np.array([10, 20, 30])
    v = np.array([15, 25, 35])
    grid_size = 64
    vis_weights = np.array([1, 2, 3])
    robust = 0.5

    result = briggs_weighting(u, v, grid_size, vis_weights, robust)
    assert result is not None
    assert result.shape == vis_weights.shape
    assert np.all(result > 0)

def test_robust_image():
    chunk_V = np.random.random((10, 3)) + 1j * np.random.random((10, 3))
    u = np.array([10, 20, 30])
    v = np.array([15, 25, 35])
    image_size = IMAGE_SIZE
    robust = 0.5

    result = robust_image(chunk_V, u, v, image_size, robust)
    assert result is not None
    assert result.shape == (image_size, image_size)
    assert isinstance(result, np.ndarray)
    assert np.all(np.isfinite(result))

def test_uniform_image_delta_visibility():
    # Delta function visibility with (u, v) at the origin and other points with zero visibility
    u = np.array([0, 1, -1, 1, -1])
    v = np.array([0, 1, -1, -1, 1])
    chunk_V = np.array([10 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])
    image_size = IMAGE_SIZE

    result = uniform_image(chunk_V, u, v, image_size)

    assert result is not None
    assert result.shape == (image_size, image_size)
    assert isinstance(result, np.ndarray)
    assert np.all(np.isfinite(result))

    # Check if the image is approximately uniform
    mean_value = np.mean(result)
    assert np.allclose(result, mean_value, atol=1e-2)

def test_uniform_image_constant_visibility():
    # Constant visibility with non-zero u, v values
    u = np.linspace(1, 10, 10)
    v = np.linspace(1, 10, 10)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    chunk_V = np.ones_like(u) + 0j
    image_size = IMAGE_SIZE

    result = uniform_image(chunk_V, u, v, image_size)

    assert result is not None
    assert result.shape == (image_size, image_size)
    assert isinstance(result, np.ndarray)
    assert np.all(np.isfinite(result))
    # Check if the image has a peak at the center (delta function)
    center_value = result[image_size // 2, image_size // 2]
    max_value = np.max(result)
    assert np.isclose(center_value, max_value)
    # Ensure the peak is significantly higher than other values
    assert center_value > 10 * np.mean(result)

if __name__ == "__main__":
    pytest.main()
