import numpy as np
from scipy.fftpack import ifftshift, ifft2
from nsfrb.config import IMAGE_SIZE


def briggs_weighting(u: np.ndarray, v: np.ndarray, grid_size: int, vis_weights: np.ndarray = None, robust: float = 0.0) -> np.ndarray:
    """
    Apply Briggs weighting to visibility data.

    Parameters:
    u, v: u,v coordinates.
    grid_size: Size of the grid to be used for imaging.
    vis_weights: Weights for each visibility. Defaults to uniform weighting if None.
    robust: Robust parameter for weighting. r=2 is close to uniform weighting.

    Returns:
    The Briggs-weighted visibility data.
    """
    if vis_weights is None:
        vis_weights = np.ones(u.shape)

    u_indices = ((u + np.max(u)) / (2 * np.max(u)) * (grid_size - 1)).astype(int)
    v_indices = ((v + np.max(v)) / (2 * np.max(v)) * (grid_size - 1)).astype(int)

    uv_grid = np.bincount(u_indices * grid_size + v_indices, weights=vis_weights, minlength=grid_size**2)
    Wk = uv_grid.flatten()

    f2 = (5 * 10 ** (-robust)) ** 2 / (np.sum(Wk ** 2) / np.sum(vis_weights))

    new_weights = vis_weights / (1 + Wk[u_indices * grid_size + v_indices] * f2)

    return new_weights


def robust_image(chunk_V: np.ndarray, u: np.ndarray, v: np.ndarray, image_size: int = IMAGE_SIZE, robust: float = 0.0) -> np.ndarray:
    """
    Process visibility data and create a dirty image using FFT and Briggs weighting.

    Parameters:
    chunk_V: Chunk of visibility data.
    u, v: u,v coordinates.
    image_size: Size of the output image.
    robust: Robust parameter for Briggs weighting.

    Returns:
    The resulting 'dirty' image.
    """
    pixel_resolution = (0.20 / np.max(np.sqrt(u**2 + v**2))) / 3
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    briggs_weights = briggs_weighting(u, v, image_size, robust=robust)

    weighted_V = chunk_V * briggs_weights
    V_avg = np.mean(weighted_V, axis=0)

    i_indices = np.clip((u + uv_max) / grid_res, 0, image_size - 1).astype(int)
    j_indices = np.clip((v + uv_max) / grid_res, 0, image_size - 1).astype(int)

    visibility_grid = np.zeros((image_size, image_size), dtype=complex)
    np.add.at(visibility_grid, (i_indices, j_indices), V_avg)

    dirty_image = ifftshift(ifft2(ifftshift(visibility_grid)))

    return dirty_image


def uniform_image(chunk_V: np.ndarray, u: np.ndarray, v: np.ndarray, image_size: int) -> np.ndarray:
    """
    Converts visibility data into a 'dirty' image.

    Parameters:
    chunk_V: Visibility data (complex numbers).
    u, v: Coordinates in UV plane.
    image_size: Output size.

    Returns:
    A numpy array representing the dirty image.
    """
    pixel_resolution = (0.20 / np.max(np.sqrt(u ** 2 + v ** 2))) / 3
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    v_avg = np.mean(np.array(chunk_V), axis=0)

    i_indices = np.clip((u + uv_max) / grid_res, 0, image_size - 1).astype(int)
    j_indices = np.clip((v + uv_max) / grid_res, 0, image_size - 1).astype(int)

    visibility_grid = np.zeros((image_size, image_size), dtype=complex)
    np.add.at(visibility_grid, (i_indices, j_indices), v_avg)

    dirty_image = ifftshift(ifft2(ifftshift(visibility_grid)))

    return np.abs(dirty_image)