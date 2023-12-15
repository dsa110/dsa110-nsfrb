import numpy as np
from scipy.fftpack import ifftshift, ifft2
from nsfrb.config import IMAGE_SIZE


def briggs_weighting(u, v, grid_size, vis_weights=None, robust=0.0):
    """
    Apply Briggs weighting to visibility data (see, e.g., https://casa.nrao.edu/Release4.1.0/doc/UserMan/UserMansu262.html).

    Parameters:
    u, v (arrays): u,v coordinates.
    grid_size: Size of the grid to be used for imaging.
    vis_weights (array, optional): Weights for each visibility. Defaults to uniform weighting.
    robust (float, optional): Robust parameter for weighting. r=2 is close to uniform weighting.

    Returns:
    array: The Briggs-weighted visibility data.
    """
    # If vis_weights is None, default to an array of ones
    if vis_weights is None:
        vis_weights = np.ones(u.shape)

    uv_grid = np.zeros((grid_size, grid_size))

    # u and v coordinates to grid indices
    u_indices = ((u + np.max(u)) / (2 * np.max(u)) * (grid_size - 1)).astype(int)
    v_indices = ((v + np.max(v)) / (2 * np.max(v)) * (grid_size - 1)).astype(int)

    # Wk - sum of the weights of visibilities that fall into each grid cell
    for u_idx, v_idx, weight in zip(u_indices, v_indices, vis_weights):
        uv_grid[u_idx, v_idx] += weight

    Wk = uv_grid.flatten()

    # Compute f^2
    f2 = (5 * 10 ** (-robust)) ** 2 / (np.sum(Wk ** 2) / np.sum(vis_weights))

    new_weights = vis_weights / (1 + Wk[u_indices * grid_size + v_indices] * f2)

    return new_weights


def robust_image(chunk_V, u, v, image_size = IMAGE_SIZE, robust=0.0):
    """
     Process visibility data and create a dirty image using FFT and Briggs weighting.

     Parameters:
     chunk_V (array): Chunk of visibility data.
     u, v (arrays): u,v coordinates.
     image_size (int): Size of the output image (e.g., image_size = 300 --> 300 by 300 pixels).
     robust (float, optional): Robust parameter for Briggs weighting.

     Returns:
     array: The resulting 'dirty' image.
     """
    # Calculate uv_max and grid_res based on u, v, and image_size
    pixel_resolution = (0.20 / np.max(np.sqrt(u**2 + v**2))) / 3 # 3 pixels per ~FWHM of a beam
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size


    # Briggs weights using image_size for the UV grid
    briggs_weights = briggs_weighting(u, v, image_size, robust=robust)


    weighted_V = chunk_V * briggs_weights  # Applying weights along the visibility axis

    V_avg = np.mean(weighted_V, axis=0)

    i_values = np.clip((u + uv_max) / grid_res, 0, image_size - 1).astype(int)
    j_values = np.clip((v + uv_max) / grid_res, 0, image_size - 1).astype(int)

    # weighted visibility grid
    visibility_grid = np.zeros((image_size, image_size), dtype=complex)
    np.add.at(visibility_grid, (i_values, j_values), V_avg)

    # visibility grid to the image plane
    dirty_image = ifftshift(ifft2(ifftshift(visibility_grid)))

    return dirty_image


def uniform_image(chunk_V, u, v, image_size):
    """
        Converts visibility data into a 'dirty' image.

        Parameters:
        chunk_V: Visibility data (complex numbers).
        u, v: coordinates in UV plane.
        image_size: output size.

        This function processes visibility data using the UV coordinates, averaging
        the data and mapping it onto a grid. It then generates a dirty image using
        inverse Fourier transform.

        Returns:
        A numpy array representing the dirty image.
        """
    pixel_resolution = (0.20 / np.max(np.sqrt(u ** 2 + v ** 2))) / 3  # 3 pixels per ~FWHM of a beam
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    v_avg = np.mean(np.array(chunk_V), axis=0)

    i_values = np.clip((u + uv_max) / grid_res, 0, image_size-1).astype(int)
    j_values = np.clip((v + uv_max) / grid_res, 0, image_size-1).astype(int)

    visibility_grid = np.zeros((image_size, image_size), dtype=complex)
    np.add.at(visibility_grid, (i_values, j_values), v_avg)

    dirty_image = ifftshift(ifft2(ifftshift(visibility_grid)))

    return np.abs(dirty_image)