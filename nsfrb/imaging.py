import numpy as np
from scipy.fftpack import ifftshift, ifft2


def briggs_weighting(u, v, grid_size, vis_weights=None, robust=0.0):
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


def process_and_image(chunk_V, u, v, uv_max, grid_res, image_size, robust=0.0):
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
