import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import ifftshift, ifft2
from scipy.stats import linregress
import time
import os
import pandas as pd
import random
from antpos.utils import get_itrf

def get_core_coordinates():
    """
    Extracts core antenna coordinates from a dataframe.

    Parameters:
    df: DataFrame containing antenna coordinates with columns 'x_m', 'y_m', 'z_m'.

    Returns:
    x_core, y_core, z_core: core coordinates of the antennas.
    """
    df = get_itrf()
    x_m = df['x_m'].values
    y_m = df['y_m'].values
    z_m = df['z_m'].values

    # Define core antenna limits
    x_min, x_max = -2.41e6, -2.4092e6
    y_min, y_max = -4.47830e6, -4.47775e6

    # Create a mask for core antennas
    core_mask = (x_m > x_min) & (x_m < x_max) & (y_m > y_min) & (y_m < y_max)

    # Extract core coordinates using the mask
    x_core = x_m[core_mask]
    y_core = y_m[core_mask]
    z_core = z_m[core_mask]

    return x_core, y_core, z_core
def compute_uvw(x_m, y_m, z_m, HA, Dec):

    """
    Computes the u, v, w coordinates for baseline vectors between pairs of antennas.

    Parameters:
    x_m, y_m, z_m: Arrays of x, y, and z coordinates of the antennas in meters.
    HA: Hour Angle in radians.
    Dec: Declination in radians.

    Returns:
    Arrays of u, v, w coordinates for each baseline.
    """
    N = len(x_m)
    u, v, w = [], [], []

    for i in range(N):
        for j in range(i + 1, N):
            dx = x_m[j] - x_m[i]
            dy = y_m[j] - y_m[i]
            dz = z_m[j] - z_m[i]

            u_ij = dx * np.sin(HA) + dy * np.cos(HA)
            v_ij = -dx * np.sin(Dec) * np.cos(HA) + dy * np.sin(Dec) * np.sin(HA) + dz * np.cos(Dec)
            w_ij = dx * np.cos(Dec) * np.cos(HA) - dy * np.cos(Dec) * np.sin(HA) + dz * np.sin(Dec)

            u.append(u_ij)
            v.append(v_ij)
            w.append(w_ij)

    return np.array(u), np.array(v), np.array(w)

def plot_uv_coverage(u, v, title='u-v Coverage'):
    """
    Plot the u-v coverage.
    This function creates a scatter plot of the u-v points and their symmetrical counterparts. It is used to visualize the spatial frequency coverage in radio interferometry.
    Parameters:
    u and v: Arrays of coordinates.
    """
    max_u = max(np.max(u), -np.min(u))
    min_u = min(np.min(u), -np.max(u))
    max_v = max(np.max(v), -np.min(v))
    min_v = min(np.min(v), -np.max(v))

    plt.scatter(u, v, marker='.', color='b')
    plt.scatter(-u, -v, marker='.', color='b')  # Symmetry
    plt.xlim(min_u, max_u)
    plt.ylim(min_v, max_v)
    plt.xlabel('u (m)')
    plt.ylabel('v (m)')
    plt.title(title)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    return

def apply_phase_shift(visibilities, u_shift_rad, v_shift_rad):
    """
    Apply a phase shift to the visibility data.

    Parameters:
    visibilities (array): Array of complex visibilities.
    u_shift_rad (float): Phase shift in the u-direction (radians).
    v_shift_rad (float): Phase shift in the v-direction (radians).

    Returns:
    array: The visibility data after applying the phase shift.
    """
    phase_shift = np.exp(1j * (u_shift_rad * u_core + v_shift_rad * v_core))
    return visibilities * phase_shift

def add_complex_gaussian_noise(V, mean=0, std_dev=1):
    """
    Add complex Gaussian noise to visibility data for better simulations.

    Parameters:
    V (array):visibility data.
    mean (float, optional): Mean of the Gaussian noise (default = 0).
    std_dev (float, optional): Std of the noise. (default = 1). For better simulation results should be increased.

    Returns:
    array: Vis data with noise.
    """
    real_noise = np.random.normal(mean, std_dev, V.shape)
    imag_noise = np.random.normal(mean, std_dev, V.shape)

    return V + real_noise + 1j*imag_noise

def apply_spectral_index(visibility, frequency, reference_frequency, spectral_index):
    """
    Apply a spectral index scaling to vis data.

    Parameters:
    visibility (array): vis data.
    frequency (float): Frequency at which the visibility data is observed.
    reference_frequency (float): Reference frequency for the spectral index.
    spectral_index (float): Spectral index to be applied.

    Returns:
    array: new vis data.
    """
    scaling_factor = (frequency / reference_frequency)**spectral_index
    return visibility * scaling_factor

def rfi_source_position(x_core, y_core, z_core, azimuth, elevation, distance=1e7):
    """
    Calculate the position of an RFI source based on azimuth and elevation.

    Parameters:
    x_core, y_core, z_core (arrays): Coordinates of core antennas.
    azimuth (float): Azimuth angle of the RFI source (in degrees).
    elevation (float): Elevation angle of the RFI source (in degrees).
    distance (float, optional): Distance to the RFI source. Default is 1e7 meters.

    Returns:
    tuple: Position coordinates (x, y, z) of the RFI source.
    """
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    x_mean = np.mean(x_core)
    y_mean = np.mean(y_core)
    z_mean = np.mean(z_core)

    # RFI source position based on azimuth and elevation
    rfi_x = x_mean + distance * np.cos(elevation) * np.sin(azimuth)
    rfi_y = y_mean + distance * np.cos(elevation) * np.cos(azimuth)
    rfi_z = z_mean + distance * np.sin(elevation)

    return rfi_x, rfi_y, rfi_z

def simulate_near_field_rfi(x, y, z, visibilities, rfi_position, rfi_amplitude, frequency):
    """
    Simulate the effect of near-field RFI on visibilities.

    Parameters:
    x,y,z (arrays): Antenna coordinates.
    visibilities (array): Visibility data to be affected.
    rfi_position (tuple): Coordinates of the RFI source.
    rfi_amplitude (float): Amplitude of the RFI signal.
    frequency (float): Frequency of observation.

    Returns:
    array: Modified visibilities with near-field RFI effects.
    """
    rfi_x, rfi_y, rfi_z = rfi_position

    deltas = np.sqrt((x - rfi_x) ** 2 + (y - rfi_y) ** 2 + (z - rfi_z) ** 2)
    delays = deltas / c

    rfi_signal = rfi_amplitude * np.exp(2j * np.pi * frequency * delays)

    num_antennas = len(x)
    k = 0
    for i in range(num_antennas):
        for j in range(i + 1, num_antennas):
            visibilities[k] += rfi_signal[i] + rfi_signal[j]
            k += 1

    return visibilities


