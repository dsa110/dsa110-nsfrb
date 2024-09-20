import numpy as np
from scipy.fftpack import ifftshift, ifft2,fftshift,fft2,fftfreq
from nsfrb.config import IMAGE_SIZE,UVMAX
#modules for position and RA/DEC calibration
from influxdb import DataFrameClient
from astropy.coordinates import EarthLocation, AltAz, ICRS,SkyCoord
import astropy.units as u
from astropy.time import Time
import sys
from matplotlib import pyplot as plt

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
import os
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
output_file = cwd + "-logfiles/run_log.txt"

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

    return np.real(dirty_image)


def uniform_image(chunk_V: np.ndarray, u: np.ndarray, v: np.ndarray, image_size: int, return_complex=False, inject_img=None) -> np.ndarray:
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

    if inject_img is not None:
        visibility_grid += inverse_uniform_image(inject_img,u,v) 

    #count_indices = np.array([np.sum(np.logical_and(i_indices==i_indices[k],j_indices==j_indices[k])) for k in range(len(i_indices))])
    #visibility_grid[i_indices, j_indices] /= count_indices

    dirty_image = ifftshift(ifft2(ifftshift(visibility_grid)))
    
    return np.real(dirty_image) if not return_complex else dirty_image


def inverse_uniform_image(dirty_image,u,v):
    """
    Inverse of uniform_image, used for injection purposes; inverts image to get gridded visibilities

    Parameters:
    dirty_image: Dirty image with shape (gridsize,gridsize)
    pixel_resolution: image pixel size in degrees (?)
    u,v: Coordinates in UV plane at which visibilities should be returned; the nearest grid point will be used
    """

    image_size = dirty_image.shape[0]
    pixel_resolution = (0.20 / np.max(np.sqrt(u ** 2 + v ** 2))) / 3
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    visibility_grid = fftshift(fft2(fftshift(dirty_image)))
    
    
    """
    #get nearest visibility grid point for each u,v
    i_indices = np.clip((u + uv_max) / grid_res, 0, image_size - 1).astype(int)
    j_indices = np.clip((v + uv_max) / grid_res, 0, image_size - 1).astype(int)
    #count_indices = np.array([np.sum(np.logical_and(i_indices==i_indices[k],j_indices==j_indices[k])) for k in range(len(i_indices))])
    chunk_V = visibility_grid[i_indices,j_indices]#/count_indices
    """
    return visibility_grid



#added this function to output the RA and DEC coordinates of each pixel in an image
influx = DataFrameClient('influxdbservice.pro.pvt', 8086, 'root', 'root', 'dsa110')
def uv_to_pix(mjd_obs,image_size,uv_diag=UVMAX,Lat=37.23,Lon=-118.2851,timerangems=100,maxtries=5,output_file=output_file):
    """
    Takes UV grid coordinates and converts them to RA and declination

    Parameters:
    mjd_obs: Observing time as Mean Julian Date 
    uv_diag: maximum UV plane extent, i.e. max(sqrt(u^2 + v^2))
    image_size: output size
    Lat,Lon: coordinates of observing site
    timerangems: time range to query for elevation from etcd
    maxtries: maximum number of iterations before giving up

    Returns:
    RA and DEC axes as numpy arrays
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #create offset grid using pixel size and max UV diagonal distance
    pixel_resolution = (0.20 / uv_diag) / 3
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2

    offset_grid = np.arange(-1/uv_resolution/2,1/uv_resolution/2,1/uv_max/2)*180/np.pi
    #print(len(offset_grid) , image_size)
    assert(len(offset_grid) == image_size)

    #find RA, dec at center of the image 

    #(1) ovro location
    loc = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg) #default is ovro

    #(2) observation time
    tobs = Time(mjd_obs,format='mjd')
    tms = int(tobs.unix*1000) #ms

    #(3) query antenna elevation at obs time
    result = dict()
    tries = 0
    while len(result) == 0 and tries < maxtries:
        query = f'SELECT time,ant_el FROM "antmon" WHERE time >= {tms-timerangems}ms and time < {tms+timerangems}ms'
        result = influx.query(query)
        timerangems *= 10
        tries += 1
    if len(result) == 0:
        print("Failed to retrieve elevation, using RA,DEC = 0,0",file=fout)
        icrs_pos = ICRS(ra=0*u.deg,dec=0*u.deg)
    else:
        bestidx = np.argmin(np.abs(tobs.mjd - Time(np.array(result['antmon'].index),format='datetime').mjd))
        elev = result['antmon']['ant_el'].values[bestidx]
        alt = 180-elev
        alt = elev - 90
        antpos = AltAz(obstime=tobs,location=loc,az=180*u.deg,alt=alt*u.deg)

        #(4) convert to ICRS frame
        icrs_pos = antpos.transform_to(ICRS())
    
    print("Retrieved Coordinates: " + str(tobs.isot) + ", RA="+str(icrs_pos.ra.value) + "deg, DEC="+str(icrs_pos.dec.value) + "deg",file=fout)

    #add offset from image center
    ra_grid = icrs_pos.ra.value + offset_grid
    dec_grid = icrs_pos.dec.value + offset_grid
    if output_file != "":
        fout.close()
    return ra_grid,dec_grid
