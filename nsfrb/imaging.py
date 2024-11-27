import numpy as np
from scipy.fftpack import ifftshift, ifft2,fftshift,fft2,fftfreq
from nsfrb.config import IMAGE_SIZE,UVMAX,flagged_antennas
#modules for position and RA/DEC calibration
from influxdb import DataFrameClient
from astropy.coordinates import EarthLocation, AltAz, ICRS,SkyCoord
import astropy.units as u
from astropy.time import Time
import sys
from matplotlib import pyplot as plt
from nsfrb import simulating#,planning
import copy

#flagged_antennas = [21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
import os
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file
"""
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
output_file = cwd + "-logfiles/run_log.txt"
"""

def flag_vis(dat, bname, blen, UVW, antenna_order, flagged_antennas, bmin):
    """
    Removes visibilities containing flagged antennas and below minimum 
    baseline length
    
    dat: visibility data (time x baseline x channel x pol)
    bname: baseline names
    blen: baseline lengths (baseline x 3)
    UVW: U,V,W, coordinates (baseline x 3)
    antenna_order: antenna ordering
    flagged_antennas: number of antennas to be flagged
    bmin: minimum baseline length in meters

    """
    #get indices of flagged visibilities
    flagged_vis = []
    for i in flagged_antennas:
        for j in np.array(antenna_order)[:antenna_order.index(i)]:
            flagged_vis.append(list(bname).index(str(j) + "-" + str(i)))
        for j in np.array(antenna_order)[antenna_order.index(i):]:
            flagged_vis.append(list(bname).index(str(i) + "-" + str(j)))
    flagged_vis = np.array(flagged_vis)
    print("Flagged visibilities:",bname[flagged_vis])
    unflagged_vis = np.array(list(set(np.arange(len(bname)))-set(flagged_vis)))
    print("Unflagged visibilities:",bname[unflagged_vis])
    antenna_order = list(set(antenna_order)-set(flagged_antennas))
    bname = bname[unflagged_vis]
    blen = blen[unflagged_vis]
    UVW = UVW[:,unflagged_vis,:]
    dat = dat[:,unflagged_vis,:,:]

    print(dat)
    #remove short baselines
    if bmin > 0:
        blen_mask = np.sqrt(np.sum(blen**2,axis=1))>=bmin
        bname = bname[blen_mask]
        blen = blen[blen_mask]
        UVW = UVW[:,blen_mask,:]
        dat = dat[:,blen_mask,:,:]
    return dat, bname, blen, UVW, antenna_order

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

    #uv_grid = np.bincount(u_indices * grid_size + v_indices, weights=vis_weights, minlength=grid_size**2)
    #print(np.any((u_indices * grid_size + v_indices) - np.min(u_indices * grid_size + v_indices)<0))
    #print(np.any(vis_weights<0))
    uv_grid = np.bincount((u_indices * grid_size + v_indices) - np.min(u_indices * grid_size + v_indices), weights=vis_weights, minlength=grid_size**2)
    Wk = uv_grid.flatten()

    f2 = (5 * 10 ** (-robust)) ** 2 / (np.sum(Wk ** 2) / np.sum(vis_weights))

    new_weights = vis_weights / (1 + Wk[u_indices * grid_size + v_indices] * f2)

    return new_weights


def robust_image(chunk_V: np.ndarray, u: np.ndarray, v: np.ndarray, image_size: int = IMAGE_SIZE, robust: float = 0.0, return_complex=False, inject_img=None, inject_flat=False) -> np.ndarray:
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
    v_avg = np.mean(weighted_V, axis=0)

    i_indices = np.clip((u + uv_max) / grid_res, 0, image_size - 1).astype(int)
    j_indices = np.clip((v + uv_max) / grid_res, 0, image_size - 1).astype(int)

    visibility_grid = np.zeros((image_size, image_size), dtype=complex)
    #np.add.at(visibility_grid, (i_indices, j_indices), v_avg)
    np.add.at(visibility_grid, (j_indices, i_indices), v_avg)

    if inject_img is not None:
        if inject_flat:
            visibility_grid[j_indices,i_indices] += inverse_uniform_image(inject_img,u,v)[j_indices,i_indices]
        else:
            visibility_grid += inverse_uniform_image(inject_img,u,v)
    dirty_image = ifftshift(ifft2(ifftshift(visibility_grid)))

    #return np.real(dirty_image)
    return np.real(dirty_image) if not return_complex else dirty_image

def revised_uniform_image(chunk_V: np.ndarray, u: np.ndarray, v: np.ndarray, image_size: int, return_complex=False, inject_img=None, inject_flat=False) -> np.ndarray:
    """
    Converts visibility data into a 'dirty' image. The following issues are corrected:
        - require odd image size
        - long baselines are excluded
        - conjugate visibilities are included
        - FFT sign convention respected

    Parameters:
    chunk_V: Visibility data (complex numbers).
    u, v: Coordinates in UV plane.
    image_size: Output size.

    Returns:
    A numpy array representing the dirty image.
    """
    pixel_resolution = (0.20 / np.max(np.sqrt(u ** 2 + v ** 2))) / 3 #radians if UV in meters
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    v_avg = np.mean(np.array(chunk_V), axis=0)
    
    #removed clip
    i_indices = ((u + uv_max) / grid_res).astype(int)
    j_indices = ((v + uv_max) / grid_res).astype(int)
    
    #remove long baselines
    uvs = np.sqrt(u**2 + v**2)
    v_avg = v_avg[uvs<uv_max]
    i_indices = i_indices[uvs<uv_max]
    j_indices = j_indices[uvs<uv_max]

    #get conjugate baselines
    i_conj_indices = image_size - i_indices - 1
    j_conj_indices = image_size - j_indices - 1

    visibility_grid = np.zeros((image_size, image_size), dtype=complex)
    np.add.at(visibility_grid, (j_indices, i_indices), v_avg)
    np.add.at(visibility_grid, (j_conj_indices, i_conj_indices), np.conj(v_avg))

    if inject_img is not None:
        #print("IN THE WRONG PLACE")
        if inject_flat:
            visibility_grid[j_indices,i_indices] += inverse_revised_uniform_image(inject_img,u,v)[j_indices,i_indices]
            visibility_grid[j_conj_indices,i_conj_indices] += inverse_revised_uniform_image(inject_img,u,v)[j_conj_indices,i_conj_indices]
        else:
            visibility_grid += inverse_revised_uniform_image(inject_img,u,v)

    #updated sign convention
    dirty_image = fftshift(fft2(ifftshift(visibility_grid)))
    return np.real(dirty_image) if not return_complex else dirty_image
    

def inverse_revised_uniform_image(dirty_image,u,v):
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

    visibility_grid = fftshift(ifft2(ifftshift(dirty_image)))


    """
    #get nearest visibility grid point for each u,v
    i_indices = np.clip((u + uv_max) / grid_res, 0, image_size - 1).astype(int)
    j_indices = np.clip((v + uv_max) / grid_res, 0, image_size - 1).astype(int)
    #count_indices = np.array([np.sum(np.logical_and(i_indices==i_indices[k],j_indices==j_indices[k])) for k in range(len(i_indices))])
    chunk_V = visibility_grid[i_indices,j_indices]#/count_indices
    """
    return visibility_grid

def uniform_image(chunk_V: np.ndarray, u: np.ndarray, v: np.ndarray, image_size: int, return_complex=False, inject_img=None, inject_flat=False) -> np.ndarray:
    """
    Converts visibility data into a 'dirty' image.

    Parameters:
    chunk_V: Visibility data (complex numbers).
    u, v: Coordinates in UV plane.
    image_size: Output size.

    Returns:
    A numpy array representing the dirty image.
    """
    pixel_resolution = (0.20 / np.max(np.sqrt(u ** 2 + v ** 2))) / 3 #radians if UV in meters
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    v_avg = np.mean(np.array(chunk_V), axis=0)

    #print("from uniform image:",list(v_avg))
    i_indices = np.clip((u + uv_max) / grid_res, 0, image_size - 1).astype(int)
    j_indices = np.clip((v + uv_max) / grid_res, 0, image_size - 1).astype(int)
    #print("from uniform image:",list(i_indices),list(j_indices))
    visibility_grid = np.zeros((image_size, image_size), dtype=complex)
    #np.add.at(visibility_grid, (i_indices, j_indices), v_avg)
    np.add.at(visibility_grid, (j_indices, i_indices), v_avg)

    if inject_img is not None:
        #print("IN THE WRONG PLACE")
        if inject_flat:
            visibility_grid[j_indices,i_indices] += inverse_uniform_image(inject_img,u,v)[j_indices,i_indices] 
        else:
            visibility_grid += inverse_uniform_image(inject_img,u,v)
    #count_indices = np.array([np.sum(np.logical_and(i_indices==i_indices[k],j_indices==j_indices[k])) for k in range(len(i_indices))])
    #visibility_grid[i_indices, j_indices] /= count_indices
    #print("from uniform image:",visibility_grid)
    dirty_image = ifftshift(ifft2(ifftshift(visibility_grid)))
    #print("from uniform image:",dirty_image)
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

az_offset=1.23001
Lat=37.23
Lon=-118.2851
def DSAelev_to_ASTROPYalt(elev,az=az_offset):
    """
    DSA110 uses elevation from 0 to 180 with azimuth fixed at 1.23 deg
    Astropy uses altitude from -90 to 90 (<0 = below the horizon) and azimuth 0 to 360
    This function converts in between them.
    
    elev: DSA-110 specified elevation
    az_offset: offset from perfect az=0
    """

    elev = np.array(elev)

    #if elevation > 90, need to shift to 0-90 range
    alt = copy.deepcopy(elev)
    alt[elev>90] = 180 - alt[elev>90]

    #if elevation <= 90, need to rotate az by 180 deg
    az = np.array(az*np.ones_like(alt))
    az[elev<=90] = 180 + az[elev<=90]
    return alt,az



#added this function to output the RA and DEC coordinates of each pixel in an image
influx = DataFrameClient('influxdbservice.pro.pvt', 8086, 'root', 'root', 'dsa110')
def uv_to_pix(mjd_obs,image_size,Lat=37.23,Lon=-118.2851,timerangems=1000,maxtries=5,output_file=output_file,elev=None):
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
    """
    #create offset grid using pixel size and max UV diagonal distance
    x_m,y_m,z_m = simulating.get_core_coordinates(flagged_antennas)
    U,V,W = simulating.compute_uvw(x_m,y_m,z_m,0,Dec)
    pixel_resolution = (0.20 / uv_diag) / 3
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2

    offset_grid = np.arange(-1/uv_resolution/2,1/uv_resolution/2,1/uv_max/2)*180/np.pi
    #print(len(offset_grid) , image_size)
    assert(len(offset_grid) == image_size)
    """
    #find RA, dec at center of the image 

    #(1) ovro location
    loc = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg) #default is ovro

    #(2) observation time
    tobs = Time(mjd_obs,format='mjd')
    tms = int(tobs.unix*1000) #ms

    if elev is None:
        #(3) query antenna elevation at obs time
        result = dict()
        tries = 0
        while len(result) == 0 and tries < maxtries:
            query = f'SELECT time,ant_el FROM "antmon" WHERE time >= {tms-timerangems}ms and time < {tms+timerangems}ms'
            result = influx.query(query)
            tries += 1
        if len(result) == 0:
            print("Failed to retrieve elevation, using RA,DEC = 0,0",file=fout)
            icrs_pos = ICRS(ra=0*u.deg,dec=0*u.deg)
        else:
            #bestidx = np.argmin(np.abs(tobs.mjd - Time(np.array(result['antmon'].index),format='datetime').mjd))
            elev = np.nanmedian(result['antmon']['ant_el'].values)#[bestidx]

            #convert to RA,DEC using dsa110-pyutils.cli.radecel method; it can only be run from command line, so we copy/paste
            """ha = tobs.sidereal_time("apparent", Lon*u.deg)
            
            print(f'MJD, RA, Decl, Elev (deg): {mjd_obs}, {ha.to_value(u.deg)}, {elev+Lat-90}, {elev}')
            icrs_pos = ICRS(ra=(ha.to_value(u.deg))*u.deg,dec=(elev+Lat-90)*u.deg)
            """
            alt,az = DSAelev_to_ASTROPYalt(elev)
            print("Retrieved elevation: " + str(elev) + "deg",file=fout)

            antpos = AltAz(obstime=tobs,location=loc,az=az*u.deg,alt=alt*u.deg)

            #(4) convert to ICRS frame
            icrs_pos = antpos.transform_to(ICRS())
            
    else:
        print("Using input elevation: " + str(elev) + "deg",file=fout)
        """
        icrs_pos = ICRS(ra=(ha.to_value(u.deg))*u.deg,dec=(elev+Lat-90)*u.deg)
        """
        alt,az = DSAelev_to_ASTROPYalt(elev)
        print("Using input elevation: " + str(elev) + "deg",file=fout)
    
        antpos = AltAz(obstime=tobs,location=loc,az=az*u.deg,alt=alt*u.deg)

        #(4) convert to ICRS frame
        icrs_pos = antpos.transform_to(ICRS())
        
    print("Retrieved Coordinates: " + str(tobs.isot) + ", RA="+str(icrs_pos.ra.value) + "deg, DEC="+str(icrs_pos.dec.value) + "deg",file=fout)


    #create offset grid using pixel size and max UV diagonal distance
    x_m,y_m,z_m = simulating.get_core_coordinates(flagged_antennas) #meters
    U,V,W = simulating.compute_uvw(x_m,y_m,z_m,0,icrs_pos.dec.value*np.pi/180) #meters
    uv_diag = np.max(np.sqrt(U**2 + V**2)) #meters
    pixel_resolution = (0.20 / uv_diag) / 3 #radians
    offset_grid = np.arange(-image_size//2,image_size//2)*pixel_resolution*180/np.pi #degrees

    #print(offset_grid)
    assert(len(offset_grid) == image_size)

    #add offset from image center
    ra_grid = icrs_pos.ra.value + offset_grid
    dec_grid = icrs_pos.dec.value + offset_grid
    if output_file != "":
        fout.close()
    return ra_grid,dec_grid
