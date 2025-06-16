import numpy as np
from dsacalib import constants as ct
import glob
import json
from scipy.optimize import curve_fit
from dsamfs import utils as pu
from dsacalib.utils import Direction
from dsautils.coordinates import create_WCS,get_declination,get_elevation
#from nsfrb.outputlogging import printlog
from scipy.interpolate import interp1d
from astropy import wcs
from scipy.fftpack import ifftshift, ifft2,fftshift,fft2,fftfreq
from nsfrb.config import IMAGE_SIZE,UVMAX,flagged_antennas,crpix_dict,pixperFWHM,lambdaref,az_offset,Lon,Lat,Height
#modules for position and RA/DEC calibration
from influxdb import DataFrameClient
from astropy.coordinates import EarthLocation, AltAz, ICRS,SkyCoord,FK5
import astropy.units as u
from astropy.time import Time
import sys
from matplotlib import pyplot as plt
from nsfrb import simulating#,planning
import copy
import numba
from nsfrb.flagging import flag_vis

#flagged_antennas = [21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
import os
#from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,Lon,Lat,az_offset,Height,flagged_antennas,flagged_corrs,T,pixsize,table_dir
"""
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
output_file = cwd + "-logfiles/run_log.txt"
"""



def stack_images(imgs,cutoff_offsets,ref_RA_grid=None,ref_DEC_grid=None):
    """
    Given a list of images and cutoffs, aligns them. Cutoffs are estimated
    from astrometric cal fits or from comparing drift rate to pixel size
    using get_RA_cutoff()
    """
    cutoff_offsets = -np.array(cutoff_offsets)
    #if not already, reference cutoffs to the first img 
    assert(len(imgs)==len(cutoff_offsets))
    #assert(imgs[0].shape[:2] == ref_RA_grid.shape)
    assert(cutoff_offsets[0] == 0)
    
    #use maximum cutoff to get a new min_gridsize
    image_size = imgs[0].shape[0]
    min_gridsize = imgs[0].shape[1]
    ngulps = len(cutoff_offsets)
    min_gridsize_new = min_gridsize - np.nanmax(np.abs(cutoff_offsets))
    peakoffsetidx = np.nanargmax(np.abs(cutoff_offsets))
    peakoffset = cutoff_offsets[peakoffsetidx]
    new_shape = list(imgs[0].shape)
    new_shape[1] = min_gridsize_new
    #full_img_new = [np.zeros(new_shape)]*len(imgs)
    full_img_new = []

    #align the reference and peak offset images first
    ra_grid_2D = None
    dec_grid_2D = None
    if peakoffset>=0:
        full_img_new.append(imgs[0][:,np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new,...])
        if ref_RA_grid is not None:
            if len(np.array(ref_RA_grid).shape) == 2:
                ra_grid_2D = ref_RA_grid[:,np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new]
            elif len(np.array(ref_RA_grid).shape) == 1:
                ra_grid_2D = ref_RA_grid[np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new]
            else:
                print("Invalid ra grid shape")
                ra_grid_2D = None
        if ref_DEC_grid is not None:
            if len(np.array(ref_DEC_grid).shape) == 2:
                dec_grid_2D = ref_DEC_grid[:,np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new]
            elif len(np.array(ref_DEC_grid).shape) == 1:
                dec_grid_2D = ref_DEC_grid
            else:
                print("Invalid dec grid shape")
                dec_grid_2D = None
    else:
        full_img_new.append(imgs[0][:,(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset)),...])
        if ref_RA_grid is not None:
            if len(np.array(ref_RA_grid).shape) == 2:
                ra_grid_2D = ref_RA_grid[:,(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset))]
            elif len(np.array(ref_RA_grid).shape) == 1:
                ra_grid_2D = ref_RA_grid[(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset))]
            else:
                print("Invalid ra grid shape")
                ra_grid_2D = None
        if ref_DEC_grid is not None:
            if len(np.array(ref_DEC_grid).shape) == 2:
                dec_grid_2D = ref_DEC_grid[:,(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset))]
            elif len(np.array(ref_DEC_grid).shape) == 1:
                dec_grid_2D = ref_DEC_grid
            else:
                print("Invalid dec grid shape")
                dec_grid_2D = None

        
    #now the rest
    for g in range(1,ngulps):
        if g != peakoffsetidx:
            if peakoffset>=0 and cutoff_offsets[g]>=0:
                print(g,"case 1",peakoffsetidx,cutoff_offsets[g])
                full_img_new.append(imgs[g][:,np.abs(peakoffset)-np.abs(cutoff_offsets[g]):np.abs(peakoffset)-np.abs(cutoff_offsets[g])+min_gridsize_new,...])
            elif peakoffset<0 and cutoff_offsets[g]<0:
                print(g,"case 2",peakoffsetidx,cutoff_offsets[g])
                full_img_new.append(imgs[g][:,(min_gridsize - np.abs(peakoffset) + np.abs(cutoff_offsets[g]) - min_gridsize_new):(min_gridsize - np.abs(peakoffset) + np.abs(cutoff_offsets[g])),...])
            elif peakoffset>=0 and cutoff_offsets[g]<0:
                print(g,"case 3",peakoffsetidx,cutoff_offsets[g])
                tmp = np.zeros(new_shape)
                tmp[:,:min([np.abs(peakoffset)+np.abs(cutoff_offsets[g])+min_gridsize_new,min_gridsize])-(np.abs(peakoffset)+np.abs(cutoff_offsets[g]))] = imgs[g][:,np.abs(peakoffset)+np.abs(cutoff_offsets[g]):min([np.abs(peakoffset)+np.abs(cutoff_offsets[g])+min_gridsize_new,min_gridsize]),...]
                full_img_new.append(tmp)
            elif peakoffset<0 and cutoff_offsets[g]>=0:
                print(g,"case 4",peakoffsetidx,cutoff_offsets[g])
                tmp = np.zeros(new_shape)
                tmp[:,-(((min_gridsize - (np.abs(peakoffset)+np.abs(cutoff_offsets[g]))))-max([0,(min_gridsize - (np.abs(peakoffset)+np.abs(cutoff_offsets[g])) - min_gridsize_new)])):] = imgs[g][:,max([0,(min_gridsize - (np.abs(peakoffset)+np.abs(cutoff_offsets[g])) - min_gridsize_new)]):(min_gridsize - (np.abs(peakoffset)+np.abs(cutoff_offsets[g]))),...]
                full_img_new.append(tmp)
        else:
            if peakoffset>=0:
                full_img_new.append(imgs[peakoffsetidx][:,(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset)),...])
            else:
                full_img_new.append(imgs[peakoffsetidx][:,np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new,...])
    print(len(full_img_new),len(imgs))
    return full_img_new,ra_grid_2D,dec_grid_2D,min_gridsize_new


def briggs_weighting(u: np.ndarray, v: np.ndarray, grid_size: int, vis_weights: np.ndarray = None, robust: float = 0.0,pixel_resolution=None) -> np.ndarray:
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

    #u_indices = ((u + np.max(u)) / (2 * np.max(u)) * (grid_size - 1)).astype(int)
    #v_indices = ((v + np.max(v)) / (2 * np.max(v)) * (grid_size - 1)).astype(int)
    if pixel_resolution is None:
        pixel_resolution = (1 / np.max(np.sqrt(u ** 2 + v ** 2))) / 3 #radians if UV in meters
    #pixel_resolution= (1./60.)*(np.pi/180.)/2./0.2
    uv_resolution = 1 / (grid_size * pixel_resolution)
    uv_max = uv_resolution * grid_size / 2
    grid_res = 2 * uv_max / grid_size

    u_indices = ((u + uv_max) / grid_res).astype(int)
    v_indices = ((v + uv_max) / grid_res).astype(int)
    
    #uv_grid = np.bincount(u_indices * grid_size + v_indices, weights=vis_weights, minlength=grid_size**2)
    #print(np.any((u_indices * grid_size + v_indices) - np.min(u_indices * grid_size + v_indices)<0))
    #print(np.any(vis_weights<0))
    uv_grid = np.bincount((u_indices * grid_size + v_indices) - np.min(u_indices * grid_size + v_indices), weights=vis_weights, minlength=grid_size**2)
    Wk = uv_grid.flatten()

    f2 = (5 * 10 ** (-robust)) ** 2 / (np.sum(Wk ** 2) / np.sum(vis_weights))

    new_weights = vis_weights / (1 + Wk[u_indices * grid_size + v_indices- np.min(u_indices * grid_size + v_indices)] * f2)

    return new_weights/np.nansum(new_weights)


def uniform_grid(u, v, image_size, pixel_resolution, pixperFWHM):
    """
    Uniform gridding, returns indices
    """
    if pixel_resolution is None:
        pixel_resolution = (1 / np.max(np.sqrt(u ** 2 + v ** 2))) / pixperFWHM #radians if UV in meters
    #pixel_resolution= (1./60.)*(np.pi/180.)/2./0.2
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    #removed clip
    i_indices = ((u + uv_max) / grid_res).astype(int)
    j_indices = ((v + uv_max) / grid_res).astype(int)

    """
    #remove long baselines
    uvs = np.sqrt(u**2 + v**2)
    print(i_indices.shape)
    i_indices = i_indices[uvs<uv_max]
    j_indices = j_indices[uvs<uv_max]
    print(j_indices.shape)
    """
    #get conjugate baselines
    i_conj_indices = image_size - i_indices - 1
    j_conj_indices = image_size - j_indices - 1
    
    return (i_indices,j_indices,i_conj_indices,j_conj_indices)

def revised_robust_image(chunk_V: np.ndarray, u: np.ndarray, v: np.ndarray, image_size: int,  robust: float = 0.0, uniform=False, return_complex=False, inject_img=None, inject_flat=False, pixel_resolution=None, wstack=False, w=None, Nlayers_w=18,pixperFWHM=pixperFWHM, briggs_weights=None,i_indices=None,j_indices=None,i_conj_indices=None,j_conj_indices=None,clipuv=True,keeptime=False, wstack_parallel=False) -> np.ndarray:
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
    if pixel_resolution is None:
        pixel_resolution = (1 / np.max(np.sqrt(u ** 2 + v ** 2))) / pixperFWHM #radians if UV in meters
    #pixel_resolution= (1./60.)*(np.pi/180.)/2./0.2
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    if not uniform:
        #briggs weighting
        if briggs_weights is None:
            briggs_weights = briggs_weighting(u, v, image_size, robust=robust,pixel_resolution=pixel_resolution)
        #print("INPUT VIS SHAPE",chunk_V.shape,briggs_weights.shape)
        if keeptime: v_avg = chunk_V * briggs_weights * image_size #normalize since ifft has a 1/n term
        else: v_avg = np.mean(np.array(chunk_V * briggs_weights * image_size), axis=0)
    else:
        if keeptime: v_avg = chunk_V
        else: v_avg = np.mean(np.array(chunk_V), axis=0)
    
    if clipuv: v_avg = v_avg[np.sqrt(u**2 + v**2)<uv_max]
    #print("VIS SHAPE",v_avg.shape)
    if i_indices is None and j_indices is None:
        #removed clip
        i_indices = ((u + uv_max) / grid_res).astype(int)
        j_indices = ((v + uv_max) / grid_res).astype(int)
        
        #remove long baselines
        uvs = np.sqrt(u**2 + v**2)
        #v_avg = v_avg[uvs<uv_max]
        #i_indices = i_indices[uvs<uv_max]
        #j_indices = j_indices[uvs<uv_max]
        if i_conj_indices is None and j_conj_indices is None:
            #get conjugate baselines
            i_conj_indices = image_size - i_indices - 1
            j_conj_indices = image_size - j_indices - 1
    #$print(v_avg.shape,i_indices.shape,j_indices.shape,i_conj_indices.shape,j_conj_indices.shape)
    if keeptime:
        visibility_grid = np.zeros((v_avg.shape[0],image_size, image_size), dtype=complex)
        for i in range(v_avg.shape[0]):
            visibility_grid_i = np.zeros((image_size, image_size), dtype=complex)
            nancondition = ~np.isnan(v_avg[i,:])
            np.add.at(visibility_grid_i, (np.concatenate([i_indices[nancondition],i_conj_indices[nancondition]]),
                                    np.concatenate([j_indices[nancondition],j_conj_indices[nancondition]])),
                                    np.concatenate([v_avg[i,nancondition],np.conj(v_avg[i,nancondition])]))
            visibility_grid[i,:,:] = visibility_grid_i

    else:
        visibility_grid = np.zeros((image_size, image_size), dtype=complex)
        nancondition = ~np.isnan(v_avg)
        np.add.at(visibility_grid, (np.concatenate([i_indices[nancondition],i_conj_indices[nancondition]]),
                                    np.concatenate([j_indices[nancondition],j_conj_indices[nancondition]])),
                                    np.concatenate([v_avg[nancondition],np.conj(v_avg[nancondition])]))

    if inject_img is not None:
        #print("IN THE WRONG PLACE")
        if keeptime:            
            assert(v_avg.shape[0] == inject_img.shape[2])
            for i in range(v_avg.shape[0]):
                if inject_flat:
                    visibility_grid[i,i_indices,j_indices] += inverse_revised_uniform_image(inject_img[:,:,i],u,v,pixperFWHM=pixperFWHM)[i_indices,j_indices]
                    visibility_grid[i,i_conj_indices,j_conj_indices] += inverse_revised_uniform_image(inject_img[:,:,i],u,v,pixperFWHM=pixperFWHM)[i_conj_indices,j_conj_indices]
                else:
                    visibility_grid[i,:,:] += inverse_revised_uniform_image(inject_img[:,:,i],u,v,pixperFWHM=pixperFWHM)
        else:
            if inject_flat:
                visibility_grid[i_indices,j_indices] += inverse_revised_uniform_image(inject_img,u,v,pixperFWHM=pixperFWHM)[i_indices,j_indices]
                visibility_grid[i_conj_indices,j_conj_indices] += inverse_revised_uniform_image(inject_img,u,v,pixperFWHM=pixperFWHM)[i_conj_indices,j_conj_indices]
            else:
                visibility_grid += inverse_revised_uniform_image(inject_img,u,v,pixperFWHM=pixperFWHM)

    #updated sign convention
    if keeptime:
        dirty_image = ifftshift(ifft2(ifftshift(visibility_grid,axes=(1,2)),axes=(1,2)),axes=(1,2))
        if wstack and w is not None:
            if wstack_parallel:
                w_min = -np.max(np.abs(w))
                w_max = np.max(np.abs(w))
                w_grid_res = (w_max+1-w_min)/Nlayers_w
                w_bins = np.linspace(w_min,w_max+1,Nlayers_w)
                l_grid_2D,m_grid_2D = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(image_size,grid_res))[::-1],np.fft.fftshift(np.fft.fftfreq(image_size,grid_res)))
            for i in range(dirty_image.shape[0]):
                if wstack_parallel:
                    dirty_image[i,:,:] = process_w_layers_parallel(dirty_image[i,:,:,np.newaxis].repeat(Nlayers_w,2),w_bins,l_grid_2D,m_grid_2D,w_min,w_max)
                else:
                    dirty_image[i,:,:] = process_w_layers(dirty_image[i,:,:],w,Nlayers_w)
    else:
        dirty_image = ifftshift(ifft2(ifftshift(visibility_grid)))
        if wstack and w is not None:
            if wstack_parallel:
                w_min = -np.max(np.abs(w))
                w_max = np.max(np.abs(w))
                w_grid_res = (w_max+1-w_min)/Nlayers_w
                w_bins = np.linspace(w_min,w_max+1,Nlayers_w)
                l_grid_2D,m_grid_2D = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(image_size,grid_res))[::-1],np.fft.fftshift(np.fft.fftfreq(image_size,grid_res)))
                dirty_image = process_w_layers_parallel(dirty_image[:,:,np.newaxis].repeat(Nlayers_w,2),w_bins,l_grid_2D,m_grid_2D,w_min,w_max)
            else:
                dirty_image = process_w_layers(dirty_image,w,Nlayers_w)
    if keeptime:
        return np.real(dirty_image).transpose((1,2,0)) if not return_complex else dirty_image.transpose((1,2,0)) #DEC,RA,TIME
    else:
        return np.real(dirty_image) if not return_complex else dirty_image
    

def inverse_revised_uniform_image(dirty_image,u,v,pixperFWHM=pixperFWHM):
    """
    Inverse of uniform_image, used for injection purposes; inverts image to get gridded visibilities

    Parameters:
    dirty_image: Dirty image with shape (gridsize,gridsize)
    pixel_resolution: image pixel size in degrees (?)
    u,v: Coordinates in UV plane at which visibilities should be returned; the nearest grid point will be used
    """

    image_size = dirty_image.shape[0]
    pixel_resolution = (0.20 / np.max(np.sqrt(u ** 2 + v ** 2))) / pixperFWHM
    uv_resolution = 1 / (image_size * pixel_resolution)
    uv_max = uv_resolution * image_size / 2
    grid_res = 2 * uv_max / image_size

    visibility_grid = fftshift(fft2(fftshift(dirty_image.transpose())))


    """
    #get nearest visibility grid point for each u,v
    i_indices = np.clip((u + uv_max) / grid_res, 0, image_size - 1).astype(int)
    j_indices = np.clip((v + uv_max) / grid_res, 0, image_size - 1).astype(int)
    #count_indices = np.array([np.sum(np.logical_and(i_indices==i_indices[k],j_indices==j_indices[k])) for k in range(len(i_indices))])
    chunk_V = visibility_grid[i_indices,j_indices]#/count_indices
    """
    return visibility_grid


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

def ASTROPYalt_to_DSAelev(alt,az):
    alt = np.array(alt)
    az = np.array(np.around(az,0))
    elev = copy.deepcopy(alt)

    #if az is 360, need to shift by 180
    elev[az==360] = 180 - elev[az==360]
    return elev

#credit: Vikram Ravi
def get_ra(mjd,dec,Lon=Lon,Lat=Lat,Height=Height):
    """
    Gets RA, given the MJD and declination pointing
    """
    
    RA_rad,DEC_rad = Direction('HADEC',
            0.,
            (dec*u.deg).to_value(u.rad),
             mjd).J2000()
    if RA_rad < 0:
        RA_rad = 2*np.pi + RA_rad
    RA = (RA_rad*u.rad).to(u.deg)
    DEC = (DEC_rad*u.rad).to(u.deg)
    return RA.value

    #return Time(mjd,format='mjd').sidereal_time("apparent",longitude=Lon*u.deg).to(u.deg).value
    
    
    """
    ovro =  EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg,height=Height*u.m)#(lat=37.2317 * u.deg, lon=-118.2951 * u.deg, height=1222 * u.m)
    time = Time(mjd,format='mjd')
    if dec<Lat:#37.23:
        az=180.0*u.deg
        alt=(90.-(Lat-dec))*u.deg#37.23-dec))*u.deg
    else:
        az=0.0*u.deg
        alt=(90.-(dec-Lat))*u.deg#37.23))*u.deg
    altaz = SkyCoord(alt=alt,az=az,frame = 'altaz',obstime=time,location=ovro)
    return altaz.icrs.ra.deg
    """

# new position implementation - derived directly to convert from l,m to sky coords

def dec_to_m(dec0,dec_offset,d=1,Lat=Lat):
    m= np.sqrt((2*d*np.sin(dec_offset*np.pi/180/2))**2 - 
                   (d*np.sin((dec0+dec_offset)*np.pi/180)*(1 - np.cos((dec0-Lat)*np.pi/180)/np.cos((dec0-Lat+dec_offset)*np.pi/180)))**2)
    m[dec0+dec_offset==0] = np.sqrt((2*d*np.sin(dec_offset*np.pi/180/2))**2)[dec0+dec_offset==0]
    return m

    
#revision of uv_to_pix to be consistent with FRB search code
influx = DataFrameClient('influxdbservice.pro.pvt', 8086, 'root', 'root', 'dsa110')
def uv_to_pix(mjd_obs,image_size,Lat=Lat,Lon=Lon,Height=Height,timerangems=1000,maxtries=5,output_file="",elev=None,RA=None,DEC=None,flagged_antennas=flagged_antennas,uv_diag=None,az=az_offset,ref_wav=0.20,fl=False,two_dim=False,manual=False,manual_RA_offset=0,pixperFWHM=pixperFWHM):
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
    obstime = Time(mjd_obs,format='mjd')
    if DEC is None and elev is None:
        elev = get_elevation(obstime)
        DEC  = get_declination(elev).value
    
    #get ra, dec assuming phase centered on meridian
    RA_rad,DEC_rad = Direction('HADEC',
            0.,
            (DEC*u.deg).to_value(u.rad),
            obstime.mjd).J2000()
    RA = (RA_rad*u.rad).to(u.deg)
    DEC = (DEC_rad*u.rad).to(u.deg)
    pointing = SkyCoord(ra=RA,dec=DEC,frame='icrs')

    """
    if RA is None:
        RA = obstime.sidereal_time('apparent', longitude=Lon*u.deg).to(u.deg).value
    pointing = SkyCoord(ra=RA*u.deg,dec=DEC*u.deg,frame=FK5,equinox=obstime).transform_to(ICRS)
    """
    #printlog(f'Primary beam pointing: {pointing}',output_file=output_file)
    

    if uv_diag is None:
        test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
        pt_dec = DEC.value*np.pi/180.
        bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
        tmp, bname, blen, UVW, antenna_order = flag_vis(np.zeros((1,4656,8*16,2,2)), bname, blen, UVW, antenna_order, flagged_antennas, bmin=0,flagged_corrs=[],flag_channel_templates=[])

        U = UVW[0,:,0]
        V = UVW[0,:,1]
       

        uv_diag = np.max(np.sqrt(U**2 + V**2)) #meters
        #x_m,y_m,z_m = simulating.get_all_coordinates(flagged_antennas) #meters
        #U,V,W = simulating.compute_uvw(x_m,y_m,z_m,0,DEC*np.pi/180) #meters
        #uv_diag = np.max(np.sqrt(U**2 + V**2)) #meters
    pixel_resolution = (ref_wav / uv_diag) / pixperFWHM
    w2 = create_WCS(pointing,-pixel_resolution*u.rad,image_size)
    
    #get axes
    if two_dim:
        dec_grid_pix_2D,ra_grid_pix_2D = np.meshgrid(np.arange(image_size,dtype=float),np.arange(image_size,dtype=float))
        tmp = w2.wcs_pix2world(np.array([ra_grid_pix_2D.flatten(),
                                     dec_grid_pix_2D.flatten()]).transpose(),0)
        ra_grid = tmp[:,0].reshape((image_size,image_size)).transpose()
        dec_grid = tmp[:,1].reshape((image_size,image_size)).transpose()

    else:
        dec_grid_pix,ra_grid_pix = np.arange(image_size,dtype=float),np.arange(image_size,dtype=float)
        tmp = w2.wcs_pix2world(np.array([ra_grid_pix,dec_grid_pix]).transpose(),0)
        ra_grid = tmp[:,0]
        dec_grid =tmp[:,1]
    #printlog("RADECSHAPE:" + str(ra_grid.shape) + "," + str(dec_grid.shape),output_file=output_file)
    return ra_grid,dec_grid,elev

def process_w_layers(dirty_image,w,Nlayers_w):
    """
    Basic w-stacking without parallel processing
    dirty_image: (image_size, image_size) complex array
    w: w coordinates of visibilities used to form dirty_image
    Nlayers_w: number of W-layers
    """
    image_size = dirty_image.shape[0]
    w_min = -np.max(np.abs(w))
    w_max = np.max(np.abs(w))
    w_grid_res = (w_max+1-w_min)/Nlayers_w
    w_bins = np.linspace(w_min,w_max+1,Nlayers_w)
    binned_ws = w_bins[((w - w_min) / w_grid_res).astype(int)]
    l_grid,m_grid = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(image_size,grid_res))[::-1],np.fft.fftshift(np.fft.fftfreq(image_size,grid_res)))
    final_img = np.zeros_like(dirty_image)
    for w in w_bins:
        final_img += (dirty_img*np.sum(binned_ws==w)*np.exp(2*np.pi*1j*w*(np.sqrt(1-l_grid**2-m_grid**2) - 1))*np.sqrt(1 - l_grid**2 - m_grid**2)/(w_max-w_min))
    return final_img

@numba.njit(parallel=True)
def process_w_layers_parallel(dirty_image: np.ndarray, 
                              w_bins: np.ndarray, 
                              l_grid_2D: np.ndarray, 
                              m_grid_2D: np.ndarray, 
                              w_min: float, 
                              w_max: float) -> np.ndarray:
    """
    Numba-accelerated function to process w-layers in parallel.
    dirty_image: (image_size, image_size, Nlayers_w) complex array
    w_bins: (Nlayers_w,) array
    l_grid_2D, m_grid_2D: (image_size, image_size) arrays
    w_min, w_max: floats
    """
    image_size = dirty_image.shape[0]
    Nlayers_w = dirty_image.shape[2]
    out_image = np.zeros((image_size, image_size), dtype=np.complex128)

    scale_factor = (w_max - w_min)
    for i in numba.prange(image_size):
        for j in numba.prange(image_size):
            l_val = l_grid_2D[i, j]
            m_val = m_grid_2D[i, j]
            val_sqrt = 1.0 - l_val**2 - m_val**2
            if val_sqrt <= 0:
                # If invalid geometry, skip
                continue
            phase_factor = np.sqrt(val_sqrt)
            # sum over w
            temp_sum = 0.0j
            for k in range(Nlayers_w):
                w_val = w_bins[k]
                exponent = 2 * np.pi * 1j * w_val * (phase_factor - 1)
                # dirty_image[i,j,k]
                temp_sum += dirty_image[i, j, k] * np.exp(exponent) * phase_factor / scale_factor
            out_image[i, j] = temp_sum

    return out_image


def single_pix_image(dat_all,U,V,fobs,sb,dec,mjd,ngulps,nbin,target_coord,tsamp_use,pixel_resolution,pixperFWHM,uv_diag,nchans_per_node=8,gulpsize=25,image_size=301,allpix=[],DM=0,robust=-2):
    """
    robust single pixel imaging given array of visibilities (nsamp x nbase x nchan x npol)
    """
    yidxs,xidxs = np.meshgrid(np.arange(image_size,dtype=int),np.arange(image_size,dtype=int))
    nchan_per_node = nchans_per_node
    dspec = np.zeros((gulpsize*ngulps//nbin,nchan_per_node))
    #uv_diag=np.max(np.sqrt(U**2 + V**2))
    #pixel_resolution = (lambdaref / uv_diag) / pixperFWHM
    j = sb

    #gridding
    U_wavs = np.zeros((len(U),nchans_per_node))
    V_wavs = np.zeros((len(V),nchans_per_node))
    #W_wavs = np.zeros((len(W),nchans_per_node))
    i_indices_all = np.zeros(U_wavs.shape,dtype=int)
    j_indices_all = np.zeros(V_wavs.shape,dtype=int)
    #k_indices_all = np.zeros(W_wavs.shape,dtype=int)
    i_conj_indices_all = np.zeros(U_wavs.shape,dtype=int)
    j_conj_indices_all = np.zeros(V_wavs.shape,dtype=int)
    #k_conj_indices_all = np.zeros(W_wavs.shape,dtype=int)
    bweights_all = np.zeros(U_wavs.shape)
    
    for jj in range(nchans_per_node):
        chanidx = (nchans_per_node*j)+jj
        U_wavs[:,jj] = U/(ct.C_GHZ_M/fobs[chanidx])
        V_wavs[:,jj] = V/(ct.C_GHZ_M/fobs[chanidx])
        i_indices_all[:,jj],j_indices_all[:,jj],i_conj_indices_all[:,jj],j_conj_indices_all[:,jj] = uniform_grid(U_wavs[:,jj], V_wavs[:,jj], image_size, pixel_resolution, pixperFWHM)
        bweights_all[:,jj] = briggs_weighting(U_wavs[:,jj], V_wavs[:,jj], image_size, robust=robust,pixel_resolution=pixel_resolution)
    
    for gulp in range(ngulps):
  
        #get UVWs
        dat = dat_all[gulpsize*gulp:gulpsize*(gulp+1),:,:,:]
        if len(allpix)<=gulp:
            #make RA,DEC grid
            #if j == 0:
            ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + (gulp*gulpsize*tsamp_use/1000/86400),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)
            target_pix = np.unravel_index(np.argmin(target_coord.separation(SkyCoord(ra=ra_grid_2D*u.deg,dec=dec_grid_2D*u.deg,frame='icrs'))),ra_grid_2D.shape)
            print("targeting coord",target_coord,target_pix)
            allpix.append(target_pix)
            del ra_grid_2D
            del dec_grid_2D
        else:
            target_pix = allpix[gulp]
            
        #make dynamic spectrum for single coord
        uv_max = 1/(2*pixel_resolution)
        for ii in range(gulpsize//nbin):
            for jj in range(nchans_per_node):
                chanidx = (nchans_per_node*j)+jj
                vis_grid = np.zeros((image_size,image_size),dtype=complex)
                v_avg = np.nanmean(dat_all[gulpsize*gulp + (ii*nbin):gulpsize*gulp + (ii+1)*nbin,:,jj,:],(0,2))*bweights_all[:,jj]*image_size
                nancondition = ~np.isnan(v_avg)#,np.sqrt(U_wavs[:,jj]**2 + V_wavs[:,jj]**2)<uv_max)
                #for i in range(dat.shape[0]):
                np.add.at(vis_grid, (np.concatenate([i_indices_all[nancondition,jj],i_conj_indices_all[nancondition,jj]]),
                                 np.concatenate([j_indices_all[nancondition,jj],j_conj_indices_all[nancondition,jj]])),
                                 np.concatenate([v_avg[nancondition],np.conj(v_avg[nancondition])]))
        
                #beam/image
                dspec[(gulp*gulpsize//nbin) + ii,jj] = np.real(np.nansum(ifftshift(vis_grid)*np.exp(1j*2*np.pi*(1/image_size)*(((target_pix[1]+(image_size//2))*yidxs) + ((target_pix[0]+(image_size//2))*xidxs)))))/(image_size*image_size)
        dspec[(gulp*gulpsize//nbin):((gulp+1)*gulpsize//nbin),:] -= np.nanmedian(dspec[(gulp*gulpsize//nbin):((gulp+1)*gulpsize//nbin),:],0)
    if DM>0:
        final_dspec = np.zeros_like(dspec)

        tshift =np.array(np.abs((4.15)*DM*((1/np.nanmin(fobs[nchans_per_node*j:(j+1)*nchans_per_node]))**2 - (1/fobs[nchans_per_node*j:(j+1)*nchans_per_node])**2))//tsamp_use,dtype=int)
        for jj in range(nchans_per_node):
            final_dspec[:,jj] = np.pad(dspec[:,jj],((0,tshift[jj])),mode='constant')[-final_dspec.shape[0]:]
        return final_dspec,allpix
    else:
        return dspec,allpix

