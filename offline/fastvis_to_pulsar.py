import numpy as np
from nsfrb.imaging import DSAelev_to_ASTROPYalt
from dsautils.coordinates import create_WCS,get_declination,get_elevation
import argparse
import csv
from matplotlib import pyplot as plt
import os
from scipy.fftpack import ifftshift, ifft2,fftshift,fft2,fftfreq
from nsfrb.config import IMAGE_SIZE,UVMAX
#modules for position and RA/DEC calibration
from influxdb import DataFrameClient
from astropy.coordinates import EarthLocation, AltAz, ICRS,SkyCoord
import astropy.units as u
from astropy.time import Time
import sys
from matplotlib import pyplot as plt
from antpos.utils import get_itrf

#**from vikram's code**
import numpy as np, matplotlib.pyplot as plt
import struct
import os, glob
from scipy.fftpack import ifftshift, ifft2, fftshift, fft2
import matplotlib.animation as animation
from dsamfs import utils as pu
import sys
from astropy.io import fits
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from influxdb import DataFrameClient
from nsfrb.imaging import get_ra
from nsfrb.planning import nvss_cat,atnf_cat,LPT_cat
#from nsfrb.plotting import make_image_from_vis
from nsfrb.planning import find_fast_vis_label
    
# create a direction object using dsacalib Direction object

from nsfrb.imaging import get_ra,revised_robust_image,revised_uniform_image
from astropy.coordinates import FK5,GCRS
from nsfrb.imaging import get_ra,uv_to_pix,uv_to_pix_manual
import copy
from nsfrb.flagging import flag_vis,fct_BPASS, fct_BPASSBURST

from matplotlib.patches import Ellipse
from nsfrb import pipeline
from matplotlib import animation
from astropy import units 
from nsfrb.config import Lon,NUM_CHANNELS,flagged_antennas,flagged_corrs
from dsamfs import utils as pu
from nsfrb.imaging import revised_uniform_image,revised_uniform_image_parallel,uv_to_pix
from nsfrb import imaging
from nsfrb.config import tsamp as tsamp_ms
from nsfrb.config import nsamps,nchans,bmin
import pickle as pkl
from nsfrb import searching

from scipy.stats import norm
from nsfrb.config import bad_antennas,flagged_antennas
from nsfrb import config
from astropy.coordinates import SkyCoord,EarthLocation
dsaloc = EarthLocation(lon=config.Lon*u.deg,lat=config.Lat*u.deg,height=config.Height*u.m)

"""
This script is used to form 5-minute dynamic spectra from fast visibility files in order to search for pulsars,
either with a custom search or with PRESTO.
"""

sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]




#function to convert npy to psrfits
from pdat import pdat
def set_primary_header(psrfits_object,prim_dict):
    
    #prim_dict = dictionary of primary header changes
    
    PF_obj = psrfits_object
    for key in prim_dict.keys():
        PF_obj.replace_FITS_Record('PRIMARY',key,prim_dict[key])

def set_subint_header(psrfits_object,subint_dict):
    #prim_dict = dictionary of primary header changes
    PF_obj = psrfits_object
    for key in subint_dict.keys():
        PF_obj.replace_FITS_Record('SUBINT',key,subint_dict[key])

def make_subint_BinTable(self):
    subint_draft = self.make_HDU_rec_array(self.nsubint, self.subint_dtype)
    return subint_draft

def numpy_to_psrfits(data,path,fnum,fobs,dec,mjd,sample_size=tsamp_ms,nsblk=25,BMIN=3.5,BMAJ=3.5):
    assert(data.shape[1]==len(fobs))
    assert(data.shape[0]%nsblk==0)
    #based on https://pulsardatatoolbox.readthedocs.io/en/latest/psrfits_write.html

    #create file from template
    fname = path + "/nsfrb_" +str(fnum)+"_pulsarsearch.fits"
    psrfits1=pdat.psrfits(fname,from_template="/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/search_template.fits")



    #make data params
    N_Time_Bins = data.shape[0]
    ROWS = int(N_Time_Bins//nsblk)
    Total_time = N_Time_Bins*sample_size
    dt = Total_time/N_Time_Bins
    subband =(fobs[0] - fobs[1])*1000
    BW=(((fobs[0] - fobs[1]) + fobs[0] - fobs[-1]))*1000
    N_freq = int(BW/subband)
    Npols = npol =  1
    print('Total_time',Total_time/1e3)
    print('N_freq',N_freq)

    #set sub-integration params
    np.bool = np.bool_
    psrfits1.set_subint_dims(nsblk=nsblk,nchan=N_freq,nsubint=ROWS,npol=Npols)
    subint_draft = psrfits1.make_HDU_rec_array(psrfits1.nsubint, psrfits1.subint_dtype)
    tsubint = nsblk*dt*1e-3 #in seconds
    offs_sub_init = tsubint/2
    offs_sub = np.zeros((ROWS))
    for jj in range(ROWS):
        offs_sub[jj] = offs_sub_init + (jj * tsubint)

    #get RA and galactic coords
    ra = imaging.get_ra(mjd,dec)
    coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs')
    gl,gb = coord.galactic.l.value,coord.galactic.b.value
    obstime = Time(mjd,format='mjd')
    elev = get_elevation(obstime).value
    alt = float(DSAelev_to_ASTROPYalt(elev)[0])

    ones = np.ones((ROWS))
    #And assign them using arrays of the appropriate sizes
    subint_draft['TSUBINT'] = tsubint * ones
    subint_draft['OFFS_SUB'] = offs_sub
    subint_draft['LST_SUB'] = ((ra*u.deg).to(u.hourangle).value)*86400 * ones
    subint_draft['RA_SUB'] = ra * ones
    subint_draft['DEC_SUB'] = dec * ones
    subint_draft['GLON_SUB'] = gl * ones
    subint_draft['GLAT_SUB'] = gb * ones
    subint_draft['FD_ANG'] = 0 * ones
    subint_draft['POS_ANG'] = 0 * ones
    subint_draft['PAR_ANG'] = 0 * ones
    subint_draft['TEL_AZ'] = 0 * ones
    subint_draft['TEL_ZEN'] = (90-alt) * ones

    #format image
    final_image_for_fits = data.transpose()[np.newaxis,np.newaxis,:,np.newaxis,:]
    final_image_for_fits = final_image_for_fits.reshape((ROWS,1,N_freq,1,nsblk))
    final_image_for_fits = final_image_for_fits*100/np.nanpercentile(final_image_for_fits,95)
    print(final_image_for_fits.shape)
    for ii in range(subint_draft.size):
        subint_draft[ii]['DATA'] = final_image_for_fits[ii,:,:,:,:]
        subint_draft[ii]['DAT_SCL'] = np.ones(N_freq*npol)
        subint_draft[ii]['DAT_OFFS'] = np.zeros(N_freq*npol)
        subint_draft[ii]['DAT_FREQ'] = fobs*1000 #np.linspace(fobs[-1],fobs[0],N_freq)
        subint_draft[ii]['DAT_WTS'] = np.ones(N_freq)

    #make primary headers
    subint_hdr=psrfits1.draft_hdrs['SUBINT']
    pri_dic= {'OBSERVER':'NSFRB','OBSFREQ':int(np.around(np.mean(fobs),1)*1000),'OBSBW':int(np.around(BW,0)),'OBSNCHAN':N_freq,'PROJID':'nsfrb','CHAN_DM':0,
         'TELESCOP':'DSA110','ANT_X':dsaloc.itrs.x.value,'ANT_Y':dsaloc.itrs.y.value,'ANT_Z':dsaloc.itrs.z.value,
         'BACKEND':'NSFRB','DATE-OBS':Time(mjd,format='mjd').isot,'SRC_NAME':'',
         'STT_CRD1':str(np.around(ra,4)),'STT_CRD2':str('+' if dec>0 else '-') + str(np.around(np.abs(dec),4)),
          'TRK_MODE':'SCANLAT',
          'BMIN':str(np.around(BMIN,0)),
          'BMAJ':str(np.around(BMAJ,0)),
         'STP_CRD1':str(np.around(ra,4)),'STP_CRD2':str('+' if dec>0 else '-') + str(np.around(np.abs(dec),4)),
         'SCANLEN':np.around(float(Total_time)/1000,5),
          'STT_IMJD':int(np.floor(mjd)),'STT_SMJD':int(np.floor((mjd - np.floor(mjd))*86400)),'STT_OFFS':0.99999999988222,
         'STT_LST':int(np.floor(86400*(ra/15)))}
    subint_dic = {'TBIN':str(dt/1000),'CHAN_BW':-subband}
    psrfits1.make_FITS_card(subint_hdr,'TBIN',subint_dic['TBIN'])
    
    set_primary_header(psrfits1,pri_dic)
    set_subint_header(psrfits1,subint_dic)
    psrfits1.HDU_drafts['SUBINT'] = subint_draft
    print("PRIMARY HEADER:")
    print(psrfits1.draft_hdrs['PRIMARY'])
    print("SUBINT HEADER:")
    print(psrfits1.draft_hdrs['SUBINT'])
    psrfits1.write_psrfits()
    psrfits1.close()
    print("Writing data to " + fname)
    return fname

def main(args):

    #make list of files in given path to image
    fnames = glob.glob(args.path + "/nsfrb_sb00*.out")
    fnums = np.sort(np.array([int(f[f.index("sb00") + 5:f.index(".out")]) for f in fnames]))
    print(fnums)

    #for each fnum, create dynamic spectrum
    nchans = len(corrs)
    nchans_per_node = args.nchans_per_node
    gridsize = image_size = args.gridsize
    NGULPS = args.gulps
    bmin=  args.bmin
    fdir = args.path + "/"

    dynspec = np.zeros((nsamps*NGULPS,nchans*nchans_per_node),dtype=np.float32)
    gulpsize = nsamps
    for fnum in fnums: 

        #first get ra and dec axes to get better pixel size estimate
        sb,mjd,dec = pipeline.read_raw_vis(args.path + "/nsfrb_sb00_" + str(fnum) + ".out",nchan=nchans_per_node,nsamps=nsamps,gulp=0,headersize=16,get_header=True)
        ra_grid_2D_,dec_grid_2D_,elev = uv_to_pix(mjd,gridsize,DEC=dec,two_dim=True,manual=False)
        pixelsize = np.abs(ra_grid_2D_[0,1]-ra_grid_2D_[0,0])

        #get RA cutoff
        racutoff_ = searching.get_RA_cutoff(dec,tsamp_ms*gulpsize,pixelsize)
        print("RA cutoff:",racutoff_,"pixels")
        min_gridsize = int(gridsize - racutoff_*(NGULPS - 1))
        i = 0
        ra_grid_2D = ra_grid_2D_[:,gridsize-min_gridsize - racutoff_*(NGULPS - 1 - i):gridsize-racutoff_*(NGULPS - 1 - i)]
        dec_grid_2D = dec_grid_2D_[:,gridsize-min_gridsize - racutoff_*(NGULPS - 1 - i):gridsize-racutoff_*(NGULPS - 1 - i)]
        BMIN = np.abs(np.max(ra_grid_2D) - np.min(ra_grid_2D))
        BMAJ = np.abs(np.max(dec_grid_2D) - np.min(dec_grid_2D))


        ff = 1.53-np.arange(8192)*0.25/8192
        fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
        fobs = np.reshape(fobs,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1)



    
        #check if it already exists
        if len(glob.glob(args.path + "_pulsarsearch/nsfrb_" + str(fnum) + "_pulsarsearch.npy"))>0 and not args.overwrite:
            print("Numpy file already exists")
            dynspec = np.load(args.path + "_pulsarsearch/nsfrb_" + str(fnum) + "_pulsarsearch.npy")
        else:
            for g in range(NGULPS):
                gulp = g

                #imaging
                dat = None
                for i in range(nchans):
                    try:
                        dat_i,sb,mjd,dec = pipeline.read_raw_vis(fdir + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchans_per_node,nsamps=gulpsize,gulp=gulp,headersize=16)
                        print(mjd,dec,sb)
    
                        if dat is None:
                            dat = np.nan*np.ones(dat_i.shape,dtype=dat_i.dtype).repeat(len(corrs),axis=2)
                        dat[:,:,i*nchans_per_node:(i+1)*nchans_per_node,:] = dat_i


                    except Exception as exc:
                        print(exc)



                print(dat.shape)
                test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
                pt_dec = dec*np.pi/180.
                bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
                dat, bname, blen, UVW, antenna_order = flag_vis(dat, bname, blen, UVW, antenna_order, flagged_antennas, bmin=bmin,flagged_corrs=flagged_corrs,flag_channel_templates=[])
                U = UVW[0,:,0]
                V = UVW[0,:,1]
                W = UVW[0,:,2]



                uv_diag=np.max(np.sqrt(U**2 + V**2))
                pixel_resolution = (0.20/uv_diag/3)
                #break
                ff = 1.53-np.arange(8192)*0.25/8192
                fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
                fobs = np.reshape(fobs,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1)


                dat[np.isnan(dat)] = 0
                tmpimg = np.zeros((gridsize,min_gridsize,nsamps,nchans*nchans_per_node),dtype=np.float32)
                for i in range(dat.shape[0]):
                    for j in range(len(corrs)):
                        for k in range(dat.shape[-1]):
                            for jj in range(nchans_per_node):
                                tmpimg[:,:,i,(j*nchans_per_node) + jj] += revised_robust_image(dat[i:i+1,:,(j*nchans_per_node) + jj,k],
                                               U/(2.998e8/fobs[(j*nchans_per_node) + jj]/1e9),
                                               V/(2.998e8/fobs[(j*nchans_per_node) + jj]/1e9),
                                               image_size,robust=-2)[:,gridsize-min_gridsize - racutoff_*(NGULPS - 1 - g):gridsize-racutoff_*(NGULPS - 1 - g)]
                        

                dynspec[(g*gulpsize):((g+1)*gulpsize),:] = np.nansum(tmpimg - np.nanmedian(tmpimg,axis=2,keepdims=True),axis=(0,1))

            #write to npy file
            os.system("mkdir " + args.path + "_pulsarsearch/")
            np.save(args.path + "_pulsarsearch/nsfrb_" + str(fnum) + "_pulsarsearch.npy",dynspec)
    
        #write to psrfits
        if len(glob.glob(args.path + "_pulsarsearch/nsfrb_" + str(fnum) + "_pulsarsearch.fits"))>0 and not args.overwrite:
            print("PSRFITS file already exists")
        else:
            numpy_to_psrfits(dynspec,args.path+"_pulsarsearch",fnum,fobs,dec,mjd,sample_size=tsamp_ms,nsblk=nsamps,BMIN=BMIN,BMAJ=BMAJ)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Images and averages fast visibilities to create periodicity search mode data')
    parser.add_argument('path')           # positional argument
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=2)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--gulps',type=int,help='Number of gulps to image, default = 90',default=90)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--overwrite',action='store_true',help='Overwrite existing files')
    args = parser.parse_args()
    main(args)
