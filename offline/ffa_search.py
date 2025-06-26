import time
from nsfrb import periodicity
import argparse
from dsamfs import fringestopping
import os
import numpy as np
import scipy  # noqa
import casatools as cc
import astropy.units as u
from dsacalib import constants as ct
from dsacalib.fringestopping import calc_uvw
import numba

from nsfrb.imaging import get_ra,stack_images,briggs_weighting
from nsfrb.planning import get_RA_cutoff
from matplotlib.patches import Ellipse
from nsfrb.config import *
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from scipy.fftpack import ifftshift, ifft2,fftshift,fft2,fftfreq
from nsfrb.config import IMAGE_SIZE,UVMAX,chanbw,lambdaref,freq_axis,freq_axis_fullres
from dsacalib import constants as ct

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

from nsfrb.imaging import get_ra,revised_robust_image
from astropy.coordinates import FK5,GCRS
from nsfrb.imaging import get_ra,uv_to_pix
import copy
from nsfrb.flagging import flag_vis,fct_BPASS, fct_BPASSBURST

from matplotlib.patches import Ellipse
from nsfrb import pipeline
from matplotlib import animation
from astropy import units
from nsfrb.config import Lon,NUM_CHANNELS,flagged_antennas,flagged_corrs
from dsamfs import utils as pu
from nsfrb import imaging
from nsfrb.config import tsamp as tsamp_ms
import pickle as pkl

from nsfrb.planning import read_nvss,read_RFC,read_vlac,read_atnf
from dsautils.coordinates import create_WCS,get_declination,get_elevation
from astropy.coordinates import AltAz
from nsfrb.planning import find_fast_vis_label
from nsfrb.planning import nvss_cat
from nsfrb.config import tsamp as tsamp_ms
from nsfrb.config import IMAGE_SIZE,bmin,flagged_antennas,bad_antennas,pixperFWHM,NUM_CHANNELS
import json


def main(args):

    print("Starting incoherent search of files "+str(args.fnum))
    mingulp=args.mingulp
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
    outriggers = args.outriggers
    nchans_per_node = nchan_per_node = args.nchans_per_node
    ngulps=args.ngulps
    gulpsize=args.gulpsize
    fnum = args.fnum
    sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
    corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
    fcts = []
    if args.flagSWAVE:
        fcts.append(fct_SWAVE)
    if args.flagBPASS:
        fcts.append(fct_BPASS)
    if args.flagFRCBAND:
        fcts.append(fct_FRCBAND)
    if args.flagBPASSBURST:
        fcts.append(fct_BPASSBURST)

    print(args.path)
    if args.path==vis_dir:
        datadirs = [vis_dir + "lxd110"+corrs[i]+"/" for i in range(len(corrs))]
    else:
        datadirs = [args.path + "/" for i in range(len(corrs))]

    dspec_incoherent = np.zeros((ngulps*gulpsize,nchans_per_node*16))
    for j in range(ngulps):
        print("Reading gulp ",j,"...")
        for i in range(16):
            print(">sb",i,"...",end="")
            if i ==0 :
                tmp_dat,sb,mjd,dec = pipeline.read_raw_vis(datadirs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=mingulp+j,headersize=16)
            else:
                tmp_dat = np.concatenate([tmp_dat,pipeline.read_raw_vis(datadirs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=mingulp+j,headersize=16)[0]],2)
    
        if j ==0:
            bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)

        dat_i, bname_, blen_, UVW_, antenna_order_ = flag_vis(tmp_dat, bname, blen, UVW, antenna_order, 
                                            list(bad_antennas if outriggers else flagged_antennas) + list(args.flagants), 
                                            bmin=args.bmin,flagged_corrs=list(flagged_corrs)+list(args.flagcorrs),flag_channel_templates=fcts,
                                            flagged_chans=list(args.flagchans))
    
        dspec_incoherent[j*gulpsize:(j+1)*gulpsize,:] = np.nanmean(np.abs(dat_i),(1,3))
        dspec_incoherent[j*gulpsize:(j+1)*gulpsize,:] -= np.nanmedian(dspec_incoherent[j*gulpsize:(j+1)*gulpsize,:],0)    
    
    tseries = np.nanmean(dspec_incoherent,1)
    tseries -= np.nanmedian(tseries)

    print("starting folding search...")
    foldoutput = periodicity.ffa_slow(tseries,args.periods)
    if not np.any(foldoutput>args.SNRthresh):
        print("No candidates found")
        return
    print("Trial periods:",args.periods)
    print("Signal-to-noise:",foldoutput)
    print("Best period:",args.periods[np.argmax(foldoutput)])
    bestP = args.periods[np.argmax(foldoutput)]
    pgram = tseries[:int((bestP*args.showprofiles)*(len(tseries)//(bestP*args.showprofiles)))].reshape((len(tseries)//(bestP*args.showprofiles),bestP*args.showprofiles))


    print("starting timing search...")
    finePtrials = np.arange(max([0,bestP//2]),bestP*2,dtype=int)
    resid = periodicity.ffa_timing(tseries,np.arange(max([0,bestP//2]),bestP*2,dtype=int))
    print("Trial periods:",finePtrials)
    print(np.min(resid,1))
    fineP = finePtrials[np.argmin(np.min(resid,1))]

    print("done")
    plt.figure(figsize=(16,6))
    plt.subplot(1,3,1)
    plt.imshow(dspec_incoherent.transpose(),aspect='auto',interpolation='none',vmin=0,vmax=50,
            extent=(0,len(tseries)*tsamp_ms/1000,np.nanmin(freq_axis),np.nanmax(freq_axis)))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.subplot(1,3,2)
    plt.imshow(pgram,vmin=0,vmax=30,aspect='auto',interpolation='none',
            extent=(0,2,0,bestP*args.showprofiles*pgram.shape[0]*tsamp_ms/1000),origin='lower')
    plt.xlabel("Phase (P="+str(bestP*tsamp_ms/1000)+"s)")
    plt.ylabel("Time (s)")
    plt.subplot(1,3,3)
    plt.imshow(resid,aspect='auto',
            extent=(0,1,finePtrials[0]*tsamp_ms/1000,finePtrials[-1]*tsamp_ms/1000))
    plt.xlabel("Phase")
    plt.ylabel("Trial Period (s)")
    plt.savefig(args.path + "/"+str(args.fnum) + "_periodicity.png")
    plt.close()
    
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_dspec.npy",dspec_incoherent)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_tseries.npy",tseries)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_pgram_P"+str(bestP)+".npy",pgram)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_resid.npy",resid)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_fine_Ptrials.npy",finePtrials)
    results = dict()
    results["BestFFAPeriod_s"] = bestP*tsamp_ms/1000
    results["BestTIMEPeriod_s"] = fineP*tsamp_ms/1000
    results["RA_point"] = get_ra(mjd,dec)
    results["DEC_point"] = dec
    print(args.path + "/"+str(args.fnum) + "_periodicity_results.json")
    print(results)
    f=open(args.path + "/"+str(args.fnum) + "_periodicity_results.json","w")
    json.dump(results,f)
    f.close()
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('fnum',type=str,help="Fast visibility file number")
    parser.add_argument('--path',type=str,help="Path to fast visibility files if not lxd110 dirs",default=vis_dir)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--flagSWAVE',action='store_true',help='Flag channels when SWAVE template RFI is detected, which manifests as a 2 Hz sin wave over ~5 minutes of data')
    parser.add_argument('--flagBPASS',action='store_true',help='Flag channels when BPASS template RFI is detected, which is simpl comparison to bandpass mean in visibilities')
    parser.add_argument('--flagFRCBAND',action='store_true',help='Flag channels in FRC miltiary allocation 1435-1525 MHz')
    parser.add_argument('--flagBPASSBURST',action='store_true',help='Flag channel when BPASS template RFI is detected in any timestep, i.e. should detect pulsed narrowband RFI')
    parser.add_argument('--flagcorrs',type=int,nargs='+',default=[],help='List of sb nodes [0-15] to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagants',type=int,nargs='+',default=[],help='List of antennas to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagchans',type=int,nargs='+',default=[],help='List of channels [0,(16*nchans_per_node - 1)] to flag')
    parser.add_argument('--outriggers',action='store_true',help='Includes outrigger antennas in imaging')
    parser.add_argument('--periods',nargs='+',type=int,help='periods (in samples) to search',default=[])
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per node',default=8)
    parser.add_argument('--mingulp',type=int,help='First gulp to read',default=0)
    parser.add_argument('--ngulps',type=int,help='Number of gulps to read',default=90)
    parser.add_argument('--gulpsize',type=int,help='Number of time samples in each gulp',default=25)
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold',default=3)
    parser.add_argument('--showprofiles',type=int,help='Number of folded profiles to show',default=2)

    args = parser.parse_args()
    main(args)


