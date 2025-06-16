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
from nsfrb.planning import nvss_cat,atnf_cat,LPT_cat,read_atnf
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

from nsfrb.planning import read_nvss,read_RFC,read_vlac
from dsautils.coordinates import create_WCS,get_declination,get_elevation
from astropy.coordinates import AltAz
from nsfrb.planning import find_fast_vis_label
from nsfrb.planning import nvss_cat
from nsfrb.config import tsamp as tsamp_ms
from nsfrb.config import IMAGE_SIZE,bmin,flagged_antennas,bad_antennas,pixperFWHM,NUM_CHANNELS
import json


def get_best_elev(reftime,timestep_hr = 1):
    """
    gets elevations for times throughout the past day and uses the median
    """
    steps_hr = np.arange(0,24,timestep_hr)
    elevs = []
    for i in steps_hr:
        elevs.append(get_elevation(Time(reftime.mjd-(i/24),format='mjd')).value)
        print(Time(reftime.mjd-(i/24),format='mjd').isot,":",elevs[-1])
    return np.nanmedian(elevs)*u.deg



def pulsarobs(args):
    # find brightest continuum sources at the current declination # dec 16 continuum sources
    if len(args.reftimeISOT) == 0:
        reftime = Time.now()#Time(args.UTCday + "T12:00:00",format='isot')
    else:
        reftime = Time(args.reftimeISOT,format='isot')
    print("Reference time:",reftime.isot)

    coords,names,Ps,DMs,Ws,S1400s = read_atnf()
    
    
    #estimate and plot theoretical sensitivity
    outriggers = args.outriggers
    flagcorrs = args.flagcorrs
    flagants = args.flagants
    gridsize = args.image_size

    SEFD=7000
    N=97 - int(0 if outriggers else len(outrigger_antennas)) - len(bad_antennas)
    print("Getting UVW params...")
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
    bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
    tmp, bname, blen, UVW, antenna_order = flag_vis(np.zeros((nsamps,UVW.shape[1],16,2)), bname, blen, UVW, antenna_order, (list(bad_antennas) + list(flagants) if outriggers else list(flagged_antennas) + list(flagants)), bmin, list(flagged_corrs) + list(flagcorrs), flag_channel_templates=[])
    uv_diag=np.max(np.sqrt(UVW[0,:,1]**2 + UVW[0,:,1]**2))
    pixel_resolution = (lambdaref/uv_diag/pixperFWHM)
    bweights_all = briggs_weighting(UVW[0,:,1]/lambdaref, UVW[0,:,0]/lambdaref, gridsize, robust=robust,pixel_resolution=pixel_resolution)
    print("Weights:",bweights_all,bweights_all.shape)
    Nbase = np.nansum(bweights_all)/np.max(bweights_all)
    print("True number of baselines:",UVW.shape[1])
    print("Eff. number of baselines:",Nbase)
    BW = chanbw*nchans*1E6
    print(Nbase,BW,tsamp_ms)
    NSFRBsens = 2*SEFD*1000/np.sqrt((2*Nbase)*BW*tsamp_ms*2/1000) #mJy
    print("Comparing to theoretical sensitivity:",NSFRBsens,"mJy")
    
    expectedSNRs = S1400s*(Ps/(Ws*1e-3))*(Ws/tsamp_ms)/NSFRBsens
    
    """
    #select pulsars in dec range detectable above 3sigma
    if args.search_dec == 180:
        elev = get_best_elev(reftime)#get_elevation(reftime)
        search_dec = get_declination(elev).value
    else:
        search_dec = args.search_dec
    
    decrange = args.decrange#1.5
    """
    condition = expectedSNRs>args.SNRmin #np.logical_and(np.abs(coords.dec.value-search_dec) <decrange,expectedSNRs>args.SNRmin)
    if np.sum(condition) == 0:
        print("No available test pulsars")
        return

    print("Test pulsars:" , names[condition])
    print("Expected SNRs:",expectedSNRs[condition])
    vis_nvsscoords = coords[condition]
    vis_nvssnames = names[condition]

    #single pulsar
    if len(args.psr)>0:
        idxs = []
        idxnames = []
        for i in range(len(args.psr)):
            print(args.psr[i])
            if args.psr[i] in vis_nvssnames:
                idxnames.append(args.psr[i])
                idxs.append(list(vis_nvssnames).index(args.psr[i]))
            else:
                print(args.psr[i],"not found")
        idxs = np.array(idxs)
        bright_nvsscoords = vis_nvsscoords[idxs]
        bright_nvssnames = vis_nvssnames[idxs]
        #bright_nvssms = vis_nvssms[idxs]
        #bright_catflags = vis_catflags[idxs]
        print("Running astrometric calibration pipeline with pulsars ",idxnames)
    else:
            bright_nvsscoords = vis_nvsscoords
            bright_nvssnames = vis_nvssnames
            print("Running flux calibration pipeline with " + str(len(bright_nvsscoords)) + " brightest pulsars:")



    #find the files within the timestamp with matching dec
    besttime = []
    bright_fnames = []
    bright_offsets = []
    for i in range(len(bright_nvsscoords)):
        timeax = Time(int(reftime.mjd) - np.linspace(0,args.maxtime,int(args.maxtime))[::-1]/24,format='mjd')
        DSA = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg,height=Height*u.m)

        hourvis = SkyCoord(bright_nvsscoords[i],location=DSA,obstime=timeax)

        #narrow to best minute
        antpos = hourvis.transform_to(AltAz)
        timeax = Time(timeax[np.argmax(antpos.alt.value)].mjd + np.linspace(-1,1,24)/24,format='mjd')
        minvis = SkyCoord(bright_nvsscoords[i],location=DSA,obstime=timeax)

        #narrow to best second
        antpos = minvis.transform_to(AltAz)
        timeax = Time(timeax[np.argmax(antpos.alt.value)].mjd + np.linspace(-1/60,1/60,24)/24,format='mjd')
        secvis = SkyCoord(bright_nvsscoords[i],location=DSA,obstime=timeax)

        antpos = secvis.transform_to(AltAz)
        besttime.append(timeax[np.argmax(antpos.alt.value)])
        ffvl = find_fast_vis_label(besttime[-1].mjd,return_dec=True)
        if ffvl[0] != -1 and np.abs(bright_nvsscoords[i].dec.value-ffvl[-1]) <args.decrange:#int(ffvl[-1])==int(search_dec):
            print(bright_nvssnames[i])
            print(besttime[-1].isot)
            print(ffvl)
            print("")
            bright_fnames.append(ffvl[0])
            bright_offsets.append(ffvl[1])
        else:
            print("Excluding " + bright_nvssnames[i])
            print("")
            bright_fnames.append(-1)
            bright_offsets.append(-1)

    print(bright_offsets)
    print("pulsars with fast vis data:",bright_nvssnames[np.array(bright_offsets)!= -1])

    #select gulps to search
    gulpsize=25
    for bright_idx in np.arange(len(bright_fnames),dtype=int)[np.array(bright_offsets)!= -1]:
        gulps = np.arange(bright_offsets[bright_idx]//gulpsize,min([(bright_offsets[bright_idx]//gulpsize)+args.ngulps,90]),dtype=int)
        if len(gulps)<args.ngulps:
            gulps = np.concatenate([np.arange(max([0,bright_offsets[bright_idx]//gulpsize - (args.ngulps-len(gulps))]),bright_offsets[bright_idx]//gulpsize),gulps])
            gulps = np.unique(gulps)

        start_gulp = gulps[0]
        num_gulps = len(gulps)
        print("Searching ",num_gulps,"gulps for pulsar",bright_nvssnames[bright_idx])
        os.system("python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py _" + str(bright_fnames[bright_idx]) + " --verbose --offline --num_gulps " + str(num_gulps) + " --gulp_offset " + str(start_gulp) + " --num_time_samples 25 --sb --nchans_per_node 8 --gridsize 301 --flagBPASS --flagBPASSBURST --sleeptime 0 --offsetRA_inject 0 --offsetDEC_inject 0 --robust -2 --bmin 20 --maxProcesses 32 --port 8080 --multiimage --stagger_multisend 0 --multisend --multiport 8810 8811 8812 8813 8814 8815 8816 8817 8818 8819 8820 8821 8822 8823 8824 8825 --briggs --search")
        

    
    return

def main(args):
    pulsarobs(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('--astrocal_only',action='store_true',help='Run astrometric cal only')
    parser.add_argument('--speccal_only',action='store_true',help='Run flux cal only')
    parser.add_argument('--init_astrocal',action='store_true',help='Initialize json astrometry table')
    parser.add_argument('--init_speccal',action='store_true',help='Initialize json flux cal table')
    #parser.add_argument('--UTCday',type=str,help='UTC day to run fluxcal with in ISO format (e.g. 2024-06-12); if not given, uses the previous day',default=Time(Time.now().mjd,format='mjd').isot[:10])
    parser.add_argument('--search_dec',type=float,help='If given, searches for source observations at this dec; otherwise uses median dec from past day',default=180.0)
    parser.add_argument('--numsources_NVSS',type=int,help='Maximum number of sources to use for fluxcal, takes the brightest within 0.5 degrees of the current dec, default=10',default=10)
    parser.add_argument('--numsources_RFC',type=int,help='Maximum number of sources to use for fluxcal, takes the brightest within 0.5 degrees of the current dec, default=10',default=10)
    parser.add_argument('--minsrc_NVSS',type=int,help='Number of sources to skip',default=0)
    parser.add_argument('--minsrc_RFC',type=int,help='Number of sources to skip',default=0)
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=8)
    parser.add_argument('--image_size',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--buff_speccal',type=int,help='Radius in pixels around the NVSS position to search for peak pixel, default=5',default=5)
    parser.add_argument('--buff_astrocal',type=int,help='Radius in pixels around the NVSS position to search for peak pixel, default=5',default=5)
    parser.add_argument('--flagSWAVE',action='store_true',help='Flag channels when SWAVE template RFI is detected, which manifests as a 2 Hz sin wave over ~5 minutes of data')
    parser.add_argument('--flagBPASS',action='store_true',help='Flag channels when BPASS template RFI is detected, which is simpl comparison to bandpass mean in visibilities')
    parser.add_argument('--flagFRCBAND',action='store_true',help='Flag channels in FRC miltiary allocation 1435-1525 MHz')
    parser.add_argument('--flagBPASSBURST',action='store_true',help='Flag channel when BPASS template RFI is detected in any timestep, i.e. should detect pulsed narrowband RFI')
    parser.add_argument('--flagcorrs',type=int,nargs='+',default=[],help='List of sb nodes [0-15] to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagants',type=int,nargs='+',default=[],help='List of antennas to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--dec',type=float,help='Pointing declination to search for calibrators, ideally the one that the array has targeted over the day in question, default pulls the dec on the given day at 12:00:00 UTC from ETCD',default=180)
    parser.add_argument('--outriggers',action='store_true',help='Includes outrigger antennas in imaging')
    parser.add_argument('--astroresid_th',type=float,help='Maximum allowed normalized residual to include in astrometric or flux cal; default=0.2',default=0.2)
    parser.add_argument('--specresid_th',type=float,help='Maximum allowed residual from linear speccal fit,default=0.1',default=0.1)
    parser.add_argument('--uselastimage',action='store_true',help='Use previously saved .npy file if available')
    #parser.add_argument('--image_flux',action='store_true',help='Derive flux from image instead of beamformed flux')
    parser.add_argument('--psr',nargs='+',type=str,help='J-name of specific pulsar to calibrate on; e.g. \'J182210+160015\'',default=[])
    parser.add_argument('--vmin',type=float,help='VMIN for astrocal plot',default=None)
    parser.add_argument('--vmax',type=float,help='VMAX for astrocal plot',default=None)
    parser.add_argument('--update_only',action='store_true',help='Updates based on the existing tables')
    parser.add_argument('--target',type=str,help='J2000 coordinates of target for which astrometric and flux cal are needed',default='')
    parser.add_argument('--targetMJD',type=float,help='MJD at which target was observed',default=0.0)
    parser.add_argument('--target_timerange',type=float,help='Time range in hours within which sources should be included in astrometric and flux cal,default=5',default=5)
    parser.add_argument('--target_decrange',type=float,help='Dec range in degrees within which sources should be included in astrometric and flux cal,default=0.5',default=0.5)
    parser.add_argument('--boxmean',action='store_true',help='When --image_flux is set, takes the mean flux within --buff pixels of the NVSS coordinate instead of the peak pixel flux')
    parser.add_argument('--ngulps',type=int,help='Number of gulps of 25 samples (3.25 s) to integrate, default=1',default=1)
    parser.add_argument('--timebin',type=int,help='Number of time samples to bin by, default=1',default=1)
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging,default=-2 for uniform weighting',default=-2)
    #parser.add_argument('--vlac_only',action='store_true',help='Only use VLAC sources')
    #parser.add_argument('--nvss_only',action='store_true',help='Only use NVSS sources')
    parser.add_argument('--decrange',type=float,help='radius in degrees to search for sources around search dec, default=0.5',default=0.5)
    parser.add_argument('--maxtime',type=float,help='max time in hours to look backwards for calibrator passes, default 24',default=24)
    parser.add_argument('--reftimeISOT',type=str,help='reference time, default now',default='')
    parser.add_argument('--fluxmin',type=float,help='minimum flux of sources for speccal in mJy',default=0)
    parser.add_argument('--fluxmax',type=float,help='maximum flux of sources for speccal in mJy',default=np.inf)
    parser.add_argument('--randomsources',action='store_true',help='Select random set of calibrators near the specified declination instead of taking the brightest')
    parser.add_argument('--exactposition',action='store_true',help='Set to measure flux at pixel closest to NVSS position instead of peak pixel')
    parser.add_argument('--SNRmin',type=float,help='minimum S/N of pulsars',default=3)
    args = parser.parse_args()
    main(args)
