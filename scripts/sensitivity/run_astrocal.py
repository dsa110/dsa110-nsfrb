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

from nsfrb.config import *
from nsfrb.imaging import get_ra
from matplotlib.patches import Ellipse
from nsfrb.config import *
import numpy as np
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
import pickle as pkl

from nsfrb.planning import read_nvss
from dsautils.coordinates import create_WCS,get_declination,get_elevation
from astropy.coordinates import AltAz
from nsfrb.planning import find_fast_vis_label
from nsfrb.planning import nvss_cat
from nsfrb.config import tsamp as tsamp_ms
from nsfrb.config import IMAGE_SIZE,bmin,flagged_antennas,bad_antennas,pixperFWHM,NUM_CHANNELS
import json

def init_table(outriggers=False,astrocal_table=table_dir + "/NSFRB_astrocal.json"):
    print("Initializing table " + astrocal_table)
    arraykey = str('outriggers' if outriggers else 'core')
    #read current table
    if len(glob.glob(astrocal_table))>0:
        f = open(astrocal_table,"r")
        tab = json.load(f)
        f.close()
        if arraykey in tab.keys():
            del tab[arraykey]
    else:
        tab = dict()
    tab[arraykey] = dict()
    f = open(astrocal_table,"w")
    json.dump(tab,f)
    f.close()
    return

def update_speccal_table(bright_nvssnames,bright_fnames,bright_measfluxs,bright_measfluxerrs,bright_nvssfluxes,bright_resid,outriggers,speccal_table=table_dir + "/NSFRB_speccal.json",init=False,resid_th=np.inf,exclude_table=table_dir + "/NSFRB_excludecal.json",nsamps=nsamps):
    """
    This function updates the flux calibration table with the most
    recent NVSS observations.
    """
    #read current table
    if init:
        init_table(outriggers,speccal_table)
    f = open(speccal_table,"r")
    arraykey = str('outriggers' if outriggers else 'core')
    tab = json.load(f)
    f.close()

    #read sources to exclude
    if len(exclude_table) > 0:
        f = open(exclude_table,"r")
        ex_table = json.load(f)['exclude']
        f.close()
    else:
        ex_table =  []
    print("sources to exclude:",ex_table)

    #add new sources to table (or update if already there)
    print(bright_measfluxs)
    print(bright_measfluxerrs)
    print(bright_nvssfluxes)
    for i in range(len(bright_nvssnames)):
        if bright_measfluxs[i]==-1: continue
        if bright_nvssnames[i] not in tab[arraykey].keys():
            tab[arraykey][bright_nvssnames[i]] = dict()
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]] = dict()
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["meas_flux"] = float(bright_measfluxs[i]/nsamps)
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["meas_flux_error"] = float(bright_measfluxerrs[i]/nsamps)
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["flux_mJy"] = float(bright_nvssfluxes[i])
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["RMS_fit_residual"] = bright_resid[i]

    #update calibration curve
    allfluxs = []
    allfluxerrs = []
    allnvssfluxes = []
    allresids = []
    for k in tab[arraykey].keys():
        for kk in tab[arraykey][k].keys():
            if str(k) not in ex_table:
                allfluxs.append(tab[arraykey][k][kk]['meas_flux'])
                allfluxerrs.append(tab[arraykey][k][kk]['meas_flux_error'])
                allnvssfluxes.append(tab[arraykey][k][kk]['flux_mJy'])
                allresids.append(tab[arraykey][k][kk]['RMS_fit_residual'])
    allfluxs = np.array(allfluxs)
    allfluxerrs = np.array(allfluxerrs)
    allnvssfluxes = np.array(allnvssfluxes)
    allresids = np.array(allresids)
    
    print(allfluxs,allresids)
    if len(allfluxs)<2:
        f = open(speccal_table,"w")
        json.dump(tab,f)
        f.close()
        print("No remaining sources for spec cal")
        return
    badsoln = False
    try:
        popt,pcov = np.polyfit(allfluxs[allresids<resid_th],allnvssfluxes[allresids<resid_th],1,full=False,cov=True,w=1/allresids[allresids<resid_th])
        popterrs = np.sqrt([pcov[0,0],pcov[1,1]])
        pfunc = np.poly1d(popt)
        pfunc_down = np.poly1d(popt - popterrs)
        pfunc_up = np.poly1d(popt + popterrs)
        tab[str('outriggers' if outriggers else 'core') + "_slope"] = float(popt[0])
        tab[str('outriggers' if outriggers else 'core') + "_slope_error"] = float(popterrs[0])
        tab[str('outriggers' if outriggers else 'core') + "_int"] = float(popt[1])
        tab[str('outriggers' if outriggers else 'core') + "_int_error"] = float(popterrs[1])
        print("Updated flux conversion fit: FLUX = (",popt[1],"+-",popterrs[1],") + (",popt[0],"+-",popterrs[0],")MEAS_FLUX") 
        if np.sum(allresids>resid_th)>0:
            Smin = np.nanpercentile(allnvssfluxes[allresids>resid_th],90)
            print("Estimating sensitivity limit from 90th percentile of sources with residuals>",resid_th,": Smin=",Smin,"mJy")
            tab[str('outriggers' if outriggers else 'core') + "_Smin"] = Smin
        else:
            tab[str('outriggers' if outriggers else 'core') + "_Smin"] = np.nan
    except Exception as exc:
        badsoln = True
        print("Flux cal linear fit failed with error:",exc)
        print("Not enough points for flux cal solution")
    f = open(speccal_table,"w")
    json.dump(tab,f)
    f.close()

    #plot results
    print(allfluxs)
    print(allnvssfluxes)
    plt.figure(figsize=(12,12))
    #plt.errorbar(allfluxs,allnvssfluxes,xerr=allfluxerrs,marker='o',markersize=10,linestyle='')
    faxis = np.linspace(np.min(allfluxs)/10,np.max(allfluxs)*1.2,1000)

    if not badsoln:#str('outriggers' if outriggers else 'core') + "_slope" in tab.keys():
        plt.plot(faxis,pfunc(faxis),color='red')
        plt.fill_between(faxis,pfunc_down(faxis),pfunc_up(faxis),color='red',alpha=0.5)
    plt.scatter(allfluxs,allnvssfluxes,c=allresids,marker="o",s=100,cmap='copper')
    plt.xlabel("Mean Pixel Value per Time Sample (arb. units)")
    plt.ylabel("NVSS Flux (mJy)")
    plt.title("Last Updated: " + Time.now().isot)
    plt.xlim(np.nanmin(allfluxs)/10,np.nanmax(allfluxs)*1.2)
    plt.ylim(np.nanmin(allnvssfluxes)/10,np.nanmax(allnvssfluxes)*1.2)

    if not badsoln and not np.isnan(tab[str('outriggers' if outriggers else 'core') + "_Smin"]):
        plt.axhline(tab[str('outriggers' if outriggers else 'core') + "_Smin"],color='purple',linestyle='--')
    #plt.yscale("log")
    #plt.xscale("log")
    plt.colorbar(label="Normalized RMS Residual")
    plt.savefig(img_dir+"NVSStotal_"+ str("outriggers_" if outriggers else "")+"speccal.png")
    plt.close()
    
    return

def update_astrocal_table(bright_nvssnames,bright_fnames,bright_poserrs,bright_raerrs,bright_decerrs,bright_resid,outriggers,astrocal_table=table_dir + "/NSFRB_astrocal.json",init=False,resid_th=np.inf,exclude_table=table_dir + "/NSFRB_excludecal.json"):
    """
    This function updates the astrometric calibration table with the most 
    recent NVSS observations.
    """
    #read current table
    if init:
        init_table(outriggers,astrocal_table)
    f = open(astrocal_table,"r")
    arraykey = str('outriggers' if outriggers else 'core')
    tab = json.load(f)
    f.close()

    #read sources to exclude
    if len(exclude_table) > 0:
        f = open(exclude_table,"r")
        ex_table = json.load(f)['exclude']
        f.close()
    else:
        ex_table =  []
    print("sources to exclude:",ex_table)

    #add new sources to table (or update if already there)
    print(bright_nvssnames)
    print(bright_fnames)
    print(bright_poserrs)
    for i in range(len(bright_nvssnames)):
        if bright_poserrs[i] == -1: continue
        if bright_nvssnames[i] not in tab[arraykey].keys():
            tab[arraykey][bright_nvssnames[i]] = dict()
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]] = dict()
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["position_error_deg"] = bright_poserrs[i]
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["RA_error_deg"] = bright_raerrs[i]
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["DEC_error_deg"] = bright_decerrs[i]
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["RMS_fit_residual"] = bright_resid[i]

    #update total RMS errors
    allposerrs = []
    allRAerrs = []
    allDECerrs = []
    allresids = []
    for k in tab[arraykey].keys():
        for kk in tab[arraykey][k].keys():
            if str(k) not in ex_table and tab[arraykey][k][kk]['RMS_fit_residual'] < resid_th:
                allposerrs.append(tab[arraykey][k][kk]['position_error_deg'])
                allDECerrs.append(tab[arraykey][k][kk]['DEC_error_deg'])
                allRAerrs.append(tab[arraykey][k][kk]['RA_error_deg'])
                allresids.append(tab[arraykey][k][kk]['RMS_fit_residual'])
    allposerrs = np.array(allposerrs)
    allRAerrs = np.array(allRAerrs)
    allDECerrs = np.array(allDECerrs)
    allresids = np.array(allresids)
    print(allposerrs,allresids)
    if len(allposerrs)<1:
        f = open(astrocal_table,"w")
        json.dump(tab,f)
        f.close()
        print("No remaining sources for astro cal")
        return
    tab[str('outriggers' if outriggers else 'core') + "_position_error_deg"] = np.sqrt(np.average(np.array(allposerrs)**2,weights=1/allresids))
    tab[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"] = (np.average(allRAerrs,weights=1/allresids) if len(allRAerrs)>1 else 0)
    tab[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"] = (np.average(allDECerrs,weights=1/allresids) if len(allDECerrs)>1 else 0)
    tab[str('outriggers' if outriggers else 'core') + "_RA_error_deg"] = np.sqrt(np.average((np.array(allRAerrs) - tab[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"])**2,weights=1/allresids))
    tab[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"] = np.sqrt(np.average((np.array(allDECerrs) - tab[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"])**2,weights=1/allresids))
    print("Updated Total Position Error:",tab[str('outriggers' if outriggers else 'core') + "_position_error_deg"])
    print("Updated RA Offset:",tab[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"])
    print("Updated DEC Offset:",tab[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"])
    print("Updated RA Error:",tab[str('outriggers' if outriggers else 'core') + "_RA_error_deg"])
    print("Updated DEC Error:",tab[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"])
    f = open(astrocal_table,"w")
    json.dump(tab,f)
    f.close()

    #plot results
    plt.figure(figsize=(12,12))
    plt.axvline(0,color='grey',alpha=0.5)
    plt.axhline(0,color='grey',alpha=0.5)
    plt.scatter(allRAerrs*60,allDECerrs*60,c=allresids,marker="o",s=100,cmap='copper')
    plt.errorbar(np.nanmean(allRAerrs)*60,np.nanmean(allDECerrs)*60,xerr=60*tab[str('outriggers' if outriggers else 'core') + "_RA_error_deg"],yerr=60*tab[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"],capsize=10,color='red')
    plt.xlabel("RA Error (arcmin)")
    plt.ylabel("DEC Error (arcmin)")
    plt.title("Last Updated: " + Time.now().isot)
    plt.colorbar(label="Normalized RMS Residual")
    plt.xlim(-np.max(allposerrs)*60*2,np.max(allposerrs)*60*2)
    plt.ylim(-np.max(allposerrs)*60*2,np.max(allposerrs)*60*2)
    plt.savefig(img_dir+"NVSStotal_"+ str("outriggers_" if outriggers else "")+"astrocal.png")
    plt.close()


    return
    

#ellipse fitting function
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
def ellipse_fit(theta,a,b,PA):
    t1 = ((b*np.cos(PA))**2 + (a*np.sin(PA))**2)*(np.cos(theta)**2)
    t2 = ((b*np.sin(PA))**2 + (a*np.cos(PA))**2)*(np.sin(theta)**2)
    t3 = 2*(a**2 - b**2)*np.cos(PA)*np.sin(PA)*np.cos(theta)*np.sin(theta)

    return a*b/np.sqrt(t1 + t2 + t3)

def ellipse_to_covariance(semiMajorAxis,semiMinorAxis,phi):
    varX1 = semiMajorAxis**2 * np.sin(phi)**2 + semiMinorAxis**2 * np.cos(phi)**2
    varX2 = semiMajorAxis**2 * np.cos(phi)**2 + semiMinorAxis**2 * np.sin(phi)**2
    cov12 = (semiMajorAxis**2 - semiMinorAxis**2) * np.cos(phi) * np.sin(phi)
    cmatrix = np.array([[varX1, cov12], [cov12, varX2]])
    return cmatrix

def fit_function(coord,ra,dec,width,height,PA,amp):
    cov = ellipse_to_covariance(width/2,height/2,PA)
    p = multivariate_normal.pdf(coord,mean=(ra,dec),cov=cov).flatten()
    return amp*p/np.max(p)

#modified version of dsa110-meridian-fringestopping's generate_fringestopping_table
def generate_rephasing_table(
        source_HA, source_dec, blen, pt_dec, nint, tsamp, antenna_order, outrigger_delays, bname, mjd0,
        outname="fringestopping_table"):
    """Generates a table of the w vectors towards a source.

    Generates a table for use in fringestopping and writes it to a numpy
    pickle file named fringestopping_table.npz

    Parameters
    ----------
    source_HA : Hour angle of source to phase visibilities to (i.e. not zenith)
    blen : array
        The lengths of the baselines in ITRF coordinates, in m. Dimensions
        (nbaselines, 3).
    pt_dec : float
        The pointing declination in radians.
    nint : int
        The number of time integrations to calculate the table for.
    tsamp : float
        The sampling time in seconds.
    antenna_order : list
        The order of the antennas.
    outrigger_delays : dict
        The outrigger delays in ns.
    bname : list
        The names of each baseline. Length nbaselines. Names are strings.
    outname : str
        The prefix to use for the table to which to save the w vectors. Will
        save the output to `outname`.npy Defaults ``fringestopping_table``.
    mjd0 : float
        The start time in MJD.
    """
    # Get the indices that correspond to baselines with the refant
    # Use the first antenna as the refant so that the baselines are in
    # the same order as the antennas
    refidxs = []
    refant = str(antenna_order[0])
    for i, bn in enumerate(bname):
        if refant in bn:
            refidxs += [i]

    # Get the geometric delays at the "source" position and meridian
    dt = np.arange(nint) * tsamp
    dt = dt - np.median(dt)
    hangle = source_HA + dt * 360 / ct.SECONDS_PER_SIDEREAL_DAY
    _bu, _bv, bw = calc_uvw(
        blen, mjd0 + dt / ct.SECONDS_PER_DAY, "HADEC", hangle * u.deg,
        np.ones(hangle.shape) * (source_dec * u.deg))
    _bu, _bv, bwref = calc_uvw(
        blen, mjd0, "HADEC", 0. * u.deg, (pt_dec * u.rad).to(u.deg))
    ant_bw = bwref[refidxs]
    bw = bw - bwref
    bw = bw.T
    bwref = bwref.T

    # Add in per-antenna delays for each baseline
    for i, bn in enumerate(bname):
        ant1, ant2 = bn.split('-')
        # Add back in bw at the meridian calculated per antenna
        bw[:, i] += ant_bw[antenna_order.index(int(ant2)), :] - \
            ant_bw[antenna_order.index(int(ant1)), :]
        # Add in outrigger delays
        bw[:, i] += (outrigger_delays.get(str(ant1), 0) -
                     outrigger_delays.get(str(ant2), 0)) * 0.29979245800000004

    # Save the fringestopping table
    if os.path.exists(outname):
        os.unlink(outname)
    np.savez(
        outname, dec_rad=pt_dec, tsamp_s=tsamp, ha=hangle, bw=bw, bwref=bwref,
        antenna_order=antenna_order, outrigger_delays=outrigger_delays, ant_bw=ant_bw)


def main(args):

    # find brightest continuum sources at the current declination # dec 16 continuum sources
    reftime = Time(args.UTCday + "T12:00:00",format='isot')
    allnvsscoords,allnvssfluxes,allnvssms = read_nvss()
    if args.dec == 180:
        elev = get_elevation(reftime)
        search_dec = get_declination(elev).value
    else:
        search_dec = args.dec

    #read sources to exclude
    exclude_table = table_dir + "/NSFRB_excludecal.json"
    if len(exclude_table) > 0:
        f = open(exclude_table,"r")
        ex_table = json.load(f)['exclude']
        f.close()
    else:
        ex_table =  []
    print("sources to exclude:",ex_table)

    decrange= 0.5
    vis_nvsscoords = allnvsscoords[np.abs(allnvsscoords.dec.value-search_dec) <decrange]
    vis_nvssfluxes = allnvssfluxes[np.abs(allnvsscoords.dec.value-search_dec) <decrange]
    vis_nvssms = allnvssms[np.abs(allnvsscoords.dec.value-search_dec) <decrange]
    fluxth = np.sort(vis_nvssfluxes)[-args.numsources]
    bright_nvsscoords = vis_nvsscoords[vis_nvssfluxes>=fluxth]
    bright_nvssfluxes = vis_nvssfluxes[vis_nvssfluxes>=fluxth]
    bright_nvssms = vis_nvssms[vis_nvssfluxes>=fluxth]
    print("Running astrometric calibration pipeline with " + str(len(bright_nvsscoords)) + " brightest NVSS sources at dec=" + str(search_dec) + ":")

    #find the files within the timestamp
    besttime = []
    bright_nvssnames = []
    bright_fnames = []
    bright_offsets = []
    for i in range(len(bright_nvsscoords)):
        timeax = Time(int(reftime.mjd) + np.linspace(0,24,24)/24,format='mjd')
        DSA = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg,height=Height*u.m)
        name = str('NVSS J{RH:02d}{RM:02d}{RS:02d}'.format(RH=int(bright_nvsscoords[i].ra.hms.h),
                                                               RM=int(bright_nvsscoords[i].ra.hms.m),
                                                               RS=int(bright_nvsscoords[i].ra.hms.s)) + 
                           str("+" if bright_nvsscoords[i].dec>=0 else "-") + 
                           '{DD:02d}{DM:02d}{DS:02d}'.format(DD=int(bright_nvsscoords[i].dec.dms.d),
                                                               DM=int(bright_nvsscoords[i].dec.dms.m),
                                                               DS=int(bright_nvsscoords[i].dec.dms.s)))
    
        bright_nvssnames.append(name)
        
        if name in ex_table:
            besttime.append(-1)
            print("Excluding " + name)
            print("")
            bright_fnames.append(-1)
            bright_offsets.append(-1)
            continue
    
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
        if ffvl[0] != -1 and int(ffvl[-1])==int(search_dec):
            print(name)
            print(besttime[-1].isot)
            print(ffvl)
            print("")
            bright_fnames.append(ffvl[0])
            bright_offsets.append(ffvl[1])
        else:
            print("Excluding " + name)
            print("")
            bright_fnames.append(-1)
            bright_offsets.append(-1)
    
    print(bright_fnames)
    #Run imaging pipeline

    bright_poserrs = []
    bright_raerrs = []
    bright_decerrs = []
    bright_pixcoords = []
    bright_pixs = []
    close_pixs = []
    bright_measfluxs = []
    bright_measfluxerrs = []
    bright_resid = []
    image_size=args.image_size#1101 #8001
    gulpsize = nsamps
    nchan_per_node=nchans_per_node = args.nchans_per_node
    outriggers = args.outriggers
    ref_wav=0.20
    bmin=args.bmin
    full_img = np.zeros((image_size,image_size,gulpsize,16*nchan_per_node))
    savestuff = True
    for bright_idx in range(len(bright_fnames)):
        if bright_fnames[bright_idx] == -1:
            bright_poserrs.append(-1)
            bright_raerrs.append(-1)
            bright_decerrs.append(-1)
            bright_pixcoords.append(-1)
            bright_pixs.append(-1)
            close_pixs.append(-1)
            bright_measfluxs.append(-1)
            bright_measfluxerrs.append(-1)
            bright_resid.append(-1)
            print("Excluding " + bright_nvssnames[bright_idx])
            continue
    
        print("Reading data for "+ bright_nvssnames[bright_idx])
        fnum = int(bright_fnames[bright_idx])
        gulps = np.arange(bright_offsets[bright_idx]//gulpsize,(bright_offsets[bright_idx]//gulpsize)+1,dtype=int)
        print(bright_idx,fnum,gulps)
        g=0
        full_img[:,:,:,:] = 0
        for gulp in gulps:#[77,78,79,80,81]:##0,45,75]:#range(3):

            dat = None
            sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
            corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
            copydir = vis_dir + bright_nvssnames[bright_idx].replace(" ","") + "/"
            os.system("mkdir " + copydir)
            for i in range(16):
                try:
                    if len(glob.glob(copydir + "nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out")) == 0:
                        os.system("cp " + vis_dir + "/lxd110" + corrs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out " + copydir)
                        dat_i,sb,mjd,dec = pipeline.read_raw_vis(vis_dir + "/lxd110" + corrs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=gulp,headersize=16)
                    else:
                        dat_i,sb,mjd,dec = pipeline.read_raw_vis(copydir + "nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=gulp,headersize=16)

                    print(mjd,dec,sb)

                    if dat is None:
                        dat = np.nan*np.ones(dat_i.shape,dtype=dat_i.dtype).repeat(len(corrs),axis=2)
                    dat[:,:,i*nchans_per_node:(i+1)*nchans_per_node,:] = dat_i


                except Exception as exc:
                    print(exc)

            if dat is None or int(dec) != int(search_dec):
                bright_poserrs.append(-1)
                bright_raerrs.append(-1)
                bright_decerrs.append(-1)
                bright_pixcoords.append(-1)
                bright_pixs.append(-1)
                close_pixs.append(-1)
                bright_measfluxs.append(-1)
                bright_measfluxerrs.append(-1)
                bright_resid.append(-1)
                print("Excluding " + bright_nvssnames[bright_idx])
                continue

            #save a copy for beamforming
            print("Saving a data copy for beamforming...")
            dat_copy = copy.deepcopy(dat)

            print("Getting UVW params...")
            test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
            pt_dec = dec*np.pi/180.
            bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
            ff = 1.53-np.arange(8192)*0.25/8192
            fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
            fobs = np.reshape(fobs,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1)


            #flagging andd baseline cut
            fcts = []
            if args.flagSWAVE:
                fcts.append(fct_SWAVE)
            if args.flagBPASS:
                fcts.append(fct_BPASS)
            if args.flagFRCBAND:
                fcts.append(fct_FRCBAND)
            if args.flagBPASSBURST:
                fcts.append(fct_BPASSBURST)

            dat, bname, blen, UVW, antenna_order = flag_vis(dat, bname, blen, UVW, antenna_order, (list(bad_antennas) + list(args.flagants) if outriggers else list(flagged_antennas) + list(args.flagants)), bmin, list(flagged_corrs) + list(args.flagcorrs), flag_channel_templates = fcts)
            U = UVW[0,:,0]
            V = UVW[0,:,1]
            W = UVW[0,:,2]

            uv_diag=np.max(np.sqrt(U**2 + V**2))
            pixel_resolution = (0.20/uv_diag/pixperFWHM)
            dat[np.isnan(dat)] = 0


            for i in range(dat.shape[0]):
                for j in range(len(corrs)):
                    for k in range(dat.shape[-1]):
                        for jj in range(nchans_per_node):
                            tmpimg = revised_robust_image(dat[i:i+1,:,(j*nchans_per_node) + jj,k],
                                                   U/(2.998e8/fobs[(j*nchans_per_node) + jj]/1e9),
                                                   V/(2.998e8/fobs[(j*nchans_per_node) + jj]/1e9),
                                                   image_size,robust=-2)

                            full_img[:,:,(g*gulpsize) + i,(j*nchans_per_node) + jj]  += tmpimg

            g += 1
        if dat is None or int(dec) != int(search_dec):
            continue
        if savestuff:
            np.save(copydir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_" + str("outriggers_" if outriggers else "") + "image.npy",full_img)

        # ASTROMETRIC TEST
        # find the peak pixel in the vicinity of the coordinates
        buff = args.buff#50
        ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + (gulps[0]*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)

        if (bright_nvsscoords[bright_idx].ra.value < np.min(ra_grid_2D) or
            bright_nvsscoords[bright_idx].ra.value > np.max(ra_grid_2D) or
            bright_nvsscoords[bright_idx].dec.value < np.min(dec_grid_2D) or
            bright_nvsscoords[bright_idx].dec.value > np.max(dec_grid_2D)):
            bright_poserrs.append(-1)
            bright_raerrs.append(-1)
            bright_decerrs.append(-1)
            bright_pixcoords.append(-1)
            bright_pixs.append(-1)
            close_pixs.append(-1)
            bright_measfluxs.append(-1)
            bright_measfluxerrs.append(-1)
            bright_resid.append(-1)
            print("Excluding " + bright_nvssnames[bright_idx])
            continue


        closepix = np.unravel_index(np.argmin(bright_nvsscoords[bright_idx].separation(SkyCoord(ra_grid_2D*u.deg,
                                                                                           dec_grid_2D*u.deg,frame='icrs'))),ra_grid_2D.shape)
        bbox = (max([closepix[0]-buff,0]),
            min([closepix[0]+buff+1,image_size]),
            max([closepix[1]-buff,0]),
            min([closepix[1]+buff+1,image_size]))
        input_img = full_img[bbox[0]:bbox[1],bbox[2]:bbox[3],:,:].mean((2,3))
        peakpix = np.unravel_index(np.argmax(input_img),(bbox[1]-bbox[0],bbox[3]-bbox[2]))
        bright_pix = (peakpix[0] + bbox[0] ,peakpix[1] + bbox[2])
        bright_pixcoord = SkyCoord(ra_grid_2D[bright_pix[0],bright_pix[1]]*u.deg,dec_grid_2D[bright_pix[0],bright_pix[1]]*u.deg,frame='icrs')


        #fit with an ellipse
        input_ra_grid_2D = ra_grid_2D[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        input_dec_grid_2D = dec_grid_2D[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        input_sigma = np.sqrt(np.clip(input_img,1e-10,np.inf))
        p0 = (bright_pixcoord.ra.value,bright_pixcoord.dec.value,
                pixel_resolution*pixperFWHM*2,pixel_resolution*pixperFWHM*2,
                np.pi/2,np.max(input_img))
        bounds = ([np.nanmin(input_ra_grid_2D),np.nanmin(input_dec_grid_2D),
                      pixel_resolution*pixperFWHM,
                      pixel_resolution*pixperFWHM,
                      0,np.nanmax(input_img)*0.9],
                  [np.nanmax(input_ra_grid_2D),np.nanmax(input_dec_grid_2D),
                      max([np.max(input_ra_grid_2D)-np.min(input_ra_grid_2D),np.max(input_dec_grid_2D)-np.min(input_dec_grid_2D)]),
                      max([np.max(input_ra_grid_2D)-np.min(input_ra_grid_2D),np.max(input_dec_grid_2D)-np.min(input_dec_grid_2D)]),
                      2*np.pi,np.nanmax(input_img)*2])
        popt,pcov = curve_fit(fit_function,np.concatenate([input_ra_grid_2D[:,:,np.newaxis],
                                               input_dec_grid_2D[:,:,np.newaxis]],2),
                                input_img.flatten(),p0=p0,bounds=bounds,sigma=input_sigma.flatten())
        bright_pixcoord = SkyCoord(popt[0]*u.deg,popt[1]*u.deg,frame='icrs')
        param_errs=np.sqrt(pcov[np.arange(pcov.shape[0]),np.arange(pcov.shape[1])])
        print("Optimal Parameters:")
        print("Position:",bright_pixcoord.ra.to(u.hourangle),r'+-',(param_errs[0]*u.deg).to(u.arcsecond),bright_pixcoord.dec,r'+-',(param_errs[1]*u.deg).to(u.arcsecond))
        print("Semimajor axis:",(0.5*popt[2]*u.deg).to(u.arcmin))
        print("Semiminor axis:",(0.5*popt[3]*u.deg).to(u.arcmin))
        print("Angle (counterclockwise from vertical):",(-popt[4]*u.rad).to(u.deg))
        print("Errors:",np.sqrt(pcov[np.arange(pcov.shape[0]),np.arange(pcov.shape[1])]))



        print("Brightest Pixel coordinate:",bright_pixcoord)
        pos_err = np.sqrt((param_errs[0]*60)**2 + (param_errs[1]*60)**2 + bright_pixcoord.separation(bright_nvsscoords[bright_idx]).to(u.arcmin).value**2)
        print("Total Position Error:",pos_err,"arcminutes")
        ra_pos_err = bright_pixcoord.ra.value - bright_nvsscoords[bright_idx].ra.value
        dec_pos_err = bright_pixcoord.dec.value - bright_nvsscoords[bright_idx].dec.value
        print("RA Error:",ra_pos_err*60,"arcminutes")
        print("DEC Error:",dec_pos_err*60,"arcminutes")



        bright_poserrs.append(pos_err/60)
        bright_raerrs.append(ra_pos_err)
        bright_decerrs.append(dec_pos_err)
        bright_pixcoords.append(bright_pixcoord)
        bright_pixs.append(bright_pix)
        close_pixs.append(closepix)

        #compute residual from best fit gaussian
        image_resid = np.sqrt(np.nanmean((input_img.flatten() - fit_function(np.concatenate([input_ra_grid_2D[:,:,np.newaxis],
                                                                                            input_dec_grid_2D[:,:,np.newaxis]],2),
                                                                                            popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]))**2))/popt[5] 
        bright_resid.append(image_resid)
        print("Position fit residuals:",image_resid)

        #plotting
        plt.figure(figsize=(12,12))
        fullmean = True
        median_sub = False
        vmin=None#None#-np.nanmax(full_img.mean((2,3)))/4#-1
        vmax=None#None#np.nanmax(full_img.mean((2,3)))/4#1 #0.2#1

        plt.title("SOURCE: " + bright_nvssnames[bright_idx] + "\nMJD: " + str(mjd) + "\nFNUM: " + bright_fnames[bright_idx],fontsize=20)
        plt.scatter(ra_grid_2D.flatten(),dec_grid_2D.flatten(),c=(full_img.mean((2,3))).flatten(),alpha=1,cmap='cool',marker='s',s=10,vmin=vmin,vmax=vmax)#0.8*np.nanmax((full_img.mean((2,3)))))
        plt.scatter(bright_nvsscoords[bright_idx].ra.to(u.deg).value,bright_nvsscoords[bright_idx].dec.to(u.deg).value,marker='o',s=1000,edgecolors='red',linewidth=4,facecolors="none",alpha=0.8)
        plt.xlim(bright_nvsscoords[bright_idx].ra.to(u.deg).value-(0.3 if outriggers else 1),bright_nvsscoords[bright_idx].ra.to(u.deg).value+(0.3 if outriggers else 1))
        plt.ylim(bright_nvsscoords[bright_idx].dec.to(u.deg).value-(0.3 if outriggers else 1),bright_nvsscoords[bright_idx].dec.to(u.deg).value+(0.3 if outriggers else 1))
        plt.plot([ra_grid_2D[bbox[0],bbox[2]],
              ra_grid_2D[bbox[0],bbox[3]-1],
              ra_grid_2D[bbox[1]-1,bbox[3]-1],
              ra_grid_2D[bbox[1]-1,bbox[2]],
              ra_grid_2D[bbox[0],bbox[2]]],
             [dec_grid_2D[bbox[0],bbox[2]],
              dec_grid_2D[bbox[0],bbox[3]-1],
              dec_grid_2D[bbox[1]-1,bbox[3]-1],
              dec_grid_2D[bbox[1]-1,bbox[2]],
              dec_grid_2D[bbox[0],bbox[2]]],color='red')
        plt.scatter(bright_pixcoord.ra.to(u.deg).value,bright_pixcoord.dec.to(u.deg).value,marker='o',s=1000,edgecolors='green',linewidth=4,facecolors="none",alpha=0.8)
        ell = Ellipse((bright_pixcoord.ra.value,bright_pixcoord.dec.value),popt[3]*2,popt[2]*2,angle=-(popt[4]*180/np.pi),fill=None,color='black',linewidth=1,alpha=1,linestyle='--')
        plt.gca().add_patch(ell)
        plt.gca().invert_xaxis()
        if savestuff:
            plt.savefig(img_dir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_"+ str("outriggers_" if outriggers else "")+"astrocal.png")
        plt.close()


        #FLUX CAL

        #re-do baselines
        test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
        pt_dec = dec*np.pi/180.
        bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
        ff = 1.53-np.arange(8192)*0.25/8192
        fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
        fobs = np.reshape(fobs,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1)



        HA = (get_ra(mjd + gulps[0]*tsamp_ms*gulpsize/1000/86400,dec) - bright_pixcoord.ra.value)
        srcdec = bright_pixcoord.dec.value
        print("Phasing Visibilities to Hour Angle ",HA," deg, Declination ",srcdec," and beamforming...")
        generate_rephasing_table(
                HA, srcdec, blen, pt_dec, gulpsize, tsamp, antenna_order, outrigger_delays, bname, refmjd,
                outname=table_dir + "/tmp_fringestopping_table_cal")
        vis_model = fringestopping.zenith_visibility_model(fobs, fstable=table_dir + '/tmp_fringestopping_table_cal.npz')[0,:,:,:,:].repeat(2,3)
        dat_copy /= vis_model
        print("Reflagging...")
        dat, bname, blen, UVW, antenna_order = flag_vis(dat_copy, bname, blen, UVW, antenna_order, (list(bad_antennas) + list(args.flagants) if outriggers else list(flagged_antennas) + list(args.flagants)), bmin, list(flagged_corrs) + list(args.flagcorrs), flag_channel_templates = fcts)
        print("Done")
        bright_measfluxs.append(np.nanmean(np.real(dat)))
        bright_measfluxerrs.append(np.nanstd(np.real(dat))/np.sqrt(len(dat.flatten())))
        if np.nanmean(np.real(dat))<0:
            bright_measfluxs[-1] = -1
            bright_measfluxerrs[-1] = -1
            continue
        print("Beamformed flux arb. units:",bright_measfluxs[-1]," +- ",bright_measfluxerrs[-1])
        print("-"*20)
        print("")
        
        #plotting
        bright_dynspec = np.real(dat.mean((1,3)))
        plt.figure(figsize=(16,12))
        plt.subplot(2,2,1,facecolor='black')
        plt.step(np.arange(gulpsize)*tsamp_ms,bright_dynspec.mean(1),linewidth=4)
        plt.xlim(0,tsamp_ms*gulpsize)
        plt.subplot(2,2,4,facecolor='black')
        plt.step(bright_dynspec.sum(0),fobs,linewidth=1)

        cm = plt.get_cmap('Blues')
        for i in range(gulpsize):
            plt.step(bright_dynspec[:i+1,:].sum(0),fobs,color=cm(i/gulpsize),alpha=0.5,linewidth=1)
        plt.ylim(fobs[-1],fobs[0])
        plt.scatter([],[],c=[],vmin=0,vmax=tsamp_ms*gulpsize/1000,cmap='Blues')
        plt.title("SOURCE: " + bright_nvssnames[bright_idx] + "\nMJD: " + str(mjd) + "\nFNUM: " + bright_fnames[bright_idx],fontsize=20)
        plt.colorbar(label='Time (s)')

        plt.subplot(2,2,3)
        plt.imshow(bright_dynspec.transpose(),aspect='auto',extent=(0,tsamp_ms*gulpsize/1000,fobs[-1],fobs[0]))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.subplots_adjust(hspace=0,wspace=0)
        plt.suptitle("real",fontsize=25)
        if savestuff:
            plt.savefig(img_dir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_"+ str("outriggers_" if outriggers else "")+"realvisspeccal.png")

        plt.close()

        bright_dynspec = np.imag(dat.mean((1,3)))
        plt.figure(figsize=(16,12))
        plt.subplot(2,2,1,facecolor='black')
        plt.step(np.arange(gulpsize)*tsamp_ms,bright_dynspec.mean(1),linewidth=4)
        plt.xlim(0,tsamp_ms*gulpsize)
        plt.subplot(2,2,4,facecolor='black')
        plt.step(bright_dynspec.sum(0),fobs,linewidth=1)

        cm = plt.get_cmap('Blues')
        for i in range(gulpsize):
            plt.step(bright_dynspec[:i+1,:].sum(0),fobs,color=cm(i/gulpsize),alpha=0.5,linewidth=1)
        plt.ylim(fobs[-1],fobs[0])
        plt.scatter([],[],c=[],vmin=0,vmax=tsamp_ms*gulpsize/1000,cmap='Blues')
        plt.title("SOURCE: " + bright_nvssnames[bright_idx] + "\nMJD: " + str(mjd) + "\nFNUM: " + bright_fnames[bright_idx],fontsize=20)
        plt.colorbar(label='Time (s)')

        plt.subplot(2,2,3)
        plt.imshow(bright_dynspec.transpose(),aspect='auto',extent=(0,tsamp_ms*gulpsize/1000,fobs[-1],fobs[0]))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.subplots_adjust(hspace=0,wspace=0)
        plt.suptitle("imaginary",fontsize=25)

        if savestuff:
            plt.savefig(img_dir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_"+ str("outriggers_" if outriggers else "")+"imagvisspeccal.png")
        plt.close()
    
    print(bright_measfluxs)

    update_astrocal_table(bright_nvssnames,bright_fnames,bright_poserrs,bright_raerrs,bright_decerrs,bright_resid,outriggers,init=args.init_astrocal,resid_th=args.resid_th,exclude_table=exclude_table)
    update_speccal_table(bright_nvssnames,bright_fnames,bright_measfluxs,bright_measfluxerrs,bright_nvssfluxes,bright_resid,outriggers,init=args.init_speccal,resid_th=args.resid_th,exclude_table=exclude_table)
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('--init_astrocal',action='store_true',help='Initialize json astrometry table')
    parser.add_argument('--init_speccal',action='store_true',help='Initialize json flux cal table')
    parser.add_argument('--UTCday',type=str,help='UTC day to run fluxcal with in ISO format (e.g. 2024-06-12); if not given, uses the previous day',default=Time(Time.now().mjd,format='mjd').isot[:10])
    parser.add_argument('--numsources',type=int,help='Maximum number of sources to use for fluxcal, takes the brightest within 0.5 degrees of the current dec, default=10',default=10)
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=1)
    parser.add_argument('--image_size',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--buff',type=int,help='Radius in pixels around the NVSS position to search for peak pixel, default=5',default=5)
    parser.add_argument('--flagSWAVE',action='store_true',help='Flag channels when SWAVE template RFI is detected, which manifests as a 2 Hz sin wave over ~5 minutes of data')
    parser.add_argument('--flagBPASS',action='store_true',help='Flag channels when BPASS template RFI is detected, which is simpl comparison to bandpass mean in visibilities')
    parser.add_argument('--flagFRCBAND',action='store_true',help='Flag channels in FRC miltiary allocation 1435-1525 MHz')
    parser.add_argument('--flagBPASSBURST',action='store_true',help='Flag channel when BPASS template RFI is detected in any timestep, i.e. should detect pulsed narrowband RFI')
    parser.add_argument('--flagcorrs',type=int,nargs='+',default=[],help='List of sb nodes [0-15] to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagants',type=int,nargs='+',default=[],help='List of antennas to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--dec',type=float,help='Pointing declination to search for calibrators, ideally the one that the array has targeted over the day in question, default pulls the dec on the given day at 12:00:00 UTC from ETCD',default=180)
    parser.add_argument('--outriggers',action='store_true',help='Includes outrigger antennas in imaging')
    parser.add_argument('--resid_th',type=float,help='Maximum allowed normalized residual to include in astrometric or flux cal; default=0.2',default=0.2)
    args = parser.parse_args()
    main(args)
