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

from nsfrb.imaging import get_ra,get_RA_cutoff,stack_images
from matplotlib.patches import Ellipse
from nsfrb.config import *
import numpy as np
import csv
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

from nsfrb.planning import read_nvss,read_RFC
from dsautils.coordinates import create_WCS,get_declination,get_elevation
from astropy.coordinates import AltAz
from nsfrb.planning import find_fast_vis_label
from nsfrb.planning import nvss_cat
from nsfrb.config import tsamp as tsamp_ms
from nsfrb.config import IMAGE_SIZE,bmin,flagged_antennas,bad_antennas,pixperFWHM,NUM_CHANNELS
import json

def init_table(outriggers=False,astrocal_table=table_dir + "/NSFRB_astrocal.json",image_flux=False):
    print("Initializing table " + astrocal_table)
    arraykey = str('outriggers' if outriggers else 'core') + str("_image" if image_flux else "")
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

def update_speccal_table(bright_nvssnames,bright_nvsscoords,bright_fnames,bright_measfluxs,bright_measfluxerrs,bright_nvssfluxes,outriggers,speccal_table=table_dir + "/NSFRB_speccal.json",init=False,exclude_table=table_dir + "/NSFRB_excludecal.json",nsamps=nsamps,image_flux=False,fitresid_th = 0.1,target='',targetMJD=0.0,target_timerange=5,target_decrange=0.5):
    """
    This function updates the flux calibration table with the most
    recent NVSS observations.
    """
    #read current table
    if init:
        init_table(outriggers,speccal_table,image_flux=image_flux)
    f = open(speccal_table,"r")
    arraykey = str('outriggers' if outriggers else 'core') + str("_image" if image_flux else "")
    tab = json.load(f)
    f.close()

    #read sources to exclude
    if len(exclude_table) > 0:
        f = open(exclude_table,"r")
        ex_table = json.load(f)['NVSS_exclude']
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
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["nvss_ra"] = bright_nvsscoords[i].ra.value
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["nvss_dec"] = bright_nvsscoords[i].dec.value
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["dir"] = vis_dir + bright_nvssnames[i].replace(" ","") + "/"
        if len(glob.glob(tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["dir"]+"nsfrb_sb00_"+str(bright_fnames[i])+".out"))>0:
            tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["mjd"] = pipeline.read_raw_vis(tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["dir"]+"nsfrb_sb00_"+str(bright_fnames[i])+".out",get_header=True)[1]
        #tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["RMS_fit_residual"] = bright_resid[i]

    print("Sources in table:",tab[arraykey].keys())
    #update calibration curve
    allfluxs = []
    allfluxerrs = []
    allnvssfluxes = []
    #allresids = []
    if len(target)>0:
        target_coord = SkyCoord(target,unit=(u.hourangle,u.deg),frame='icrs')
        target_obstime = Time(targetMJD,format='mjd')
    for k in tab[arraykey].keys():
        for kk in tab[arraykey][k].keys():
            if str(k) not in ex_table:
                #if a target is given, check that sources are within range
                if len(target)>0 and np.abs(target_coord.dec.value - tab[arraykey][k][kk]["nvss_dec"])<target_decrange and ('mjd' not in tab[arraykey][k][kk].keys() or np.abs(targetMJD - tab[arraykey][k][kk]['mjd'])*24<target_timerange):
                    print("Including ",k)
                    allfluxs.append(tab[arraykey][k][kk]['meas_flux'])
                    allfluxerrs.append(tab[arraykey][k][kk]['meas_flux_error'])
                    allnvssfluxes.append(tab[arraykey][k][kk]['flux_mJy'])
                    #allresids.append(tab[arraykey][k][kk]['RMS_fit_residual'])
                elif len(target)==0:
                    allfluxs.append(tab[arraykey][k][kk]['meas_flux'])
                    allfluxerrs.append(tab[arraykey][k][kk]['meas_flux_error'])
                    allnvssfluxes.append(tab[arraykey][k][kk]['flux_mJy'])
                    #allresids.append(tab[arraykey][k][kk]['RMS_fit_residual'])
    allfluxs = np.array(allfluxs)
    allfluxerrs = np.array(allfluxerrs)
    allnvssfluxes = np.array(allnvssfluxes)
    #allresids = np.array(allresids)
    
    #print(allfluxs,allresids)
    if len(allfluxs)<2:
        f = open(speccal_table,"w")
        json.dump(tab,f)
        f.close()
        print("No remaining sources for spec cal")
        return
    badsoln = False
    try:
        #popt,pcov = np.polyfit(allfluxs[allresids<resid_th],allnvssfluxes[allresids<resid_th],1,full=False,cov=True,w=1/allresids[allresids<resid_th])
        popt,pcov = np.polyfit(allfluxs,allnvssfluxes,1,full=False,cov=True,w=allnvssfluxes)
        popterrs = np.sqrt([pcov[0,0],pcov[1,1]])
        pfunc = np.poly1d(popt)
        pfunc_down = np.poly1d(popt - popterrs)
        pfunc_up = np.poly1d(popt + popterrs)
        if len(target)>0:
            target_table = dict()
            target_table['target'] = target
            target_table['ra'] = target_coord.ra.value
            target_table['dec'] = target_coord.dec.value
            target_table['MJD'] = targetMJD
            target_table[arraykey + "_slope"] = float(popt[0])
            target_table[arraykey + "_slope_error"] = float(popterrs[0])
            target_table[arraykey + "_int"] = float(popt[1])
            target_table[arraykey + "_int_error"] = float(popterrs[1])
            print(target+" flux conversion fit: FLUX = (",popt[1],"+-",popterrs[1],") + (",popt[0],"+-",popterrs[0],")MEAS_FLUX")
        else:
            tab[arraykey + "_slope"] = float(popt[0])
            tab[arraykey + "_slope_error"] = float(popterrs[0])
            tab[arraykey + "_int"] = float(popt[1])
            tab[arraykey + "_int_error"] = float(popterrs[1])
        
            print("Updated flux conversion fit: FLUX = (",popt[1],"+-",popterrs[1],") + (",popt[0],"+-",popterrs[0],")MEAS_FLUX") 
        allresids = np.abs((allnvssfluxes - pfunc(allfluxs))/allnvssfluxes)
        if np.any(allresids > fitresid_th):
            Smin = np.nanpercentile(allnvssfluxes[allresids > fitresid_th],90)
            print("Estimating sensitivity limit from 90th percentile of sources with > ",100*fitresid_th,"% Flux Error: Smin=",Smin,"mJy")
            if len(target)>0:
                target_table[arraykey + "_Smin"] = Smin
            else:
                tab[arraykey + "_Smin"] = Smin
        else:
            if len(target)>0:
                target_table[arraykey + "_Smin"] = np.nan
            else:
                tab[arraykey + "_Smin"] = np.nan
            Smin = np.nan
    except Exception as exc:
        badsoln = True
        print("Flux cal linear fit failed with error:",exc)
        print("Not enough points for flux cal solution")
    f = open(speccal_table,"w")
    json.dump(tab,f)
    f.close()

    if len(target)>0:
        f = open(table_dir + "/NSFRB_" + target.replace(" ","") + "_speccal.json","w")
        json.dump(target_table,f)
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

    if not badsoln and not np.isnan(Smin):
        plt.axhline(Smin,color='purple',linestyle='--')
    #plt.yscale("log")
    #plt.xscale("log")
    plt.colorbar(label="Linear Fit Residual")
    plt.savefig(img_dir+str(target.replace(" ","") + "_" if len(target)>0 else "") + "NVSStotal_"+ str("image_" if image_flux else "") + str("outriggers_" if outriggers else "") + "speccal.png")
    plt.close()
    
    return

def update_astrocal_table(bright_nvssnames,bright_nvsscoords,bright_RAerrs_mas,bright_DECerrs_mas,bright_fnames,bright_poserrs,bright_raerrs,bright_decerrs,bright_resid,bright_gulpoffsets,bright_gulpoffset_times,outriggers,astrocal_table=table_dir + "/NSFRB_astrocal.json",init=False,resid_th=np.inf,exclude_table=table_dir + "/NSFRB_excludecal.json",target='',targetMJD=0.0,target_timerange=5,target_decrange=0.5):
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
        ex_table = json.load(f)['RFC_exclude']
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
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["RFC_RA_error_deg"] = bright_RAerrs_mas[i]/3600
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["RFC_DEC_error_deg"] = bright_DECerrs_mas[i]/3600
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["rfc_ra"] = bright_nvsscoords[i].ra.value
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["rfc_dec"] = bright_nvsscoords[i].dec.value
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["dir"] = vis_dir + bright_nvssnames[i].replace(" ","") + "/"
        tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["pixoffsets"] = (list(bright_gulpoffsets[i].astype(float)),list(np.array(bright_gulpoffset_times[i])-bright_gulpoffset_times[i][0]))
        if len(glob.glob(tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["dir"]+"nsfrb_sb00_"+str(bright_fnames[i])+".out"))>0:
            tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["mjd"] = pipeline.read_raw_vis(tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["dir"]+"nsfrb_sb00_"+str(bright_fnames[i])+".out",get_header=True)[1]
        print(tab[arraykey][bright_nvssnames[i]][bright_fnames[i]]["pixoffsets"])
    #update total RMS errors
    allposerrs = []
    allRAerrs = []
    allDECerrs = []
    allresids = []
    allRFCRAerrs = []
    allRFCDECerrs = []
    allgulpoffsets = []
    allgulpoffset_times = []
    allgulpresids = []
    if len(target)>0:
        target_coord = SkyCoord(target,unit=(u.hourangle,u.deg),frame='icrs')
        target_obstime = Time(targetMJD,format='mjd')
    for k in tab[arraykey].keys():
        for kk in tab[arraykey][k].keys():
            if str(k) not in ex_table and tab[arraykey][k][kk]['RMS_fit_residual'] < resid_th:
                if len(target)>0 and np.abs(target_coord.dec.value - tab[arraykey][k][kk]["rfc_dec"])<target_decrange and ('mjd' not in tab[arraykey][k][kk].keys() or np.abs(targetMJD - tab[arraykey][k][kk]['mjd'])*24<target_timerange):
                    allposerrs.append(tab[arraykey][k][kk]['position_error_deg'])
                    allDECerrs.append(tab[arraykey][k][kk]['DEC_error_deg'])
                    allRAerrs.append(tab[arraykey][k][kk]['RA_error_deg'])
                    allresids.append(tab[arraykey][k][kk]['RMS_fit_residual'])
                    allRFCRAerrs.append(tab[arraykey][k][kk]['RFC_RA_error_deg'])
                    allRFCDECerrs.append(tab[arraykey][k][kk]['RFC_DEC_error_deg'])
                    allgulpoffsets = np.concatenate([allgulpoffsets,tab[arraykey][k][kk]["pixoffsets"][0]])
                    allgulpoffset_times = np.concatenate([allgulpoffset_times,np.array(tab[arraykey][k][kk]["pixoffsets"][1]) - tab[arraykey][k][kk]["pixoffsets"][1][0]])
                    allgulpresids = np.concatenate([allgulpresids,[tab[arraykey][k][kk]['RMS_fit_residual']]*len(tab[arraykey][k][kk]["pixoffsets"][0])])
                elif len(target)==0:
                    allposerrs.append(tab[arraykey][k][kk]['position_error_deg'])
                    allDECerrs.append(tab[arraykey][k][kk]['DEC_error_deg'])
                    allRAerrs.append(tab[arraykey][k][kk]['RA_error_deg'])
                    allresids.append(tab[arraykey][k][kk]['RMS_fit_residual'])
                    allRFCRAerrs.append(tab[arraykey][k][kk]['RFC_RA_error_deg'])
                    allRFCDECerrs.append(tab[arraykey][k][kk]['RFC_DEC_error_deg'])
                    allgulpoffsets = np.concatenate([allgulpoffsets,tab[arraykey][k][kk]["pixoffsets"][0]])
                    allgulpoffset_times = np.concatenate([allgulpoffset_times,np.array(tab[arraykey][k][kk]["pixoffsets"][1])-tab[arraykey][k][kk]["pixoffsets"][1][0]])
                    allgulpresids = np.concatenate([allgulpresids,[tab[arraykey][k][kk]['RMS_fit_residual']]*len(tab[arraykey][k][kk]["pixoffsets"][0])])
    allposerrs = np.array(allposerrs)
    allRAerrs = np.array(allRAerrs)
    allDECerrs = np.array(allDECerrs)
    allresids = np.array(allresids)
    allRFCRAerrs = np.array(allRFCRAerrs)
    allRFCDECerrs = np.array(allRFCDECerrs)
    print(allposerrs,allresids)
    if len(allposerrs)<1:
        f = open(astrocal_table,"w")
        json.dump(tab,f)
        f.close()
        print("No remaining sources for astro cal")
        return
    if len(target)>0:
        target_table = dict()
        target_table['target'] = target
        target_table['ra'] = target_coord.ra.value
        target_table['dec'] = target_coord.dec.value
        target_table['MJD'] = targetMJD
        target_table[str('outriggers' if outriggers else 'core') + "_position_error_deg"] = np.sqrt(np.average(np.array(allposerrs)**2,weights=1/allresids))
        target_table[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"] = (np.average(allRAerrs,weights=1/allresids) if len(allRAerrs)>1 else 0)
        target_table[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"] = (np.average(allDECerrs,weights=1/allresids) if len(allDECerrs)>1 else 0)
        target_table[str('outriggers' if outriggers else 'core') + "_RA_error_deg"] = np.sqrt(np.average((np.array(allRAerrs) - tab[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"])**2 + np.array(allRFCRAerrs)**2,weights=1/allresids))
        target_table[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"] = np.sqrt(np.average((np.array(allDECerrs) - tab[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"])**2 + np.array(allRFCDECerrs)**2,weights=1/allresids))
        print(target + " Total Position Error:",target_table[str('outriggers' if outriggers else 'core') + "_position_error_deg"])
        print(target + " RA Offset:",target_table[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"])
        print(target + " DEC Offset:",target_table[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"])
        print(target + " RA Error:",target_table[str('outriggers' if outriggers else 'core') + "_RA_error_deg"])
        print(target + " DEC Error:",target_table[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"])

        #RA pixel drift fit
        if len(allgulpoffsets)>=2:
            def slopefit(x,m):
                return m*x
            popt_gulp,TMP = curve_fit(slopefit,allgulpoffset_times,allgulpoffsets,sigma=allgulpresids)                                        
            #popt_gulp = np.polyfit(allgulpoffset_times,allgulpoffsets,1,w=1/allgulpresids)
            popt_gulp = (float(popt_gulp),0)
            print("Gulp offset curve:",popt_gulp)
            target_table[str('outriggers' if outriggers else 'core') + "_gulp_RA_drift_slope"] = popt_gulp[0]
            target_table[str('outriggers' if outriggers else 'core') + "_gulp_RA_drift_int"] = popt_gulp[1]

    else:
        tab[str('outriggers' if outriggers else 'core') + "_position_error_deg"] = np.sqrt(np.average(np.array(allposerrs)**2,weights=1/allresids))
        tab[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"] = (np.average(allRAerrs,weights=1/allresids) if len(allRAerrs)>1 else 0)
        tab[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"] = (np.average(allDECerrs,weights=1/allresids) if len(allDECerrs)>1 else 0)
        tab[str('outriggers' if outriggers else 'core') + "_RA_error_deg"] = np.sqrt(np.average((np.array(allRAerrs) - tab[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"])**2 + np.array(allRFCRAerrs)**2,weights=1/allresids))
        tab[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"] = np.sqrt(np.average((np.array(allDECerrs) - tab[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"])**2 + np.array(allRFCDECerrs)**2,weights=1/allresids))
        print("Updated Total Position Error:",tab[str('outriggers' if outriggers else 'core') + "_position_error_deg"])
        print("Updated RA Offset:",tab[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"])
        print("Updated DEC Offset:",tab[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"])
        print("Updated RA Error:",tab[str('outriggers' if outriggers else 'core') + "_RA_error_deg"])
        print("Updated DEC Error:",tab[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"])


        #RA pixel drift fit
        if len(allgulpoffsets)>=2:
            def slopefit(x,m):
                return m*x
            popt_gulp,TMP = curve_fit(slopefit,allgulpoffset_times,allgulpoffsets,sigma=allgulpresids)
            #popt_gulp = np.polyfit(allgulpoffset_times,allgulpoffsets,1,w=1/allgulpresids)
            popt_gulp = (float(popt_gulp),0)
            print("Gulp offset curve:",popt_gulp)
            tab[str('outriggers' if outriggers else 'core') + "_gulp_RA_drift_slope"] = popt_gulp[0]
            tab[str('outriggers' if outriggers else 'core') + "_gulp_RA_drift_int"] = popt_gulp[1]
    
    
    f = open(astrocal_table,"w")
    json.dump(tab,f)
    f.close()

    if len(target)>0:
        f = open(table_dir + "/NSFRB_" + target.replace(" ","") + "_astrocal.json","w")
        json.dump(target_table,f)
        f.close()

    #plot results
    plt.figure(figsize=(12,12))
    plt.axvline(0,color='grey',alpha=0.5)
    plt.axhline(0,color='grey',alpha=0.5)
    plt.scatter(allRAerrs*60,allDECerrs*60,c=allresids,marker="o",s=100,cmap='copper')
    if len(target)>0:
        plt.errorbar(target_table[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"]*60,target_table[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"]*60,
                xerr=60*target_table[str('outriggers' if outriggers else 'core') + "_RA_error_deg"],yerr=60*target_table[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"],capsize=10,color='red')
    else:
        plt.errorbar(tab[str('outriggers' if outriggers else 'core') + "_RA_offset_deg"]*60,tab[str('outriggers' if outriggers else 'core') + "_DEC_offset_deg"]*60,
                xerr=60*tab[str('outriggers' if outriggers else 'core') + "_RA_error_deg"],yerr=60*tab[str('outriggers' if outriggers else 'core') + "_DEC_error_deg"],capsize=10,color='red')
    plt.xlabel("RA Error (arcmin)")
    plt.ylabel("DEC Error (arcmin)")
    plt.title("Last Updated: " + Time.now().isot)
    plt.colorbar(label="Normalized RMS Residual")
    plt.xlim(-np.max(np.abs(allRAerrs*60))*1.1,np.max(np.abs(allRAerrs*60))*1.1)
    plt.ylim(-np.max(np.abs(allDECerrs*60))*1.1,np.max(np.abs(allDECerrs*60))*1.1)
    #plt.xlim(-np.max(allposerrs)*60*1.1,np.max(allposerrs)*60*1.1)
    #plt.ylim(-np.max(allposerrs)*60*1.1,np.max(allposerrs)*60*1.1)
    plt.savefig(img_dir+str(target.replace(" ","") + "_" if len(target)>0 else "") +"RFCtotal_"+ str("outriggers_" if outriggers else "")+"astrocal.png")
    plt.close()

    if len(allgulpoffsets)>=2:
        plt.figure(figsize=(12,12))
        plt.scatter(allgulpoffset_times,allgulpoffsets,c=allgulpresids,marker='o',s=100,cmap='copper')
        if len(target)>0:
            plt.plot(allgulpoffset_times,target_table[str('outriggers' if outriggers else 'core') + "_gulp_RA_drift_int"] + target_table[str('outriggers' if outriggers else 'core') + "_gulp_RA_drift_slope"]*allgulpoffset_times,'--',color='red')
        else:
            plt.plot(allgulpoffset_times,tab[str('outriggers' if outriggers else 'core') + "_gulp_RA_drift_int"] + tab[str('outriggers' if outriggers else 'core') + "_gulp_RA_drift_slope"]*allgulpoffset_times,'--',color='red')
        plt.colorbar(label="Normalized RMS Residual")
        plt.xlabel("Time (s)")
        plt.ylabel("RA Offset (pixels)")
        plt.savefig(img_dir+str(target.replace(" ","") + "_" if len(target)>0 else "") +"RFCtotal_"+ str("outriggers_" if outriggers else "")+"astrocal_drift.png")
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


    return 

def astrocal(args):
    """
    Main routine for astrometric calibration with RFC sources
    """


    # find brightest continuum sources at the current declination # dec 16 continuum sources
    reftime = Time.now()#Time(args.UTCday + "T12:00:00",format='isot')
    print("Reference time:",reftime.isot)
    allnames,allcoords,allRA_errs_mas,allDEC_errs_mas,allSfluxes,allCfluxes,allXfluxes,allUfluxes,allKfluxes = read_RFC()
    if args.search_dec == 180:
        elev = get_best_elev(reftime)#get_elevation(reftime)
        search_dec = get_declination(elev).value
    else:
        search_dec = args.search_dec

    #read sources to exclude
    exclude_table = table_dir + "/NSFRB_excludecal.json"
    if len(exclude_table) > 0:
        f = open(exclude_table,"r")
        ex_table = json.load(f)['RFC_exclude']
        f.close()
    else:
        ex_table =  []
    print("sources to exclude:",ex_table)

    decrange= 0.5
    condition = np.abs(allcoords.dec.value-search_dec) <decrange
    vis_coords = allcoords[condition]
    vis_Sfluxes = allSfluxes[condition]
    vis_RAerrs_mas = (allRA_errs_mas[condition])*np.cos(vis_coords.dec.to(u.rad).value) #ACCOUNT FOR COS(DEC) FACTOR
    vis_DECerrs_mas = allDEC_errs_mas[condition]
    
    #single RFC source
    if len(args.rfc)>0:
        idxs = []
        idxnames = []
        for i in range(len(args.rfc)):
            print(args.rfc[i])
            print(SkyCoord(args.rfc[i],unit=(u.hourangle,u.deg),frame='icrs').separation(vis_coords))
            print(vis_coords)
            idx = np.argmin(SkyCoord(args.rfc[i],unit=(u.hourangle,u.deg),frame='icrs').separation(vis_coords).value)
            if SkyCoord(args.rfc[i],unit=(u.hourangle,u.deg),frame='icrs').separation(vis_coords).to(u.arcsecond).value[idx] < 30:
                idxs.append(np.argmin(SkyCoord(args.rfc[i],unit=(u.hourangle,u.deg),frame='icrs').separation(vis_coords).value))
                idxnames.append(args.rfc[i])
            else:
                print(args.rfc[i],"not found")
        idxs = np.array(idxs)
        bright_coords = vis_coords[idxs]
        bright_Sfluxes = vis_Sfluxes[idxs]
        bright_RAerrs_mas = vis_RAerrs_mas[idxs]
        bright_DECerrs_mas = vis_DECerrs_mas[idxs]
        print("Running astrometric calibration pipeline with RFC sources ",idxnames)
    else:
        
        fluxth = np.sort(vis_Sfluxes)[-args.numsources]
        bright_coords = vis_coords[vis_Sfluxes>=fluxth]
        bright_RAerrs_mas = vis_RAerrs_mas[vis_Sfluxes>=fluxth]
        bright_DECerrs_mas = vis_DECerrs_mas[vis_Sfluxes>=fluxth]
        bright_Sfluxes = vis_Sfluxes[vis_Sfluxes>=fluxth]
        print("Running astrometric calibration pipeline with " + str(len(bright_coords)) + " brightest RFC sources at dec=" + str(search_dec) + ":")

   
    #find the files within the timestamp
    besttime = []
    bright_names = []
    bright_fnames = []
    bright_offsets = []
    for i in range(len(bright_coords)):
        timeax = Time(int(reftime.mjd) - np.linspace(0,24,24)[::-1]/24,format='mjd')
        DSA = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg,height=Height*u.m)
        name = str('RFC J{RH:02d}{RM:02d}{RS:02d}'.format(RH=int(bright_coords[i].ra.hms.h),
                                                               RM=int(bright_coords[i].ra.hms.m),
                                                               RS=int(bright_coords[i].ra.hms.s)) +
                           str("+" if bright_coords[i].dec>=0 else "-") +
                           '{DD:02d}{DM:02d}{DS:02d}'.format(DD=int(bright_coords[i].dec.dms.d),
                                                               DM=int(bright_coords[i].dec.dms.m),
                                                               DS=int(bright_coords[i].dec.dms.s)))

        bright_names.append(name)

        if name in ex_table:
            besttime.append(-1)
            print("Excluding " + name)
            print("")
            bright_fnames.append(-1)
            bright_offsets.append(-1)
            continue

        hourvis = SkyCoord(bright_coords[i],location=DSA,obstime=timeax)

        #narrow to best minute
        antpos = hourvis.transform_to(AltAz)
        timeax = Time(timeax[np.argmax(antpos.alt.value)].mjd + np.linspace(-1,1,24)/24,format='mjd')
        minvis = SkyCoord(bright_coords[i],location=DSA,obstime=timeax)

        #narrow to best second
        antpos = minvis.transform_to(AltAz)
        timeax = Time(timeax[np.argmax(antpos.alt.value)].mjd + np.linspace(-1/60,1/60,24)/24,format='mjd')
        secvis = SkyCoord(bright_coords[i],location=DSA,obstime=timeax)

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

    bright_poserrs = []
    bright_raerrs = []
    bright_decerrs = []
    bright_pixcoords = []
    bright_pixs = []
    close_pixs = []
    bright_resid = []
    image_size=args.image_size#1101 #8001
    gulpsize = nsamps
    nchan_per_node=nchans_per_node = args.nchans_per_node
    outriggers = args.outriggers
    ref_wav=0.20
    bmin=args.bmin
    savestuff = True
    for bright_idx in range(len(bright_fnames)):
        if bright_fnames[bright_idx] == -1:
            bright_poserrs.append(-1)
            bright_raerrs.append(-1)
            bright_decerrs.append(-1)
            bright_pixcoords.append(-1)
            bright_pixs.append(-1)
            close_pixs.append(-1)
            bright_resid.append(-1)
            print("Excluding " + bright_names[bright_idx])
            continue


        print("Reading data for "+ bright_names[bright_idx])
        fnum = int(bright_fnames[bright_idx])
        gulps = np.arange(bright_offsets[bright_idx]//gulpsize,min([(bright_offsets[bright_idx]//gulpsize)+args.ngulps,90]),dtype=int)
        if len(gulps)<args.ngulps:
            gulps = np.concatenate([np.arange(max([0,bright_offsets[bright_idx]//gulpsize - (args.ngulps-len(gulps))]),bright_offsets[bright_idx]//gulpsize),gulps])
            gulps = np.unique(gulps)
        ngulps = len(gulps)
        if len(bright_poserrs)==0 and ngulps>=5:
            bright_gulpoffsets = []
            bright_gulpoffset_times = []
        elif np.all(np.array(bright_poserrs)==-1) and ngulps>=5:
            bright_gulpoffsets = [-1]*len(bright_poserrs)
            bright_gulpoffset_times = [-1]*len(bright_poserrs)
        print(bright_idx,fnum,gulps)
       
        g=0
        
        #get RA cutoff
        """
        if ngulps>1:
            copydir = vis_dir + bright_names[bright_idx].replace(" ","") + "/"
            if len(glob.glob(copydir + "nsfrb_sb00_" + str(fnum) + ".out")) == 0:
                sb,mjd,dec = pipeline.read_raw_vis(vis_dir + "/lxd110h03/nsfrb_sb00_" + str(fnum) + ".out",get_header=True)
            else:
                sb,mjd,dec = pipeline.read_raw_vis(copydir + "nsfrb_sb00_" + str(fnum) + ".out",get_header=True)
            ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + (gulps[0]*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False)
        
            racutoff_ = get_RA_cutoff(dec,tsamp_ms*gulpsize,np.abs(ra_grid_2D[0,1]-ra_grid_2D[0,0]))
            print("RA cutoff:",racutoff_,"pixels")
            min_gridsize = int(gridsize - racutoff_*(args.ngulps - 1))
            full_img = np.zeros((image_size,min_gridsize))
        else:
        """
        min_gridsize = image_size
        full_img = np.zeros((image_size,image_size,ngulps))#,gulpsize,16*nchan_per_node))
        #full_img[:,:,:,:] = 0
        if ngulps >= 5:
            full_img = np.zeros((image_size,image_size,ngulps))
            RAs_for_fit = []
            DECs_for_fit = []
        else:
            full_img = np.zeros((image_size,image_size,ngulps))
        print("Image shape:",full_img.shape)
        buff = args.buff#50

        for gulp in gulps:#[77,78,79,80,81]:##0,45,75]:#range(3):

            dat = None
            sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
            corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
            copydir = vis_dir + bright_names[bright_idx].replace(" ","") + "/"
            os.system("mkdir " + copydir)
            for i in range(16):
                try:
                    if len(glob.glob(copydir + "nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out")) == 0:
                        os.system("cp " + vis_dir + "/lxd110" + corrs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out " + copydir)
                        dat_i,sb,mjd,dec = pipeline.read_raw_vis(vis_dir + "/lxd110" + corrs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=gulp,headersize=16)
                    else:
                        print(copydir + "nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out")
                        dat_i,sb,mjd,dec = pipeline.read_raw_vis(copydir + "nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=gulp,headersize=16)

                    print(mjd,dec,sb)
                    print(dat_i.shape)

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
                bright_resid.append(-1)
                print("Excluding " + bright_names[bright_idx])
                continue

            print("Getting UVW params...")
            test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
            pt_dec = dec*np.pi/180.
            bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
            #ff = 1.53-np.arange(8192)*0.25/8192
            #fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
            fobs = (1E-3)*(np.reshape(freq_axis_fullres,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1))


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
            U = UVW[0,:,1]
            V = UVW[0,:,0]
            W = UVW[0,:,2]

            uv_diag=np.max(np.sqrt(U**2 + V**2))
            pixel_resolution = (lambdaref/uv_diag/pixperFWHM)
            dat[np.isnan(dat)] = 0
            
            
            if len(gulps) >1 and g == 0:
                ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + (gulps[0]*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)
                decidx = np.argmin(np.abs(dec_grid_2D[:,0]-dec))
                print("Estimated pixel size:",np.abs(ra_grid_2D[decidx,1]-ra_grid_2D[decidx,0]))
                if ngulps >= 5:
                    racutoff_ = 0
                    print("Running linear fit for RA cutoff")
                else:
                    racutoff_ = get_RA_cutoff(dec,tsamp_ms*gulpsize,np.abs(ra_grid_2D[decidx,1]-ra_grid_2D[decidx,0]))
                    print("Estimating RA cutoff:",racutoff_,"pixels")
                min_gridsize = int(gridsize - racutoff_*(ngulps - 1))
                if ngulps >= 5:
                    full_img = np.zeros((image_size,min_gridsize,ngulps))
                else:
                    full_img = np.zeros((image_size,min_gridsize))
                print("Updated image size:",full_img.shape)
            
            #for i in range(dat.shape[0]):
            for j in range(len(corrs)):
                for k in range(dat.shape[-1]):
                    for jj in range(nchans_per_node):
                        if ngulps>=5:
                            full_img[:,:,g] +=  revised_robust_image(dat[:,:,(j*nchans_per_node) + jj,k],
                                                   U/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   V/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                    image_size,robust=args.robust,pixel_resolution=pixel_resolution)[:,g*racutoff_:g*racutoff_ + min_gridsize]
                        elif ngulps>1:
                            full_img += revised_robust_image(dat[:,:,(j*nchans_per_node) + jj,k],
                                                   U/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   V/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                    image_size,robust=args.robust,pixel_resolution=pixel_resolution)[:,g*racutoff_:g*racutoff_ + min_gridsize]#[:,image_size-min_gridsize - racutoff_*(ngulps - 1 - g):image_size-racutoff_*(ngulps- 1 - g)]
                        else:
                            full_img += revised_robust_image(dat[:,:,(j*nchans_per_node) + jj,k],
                                                   U/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   V/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   image_size,robust=args.robust,pixel_resolution=pixel_resolution)
                        #full_img[:,:,(g*gulpsize) + i,(j*nchans_per_node) + jj]  += tmpimg
            

            #run fit on single gulp 
            if ngulps>=5:
                #use the trimmed ra and dec axes so this correction will be appended to RA_cutoff
                """
                ra_grid_2D_g,dec_grid_2D_g,elev = uv_to_pix(mjd + (gulps[g]*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)
                ra_grid_2D_g = ra_grid_2D_g[:,image_size-min_gridsize - racutoff_*(ngulps- 1 - g):image_size-racutoff_*(ngulps - 1 - g)]
                dec_grid_2D_g = dec_grid_2D_g[:,image_size-min_gridsize - racutoff_*(ngulps- 1 - g):image_size-racutoff_*(ngulps - 1 - g)]
                """
                ra_grid_2D_g = ra_grid_2D
                dec_grid_2D_g = dec_grid_2D
                closepix = np.unravel_index(np.argmin(bright_coords[bright_idx].separation(SkyCoord(ra_grid_2D_g*u.deg,
                                                                                        dec_grid_2D_g*u.deg,frame='icrs'))),ra_grid_2D_g.shape)
                bbox = (max([closepix[0]-buff,0]),
                        min([closepix[0]+buff+1,image_size]),
                        max([closepix[1]-buff,0]),
                        min([closepix[1]+buff+1,min_gridsize]))
                input_img = (full_img[:,image_size-min_gridsize - racutoff_*(ngulps- 1 - g):image_size-racutoff_*(ngulps - 1 - g)])[bbox[0]:bbox[1],bbox[2]:bbox[3],g]#.mean((2,3))
                print(input_img)
                input_img[np.isnan(input_img)] = np.nanmedian(input_img)

                peakpix = np.unravel_index(np.argmax(input_img),(bbox[1]-bbox[0],bbox[3]-bbox[2]))
                bright_pix = (peakpix[0] + bbox[0] ,peakpix[1] + bbox[2])
                bright_pixcoord = SkyCoord(ra_grid_2D_g[bright_pix[0],bright_pix[1]]*u.deg,dec_grid_2D_g[bright_pix[0],bright_pix[1]]*u.deg,frame='icrs')


                #fit with an ellipse
                input_ra_grid_2D = ra_grid_2D_g[bbox[0]:bbox[1],bbox[2]:bbox[3]]
                input_dec_grid_2D = dec_grid_2D_g[bbox[0]:bbox[1],bbox[2]:bbox[3]]
                input_sigma = 1/np.sqrt(np.clip(input_img,1e-10,np.inf))
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
                print("Gulp " + str(g) + "  Optimal Parameters:")
                print("Position:",bright_pixcoord.ra.to(u.hourangle),r'+-',(param_errs[0]*u.deg).to(u.arcsecond),bright_pixcoord.dec,r'+-',(param_errs[1]*u.deg).to(u.arcsecond))
                print("Semimajor axis:",(0.5*popt[2]*u.deg).to(u.arcmin))
                print("Semiminor axis:",(0.5*popt[3]*u.deg).to(u.arcmin))
                print("Angle (counterclockwise from vertical):",(-popt[4]*u.rad).to(u.deg))
                print("Errors:",np.sqrt(pcov[np.arange(pcov.shape[0]),np.arange(pcov.shape[1])]))
                RAs_for_fit.append(bright_pixcoord.ra.value)
                DECs_for_fit.append(bright_pixcoord.dec.value)
            g += 1

        full_img /= (gulpsize*ngulps)
        if dat is None or int(dec) != int(search_dec):
            continue
        np.save(copydir+bright_names[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_" + str("outriggers_" if outriggers else "") + "image.npy",full_img)
        #re-align gulps
        ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + (gulps[0]*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)
        if ngulps >=5:
            #get offset in pixels from the first gulp
            decidx = np.argmin(np.abs(dec_grid_2D[:,0]-dec))
            pixsize = np.abs(ra_grid_2D[decidx,1]-ra_grid_2D[decidx,0])
            cutoff_offsets = np.array((np.array(RAs_for_fit) - RAs_for_fit[0])/pixsize,dtype=int)
            print("Cutoff corrections:",RAs_for_fit,cutoff_offsets,pixsize)

            #use maximum cutoff to get a new min_gridsize
            min_gridsize_new = min_gridsize - np.nanmax(np.abs(cutoff_offsets))
            peakoffsetidx = np.nanargmax(np.abs(cutoff_offsets))
            peakoffset = cutoff_offsets[peakoffsetidx]
            full_img_new = np.zeros((image_size,min_gridsize_new))

            #align the reference and peak offset images first
            if peakoffset>=0:
                full_img_new += full_img[:,np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new,0]
                ra_grid_2D = ra_grid_2D[:,np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new]
                dec_grid_2D = dec_grid_2D[:,np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new]
                full_img_new += full_img[:,(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset)),peakoffsetidx]
            else:
                full_img_new += full_img[:,np.abs(peakoffset):np.abs(peakoffset)+min_gridsize_new,peakoffsetidx]
                full_img_new += full_img[:,(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset)),0]
                ra_grid_2D = ra_grid_2D[:,(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset))]
                dec_grid_2D = dec_grid_2D[:,(min_gridsize - np.abs(peakoffset) - min_gridsize_new):(min_gridsize - np.abs(peakoffset))]
            #now the rest
            for g in range(1,len(gulps)):
                if g != peakoffsetidx:
                    if peakoffset>=0 and cutoff_offsets[g]>=0:
                        print(g,"case 1",peakoffsetidx,cutoff_offsets[g])
                        full_img_new += full_img[:,np.abs(peakoffset)-np.abs(cutoff_offsets[g]):np.abs(peakoffset)-np.abs(cutoff_offsets[g])+min_gridsize_new,g]
                    elif peakoffset<0 and cutoff_offsets[g]<0:
                        print(g,"case 2",peakoffsetidx,cutoff_offsets[g])
                        full_img_new += full_img[:,(min_gridsize - np.abs(peakoffset) + np.abs(cutoff_offsets[g]) - min_gridsize_new):(min_gridsize - np.abs(peakoffset) + np.abs(cutoff_offsets[g])),g]
                    elif peakoffset>=0 and cutoff_offsets[g]<0:
                        print(g,"case 3",peakoffsetidx,cutoff_offsets[g])
                        full_img_new[:,:min([np.abs(peakoffset)+np.abs(cutoff_offsets[g])+min_gridsize_new,min_gridsize])-(np.abs(peakoffset)+np.abs(cutoff_offsets[g]))] += full_img[:,np.abs(peakoffset)+np.abs(cutoff_offsets[g]):min([np.abs(peakoffset)+np.abs(cutoff_offsets[g])+min_gridsize_new,min_gridsize]),g]
                    elif peakoffset<0 and cutoff_offsets[g]>=0:
                        print(g,"case 4",peakoffsetidx,cutoff_offsets[g])
                        full_img_new[:,-(((min_gridsize - (np.abs(peakoffset)+np.abs(cutoff_offsets[g]))))-max([0,(min_gridsize - (np.abs(peakoffset)+np.abs(cutoff_offsets[g])) - min_gridsize_new)])):] += full_img[:,max([0,(min_gridsize - (np.abs(peakoffset)+np.abs(cutoff_offsets[g])) - min_gridsize_new)]):(min_gridsize - (np.abs(peakoffset)+np.abs(cutoff_offsets[g]))),g]
            
            full_img = full_img_new
            min_gridsize = min_gridsize_new
            bright_gulpoffsets.append(cutoff_offsets)
            bright_gulpoffset_times.append((np.array(gulps)-gulps[0])*tsamp_ms*gulpsize/1000)
        elif ngulps>1:
            ra_grid_2D = ra_grid_2D[:,image_size-min_gridsize - racutoff_*(ngulps- 1 - 0):image_size-racutoff_*(ngulps - 1 - 0)]
            dec_grid_2D = dec_grid_2D[:,image_size-min_gridsize - racutoff_*(ngulps- 1 - 0):image_size-racutoff_*(ngulps - 1 - 0)]
        
            
        # ASTROMETRIC TEST
        # find the peak pixel in the vicinity of the coordinates
        buff = args.buff#50
        


        if (bright_coords[bright_idx].ra.value < np.min(ra_grid_2D) or
            bright_coords[bright_idx].ra.value > np.max(ra_grid_2D) or
            bright_coords[bright_idx].dec.value < np.min(dec_grid_2D) or
            bright_coords[bright_idx].dec.value > np.max(dec_grid_2D)):
            bright_poserrs.append(-1)
            bright_raerrs.append(-1)
            bright_decerrs.append(-1)
            bright_pixcoords.append(-1)
            bright_pixs.append(-1)
            close_pixs.append(-1)
            bright_resid.append(-1)
            bright_gulpoffsets.append(-1)
            bright_gulpoffset_times.append(-1)
            print("Excluding " + bright_names[bright_idx])
            continue
        if ngulps>=5:
            bright_gulpoffsets.append(cutoff_offsets)
            bright_gulpoffset_times.append(np.array(gulps)*tsamp_ms*gulpsize/1000) #s
        

        closepix = np.unravel_index(np.argmin(bright_coords[bright_idx].separation(SkyCoord(ra_grid_2D*u.deg,
                                                                                        dec_grid_2D*u.deg,frame='icrs'))),ra_grid_2D.shape)
        bbox = (max([closepix[0]-buff,0]),
                min([closepix[0]+buff+1,image_size]),
                max([closepix[1]-buff,0]),
                min([closepix[1]+buff+1,min_gridsize]))
        input_img = full_img[bbox[0]:bbox[1],bbox[2]:bbox[3]]#.mean((2,3))
        print(input_img)
        input_img[np.isnan(input_img)] = np.nanmedian(input_img)

        peakpix = np.unravel_index(np.argmax(input_img),(bbox[1]-bbox[0],bbox[3]-bbox[2]))
        bright_pix = (peakpix[0] + bbox[0] ,peakpix[1] + bbox[2])
        bright_pixcoord = SkyCoord(ra_grid_2D[bright_pix[0],bright_pix[1]]*u.deg,dec_grid_2D[bright_pix[0],bright_pix[1]]*u.deg,frame='icrs')


        #fit with an ellipse
        input_ra_grid_2D = ra_grid_2D[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        input_dec_grid_2D = dec_grid_2D[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        input_sigma = 1/np.sqrt(np.clip(input_img,1e-10,np.inf))
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
        pos_err = np.sqrt((bright_DECerrs_mas[bright_idx]/60)**2 + (bright_RAerrs_mas[bright_idx]/60)**2 + (param_errs[0]*60)**2 + (param_errs[1]*60)**2 + bright_pixcoord.separation(bright_coords[bright_idx]).to(u.arcmin).value**2)
        print("Total Position Error:",pos_err,"arcminutes")
        ra_pos_err = bright_pixcoord.ra.value - bright_coords[bright_idx].ra.value
        dec_pos_err = bright_pixcoord.dec.value - bright_coords[bright_idx].dec.value
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
        vmin=args.vmin#None#-np.nanmax(full_img.mean((2,3)))/4#-1
        vmax=args.vmax#None#np.nanmax(full_img.mean((2,3)))/4#1 #0.2#1

        plt.title("SOURCE: " + bright_names[bright_idx] + "\nMJD: " + str(mjd) + "\nFNUM: " + bright_fnames[bright_idx],fontsize=20)
        plt.scatter(ra_grid_2D.flatten(),dec_grid_2D.flatten(),c=full_img.flatten(),alpha=0.5,cmap='cool',marker='s',s=100,vmin=vmin,vmax=vmax)#0.8*np.nanmax((full_img.mean((2,3)))))
        plt.scatter(bright_coords[bright_idx].ra.to(u.deg).value,bright_coords[bright_idx].dec.to(u.deg).value,marker='o',s=1000,edgecolors='red',linewidth=4,facecolors="none",alpha=0.8)
        plt.scatter(RAs_for_fit,DECs_for_fit,marker='o',s=1000,edgecolors='red',linewidth=1,facecolors="none",alpha=0.5)
        plt.errorbar(bright_coords[bright_idx].ra.to(u.deg).value,bright_coords[bright_idx].dec.to(u.deg).value,xerr=bright_RAerrs_mas[bright_idx]/3600,yerr=bright_DECerrs_mas[bright_idx]/3600,color='red',elinewidth=3)
        plt.xlim(bright_coords[bright_idx].ra.to(u.deg).value-(0.3 if outriggers else 1.5),bright_coords[bright_idx].ra.to(u.deg).value+(0.3 if outriggers else 1.5))
        plt.ylim(bright_coords[bright_idx].dec.to(u.deg).value-(0.3 if outriggers else 1.5),bright_coords[bright_idx].dec.to(u.deg).value+(0.3 if outriggers else 1.5))
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
        plt.savefig(img_dir+bright_names[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_"+ str("outriggers_" if outriggers else "")+"astrocal.png")
        plt.close()

    update_astrocal_table(bright_names,bright_coords,bright_RAerrs_mas,bright_DECerrs_mas,bright_fnames,bright_poserrs,bright_raerrs,bright_decerrs,bright_resid,bright_gulpoffsets,bright_gulpoffset_times,outriggers,init=args.init_astrocal,resid_th=args.astroresid_th,exclude_table=exclude_table,target=args.target,targetMJD=args.targetMJD,target_timerange=args.target_timerange,target_decrange=args.target_decrange)
    return


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


def speccal(args):
    # find brightest continuum sources at the current declination # dec 16 continuum sources
    reftime = Time.now()#Time(args.UTCday + "T12:00:00",format='isot')
    print("Reference time:",reftime.isot)
    allnvsscoords,allnvssfluxes,allnvssms = read_nvss()
    if args.search_dec == 180:
        elev = get_best_elev(reftime)#get_elevation(reftime)
        search_dec = get_declination(elev).value
    else:
        search_dec = args.search_dec
    #read sources to exclude
    exclude_table = table_dir + "/NSFRB_excludecal.json"
    if len(exclude_table) > 0:
        f = open(exclude_table,"r")
        ex_table = json.load(f)['NVSS_exclude']
        f.close()
    else:
        ex_table =  []
    print("sources to exclude:",ex_table)

    decrange= 0.5
    vis_nvsscoords = allnvsscoords[np.abs(allnvsscoords.dec.value-search_dec) <decrange]
    vis_nvssfluxes = allnvssfluxes[np.abs(allnvsscoords.dec.value-search_dec) <decrange]
    vis_nvssms = allnvssms[np.abs(allnvsscoords.dec.value-search_dec) <decrange]

    #single nvss source
    if len(args.nvss)>0:
        idxs = []
        idxnames = []
        for i in range(len(args.nvss)):
            print(args.nvss[i])
            print(SkyCoord(args.nvss[i],unit=(u.hourangle,u.deg),frame='icrs').separation(vis_nvsscoords))
            print(vis_nvsscoords)
            idx = np.argmin(SkyCoord(args.nvss[i],unit=(u.hourangle,u.deg),frame='icrs').separation(vis_nvsscoords).value)
            if SkyCoord(args.nvss[i],unit=(u.hourangle,u.deg),frame='icrs').separation(vis_nvsscoords).to(u.arcsecond).value[idx] < 60:
                idxs.append(np.argmin(SkyCoord(args.nvss[i],unit=(u.hourangle,u.deg),frame='icrs').separation(vis_nvsscoords).value))
                idxnames.append(args.nvss[i])
            else:
                print(args.nvss[i],"not found")
        idxs = np.array(idxs)
        bright_nvsscoords = vis_nvsscoords[idxs]
        bright_nvssfluxes = vis_nvssfluxes[idxs]
        bright_nvssms = vis_nvssms[idxs]
        print("Running astrometric calibration pipeline with NVSS sources ",idxnames)
    else:
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
        timeax = Time(int(reftime.mjd) - np.linspace(0,24,24)[::-1]/24,format='mjd')
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
    full_img = np.zeros((image_size,image_size,int(gulpsize/args.timebin),16*nchan_per_node),dtype=float)
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
        gulps = np.arange(bright_offsets[bright_idx]//gulpsize,min([(bright_offsets[bright_idx]//gulpsize)+args.ngulps,90]),dtype=int)
        if len(gulps)<args.ngulps:
            gulps = np.concatenate([np.arange(max([0,bright_offsets[bright_idx]//gulpsize - (args.ngulps-len(gulps))]),bright_offsets[bright_idx]//gulpsize),gulps])
            gulps = np.unique(gulps)
        ngulps = len(gulps)
        if ngulps>1:
            #get RA cutoffs for each gulp
            racutoffs = []
            for g in range(ngulps):
                racutoffs.append(get_RA_cutoff(bright_nvsscoords[bright_idx].dec.value,usefit=True,asint=True,offset_s=g*tsamp_ms*gulpsize/1000))
            print("RA CUTOFFS:",racutoffs)
        else:
            racutoffs = [0]

        print(bright_idx,fnum,gulps)
        g=0
        buff = args.buff
        min_gridsize = image_size
        copydir = vis_dir + bright_nvssnames[bright_idx].replace(" ","") + "/"
        sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
        corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]

        if args.uselastimage and ((len(glob.glob(copydir + "nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out")) == 0) or (len(glob.glob(vis_dir + "/lxd110" + corrs[0] + "/nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out"))>0)):
            if len(glob.glob(copydir + "nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out")) == 0:
                sb,mjd,dec = pipeline.read_raw_vis(vis_dir + "/lxd110" + corrs[0] + "/nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=0,headersize=16,get_header=True)
            else:
                print(copydir + "nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out")
                sb,mjd,dec = pipeline.read_raw_vis(copydir + "nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=0,headersize=16,get_header=True)

            if glob.glob(copydir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_" + str("outriggers_" if outriggers else "") + "image.npy"):
                print("Using image at " + copydir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_" + str("outriggers_" if outriggers else "") + "image.npy")
                full_img = np.load(copydir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_" + str("outriggers_" if outriggers else "") + "image.npy")

                test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
                pt_dec = dec*np.pi/180.
                bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
                #ff = 1.53-np.arange(8192)*0.25/8192
                #fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
                fobs = (1e-3)*(np.reshape(freq_axis_fullres,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1))


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

                tmp, bname, blen, UVW, antenna_order = flag_vis(np.zeros((25,len(bname),nchans_per_node,2)), bname, blen, UVW, antenna_order, (list(bad_antennas) + list(args.flagants) if outriggers else list(flagged_antennas) + list(args.flagants)), bmin, list(flagged_corrs) + list(args.flagcorrs), flag_channel_templates = fcts)
                U = UVW[0,:,1]
                V = UVW[0,:,0]
                W = UVW[0,:,2]

                uv_diag=np.max(np.sqrt(U**2 + V**2))
                pixel_resolution = (lambdaref/uv_diag/pixperFWHM)
                
                ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + (gulps[0]*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)
                image_size,min_gridsize = full_img.shape[:2]
                ra_grid_2D = ra_grid_2D[:,-min_gridsize:]
                dec_grid_2D = dec_grid_2D[:,-min_gridsize:]
            
        else:

            full_img = np.zeros((image_size,image_size,int(gulpsize*ngulps/args.timebin),16*nchan_per_node),dtype=float) #,gulpsize,16*nchan_per_node))
            print("Image shape:",full_img.shape)
            for gulp in gulps:#[77,78,79,80,81]:##0,45,75]:#range(3):





                dat = None
                sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
                corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
                os.system("mkdir " + copydir)
                for i in range(16):
                    try:
                        if len(glob.glob(copydir + "nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out")) == 0:
                            os.system("cp " + vis_dir + "/lxd110" + corrs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out " + copydir)
                            dat_i,sb,mjd,dec = pipeline.read_raw_vis(vis_dir + "/lxd110" + corrs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=gulp,headersize=16)
                        else:
                            print(copydir + "nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out")
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

                #FLUX CAL
        
                print("Getting UVW params...")
                test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
                pt_dec = dec*np.pi/180.
                bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
                #ff = 1.53-np.arange(8192)*0.25/8192
                #fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
                fobs = (1e-3)*(np.reshape(freq_axis_fullres,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1))


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
                U = UVW[0,:,1]
                V = UVW[0,:,0]
                W = UVW[0,:,2]

                uv_diag=np.max(np.sqrt(U**2 + V**2))
                pixel_resolution = (lambdaref/uv_diag/pixperFWHM)
                dat[np.isnan(dat)] = 0
                if g == 0:
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

                for i in range(int(dat.shape[0]/args.timebin)):
                    for j in range(len(corrs)):
                        for k in range(dat.shape[-1]):
                            for jj in range(nchans_per_node):
                                if ngulps>1:
                                    full_img[:,:,int(g*gulpsize/args.timebin) + i,(j*nchans_per_node) + jj] += revised_robust_image(dat[i*args.timebin:(i+1)*args.timebin,:,(j*nchans_per_node) + jj,k],
                                                   U/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   V/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   image_size,robust=args.robust,
                                                   pixel_resolution=pixel_resolution)
                                                     
                                else:
                                    full_img[:,:,int(g*gulpsize/args.timebin) + i,(j*nchans_per_node) + jj] += revised_robust_image(dat[i*args.timebin:(i+1)*args.timebin,:,(j*nchans_per_node) + jj,k],
                                                   U/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   V/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   image_size,robust=args.robust,
                                                   pixel_resolution=pixel_resolution)
    
                g += 1

            if dat is None or int(dec) != int(search_dec):
                continue
        
        
            np.save(copydir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_" + str("outriggers_" if outriggers else "") + "image.npy",full_img)

        
        #find the expected source position based on astrometric cal offsets and errors...somehow...then get brightest pixel
        closepix = np.unravel_index(np.argmin(bright_nvsscoords[bright_idx].separation(SkyCoord(ra_grid_2D*u.deg,
                                                                                           dec_grid_2D*u.deg,frame='icrs'))),ra_grid_2D.shape)
        bbox = (max([closepix[0]-buff,0]),
                    min([closepix[0]+buff+1,image_size]),
                    max([closepix[1]-buff,0]),
                    min([closepix[1]+buff+1,min_gridsize]))
        if ngulps>1:
            all_full_imgs = [full_img[bbox[0]:bbox[1],bbox[2]:bbox[3],g*int(gulpsize/args.timebin):(g+1)*int(gulpsize/args.timebin),:] for g in range(ngulps)]
            input_imgs,input_ra_grid_2D,input_dec_grid_2D,min_gridsize = stack_images(all_full_imgs,racutoffs,ra_grid_2D[bbox[0]:bbox[1],bbox[2]:bbox[3]],dec_grid_2D[bbox[0]:bbox[1],bbox[2]:bbox[3]])
            stacked_img = np.concatenate(input_imgs,axis=2)
            print("aligned images, new shape:",stacked_img.shape)
            input_img = np.nanmean(stacked_img,(2,3))
            peakpix = np.unravel_index(np.argmax(input_img),input_img.shape)
            bright_pixcoord = SkyCoord(input_ra_grid_2D[peakpix[0],peakpix[1]]*u.deg,input_dec_grid_2D[peakpix[0],peakpix[1]]*u.deg,frame='icrs')
        else:
            input_img = np.nanmean(full_img[bbox[0]:bbox[1],bbox[2]:bbox[3],:,:],(2,3))
            peakpix = np.unravel_index(np.argmax(input_img),(bbox[1]-bbox[0],bbox[3]-bbox[2]))
            bright_pix = (peakpix[0] + bbox[0] ,peakpix[1] + bbox[2])
            bright_pixcoord = SkyCoord(ra_grid_2D[bright_pix[0],bright_pix[1]]*u.deg,dec_grid_2D[bright_pix[0],bright_pix[1]]*u.deg,frame='icrs')
        bright_pixel = np.unravel_index(np.argmin(bright_pixcoord.separation(SkyCoord(ra=ra_grid_2D*u.deg,dec=dec_grid_2D*u.deg,frame='icrs')).value),ra_grid_2D.shape)

        #plotting
        plt.figure(figsize=(12,12))
        fullmean = True
        median_sub = False
        vmin=args.vmin#None#-np.nanmax(full_img.mean((2,3)))/4#-1
        vmax=args.vmax#None#np.nanmax(full_img.mean((2,3)))/4#1 #0.2#1

        plt.title("SOURCE: " + bright_nvssnames[bright_idx] + "\nMJD: " + str(mjd) + "\nFNUM: " + bright_fnames[bright_idx],fontsize=20)
        if ngulps>1:
            plt.scatter(input_ra_grid_2D.flatten(),input_dec_grid_2D.flatten(),c=input_img.flatten(),alpha=0.5,cmap='cool',marker='s',s=100,vmin=vmin,vmax=vmax)
        else:
            plt.scatter(ra_grid_2D.flatten(),dec_grid_2D.flatten(),c=np.nanmean(full_img,(2,3)).flatten(),alpha=0.5,cmap='cool',marker='s',s=100,vmin=vmin,vmax=vmax)#0.8*np.nanmax((full_img.mean((2,3)))))
        plt.scatter(bright_nvsscoords[bright_idx].ra.to(u.deg).value,bright_nvsscoords[bright_idx].dec.to(u.deg).value,marker='o',s=1000,edgecolors='red',linewidth=4,facecolors="none",alpha=0.8)
        plt.xlim(bright_nvsscoords[bright_idx].ra.to(u.deg).value-(0.3 if outriggers else 1.5),bright_nvsscoords[bright_idx].ra.to(u.deg).value+(0.3 if outriggers else 1.5))
        plt.ylim(bright_nvsscoords[bright_idx].dec.to(u.deg).value-(0.3 if outriggers else 1.5),bright_nvsscoords[bright_idx].dec.to(u.deg).value+(0.3 if outriggers else 1.5))
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
        plt.gca().invert_xaxis()
        plt.savefig(img_dir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_"+ str("outriggers_" if outriggers else "")+"speccal_refimage.png")
        plt.close()
            
        #get dynamic spectrum
        if ngulps > 1:
            if args.boxmean:
                bright_dynspec = stacked_img.mean((0,1))
            else:
                bright_dynspec = stacked_img[peakpix[0],peakpix[1],:,:]
        else:
            if args.boxmean:
                bright_dynspec = full_img[bbox[0]:bbox[1],bbox[2]:bbox[3],:,:].mean((0,1))
            else:
                bright_dynspec = full_img[bright_pixel[0],bright_pixel[1],:,:]
        bright_measfluxs.append(np.nanmean(bright_dynspec))
        bright_measfluxerrs.append(np.nanstd(np.nanmean(bright_dynspec,1))/np.sqrt(bright_dynspec.shape[0]))
        if bright_measfluxs[-1]<0:
            bright_measfluxs[-1] = -1
            bright_measfluxerrs[-1] = -1
            continue
        print("Beamformed flux arb. units:",bright_measfluxs[-1]," +- ",bright_measfluxerrs[-1])
        print("-"*20)
        print("")
        
        #plotting
        print(bright_dynspec.shape)
        print(bright_dynspec)
        plt.figure(figsize=(16,12))
        plt.subplot(2,2,1,facecolor='black')
        plt.step(np.arange(int(gulpsize*ngulps/args.timebin))*tsamp_ms*args.timebin/1000,np.nanmean(bright_dynspec,1),linewidth=4)
        plt.xlim(0,tsamp_ms*gulpsize*ngulps/1000)

        plt.subplot(2,2,4,facecolor='black')
        plt.step(np.nansum(bright_dynspec,0),fobs,linewidth=1)
        cm = plt.get_cmap('Blues')
        for i in range(int(gulpsize*ngulps/args.timebin)):
            plt.step(np.nansum(bright_dynspec[:i+1,:],0),fobs,color=cm(i/int(gulpsize*ngulps/args.timebin)),alpha=0.5,linewidth=1)
        plt.ylim(fobs[-1],fobs[0])
        plt.scatter([],[],c=[],vmin=0,vmax=tsamp_ms*gulpsize/1000,cmap='Blues')
        plt.title("SOURCE: " + bright_nvssnames[bright_idx] + "\nMJD: " + str(mjd) + "\nFNUM: " + bright_fnames[bright_idx],fontsize=20)
        plt.colorbar(label='Time (s)')

        plt.subplot(2,2,3)
        plt.imshow(bright_dynspec.transpose(),aspect='auto',extent=(0,tsamp_ms*gulpsize*ngulps/1000,fobs[-1],fobs[0]))
        plt.xlim(0,tsamp_ms*gulpsize*ngulps/1000)
        plt.ylim(fobs[-1],fobs[0])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.subplots_adjust(hspace=0,wspace=0)
        plt.suptitle("real",fontsize=25)
        plt.savefig(img_dir+bright_nvssnames[bright_idx].replace(" ","")+"_" +str(Time(mjd,format='mjd').isot) + "_" + str(fnum) + "_"+ str("outriggers_" if outriggers else "")+ "image_speccal.png")

        plt.close()


    update_speccal_table(bright_nvssnames,bright_nvsscoords,bright_fnames,bright_measfluxs,bright_measfluxerrs,bright_nvssfluxes,outriggers,init=args.init_speccal,fitresid_th=args.specresid_th,exclude_table=exclude_table,image_flux=True,target=args.target,targetMJD=args.targetMJD,target_timerange=args.target_timerange,target_decrange=args.target_decrange)
    return




def main(args):
    if (not args.astrocal_only and not args.speccal_only) or (args.astrocal_only and not args.speccal_only):
        if args.update_only:
            update_astrocal_table([],[],[],[],[],[],[],[],[],[],[],args.outriggers,init=args.init_astrocal,resid_th=args.astroresid_th,exclude_table=table_dir + "/NSFRB_excludecal.json",target=args.target,targetMJD=args.targetMJD,target_timerange=args.target_timerange,target_decrange=args.target_decrange)
        else:
            astrocal(args)
    if (not args.astrocal_only and not args.speccal_only) or (not args.astrocal_only and args.speccal_only):
        if args.update_only:
            update_speccal_table([],[],[],[],[],[],args.outriggers,init=args.init_speccal,fitresid_th=args.specresid_th,exclude_table=table_dir + "/NSFRB_excludecal.json",image_flux=True,target=args.target,targetMJD=args.targetMJD,target_timerange=args.target_timerange,target_decrange=args.target_decrange)
        else:
            speccal(args)
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('--astrocal_only',action='store_true',help='Run astrometric cal only')
    parser.add_argument('--speccal_only',action='store_true',help='Run flux cal only')
    parser.add_argument('--init_astrocal',action='store_true',help='Initialize json astrometry table')
    parser.add_argument('--init_speccal',action='store_true',help='Initialize json flux cal table')
    #parser.add_argument('--UTCday',type=str,help='UTC day to run fluxcal with in ISO format (e.g. 2024-06-12); if not given, uses the previous day',default=Time(Time.now().mjd,format='mjd').isot[:10])
    parser.add_argument('--search_dec',type=float,help='If given, searches for source observations at this dec; otherwise uses median dec from past day',default=180.0)
    parser.add_argument('--numsources',type=int,help='Maximum number of sources to use for fluxcal, takes the brightest within 0.5 degrees of the current dec, default=10',default=10)
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=8)
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
    parser.add_argument('--astroresid_th',type=float,help='Maximum allowed normalized residual to include in astrometric or flux cal; default=0.2',default=0.2)
    parser.add_argument('--specresid_th',type=float,help='Maximum allowed residual from linear speccal fit,default=0.1',default=0.1)
    parser.add_argument('--uselastimage',action='store_true',help='Use previously saved .npy file if available')
    #parser.add_argument('--image_flux',action='store_true',help='Derive flux from image instead of beamformed flux')
    parser.add_argument('--nvss',nargs='+',type=str,help='J-name of specific NVSS source to calibrate on; e.g. \'J182210+160015\'',default=[])
    parser.add_argument('--rfc',nargs='+',type=str,help='J-name of specific RFC source to calibrate on; e.g. \'J182210+160015\'',default=[])
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
    args = parser.parse_args()
    main(args)
