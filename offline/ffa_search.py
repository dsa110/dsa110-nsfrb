import time
from concurrent.futures import ThreadPoolExecutor,wait
from nsfrb.imaging import stack_images,single_pix_image
from scipy.stats import norm
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
from nsfrb.planning import get_RA_cutoff,atnf_cat
from matplotlib.patches import Ellipse
from nsfrb.config import *
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from scipy.fftpack import ifftshift, ifft2,fftshift,fft2,fftfreq
from nsfrb.config import IMAGE_SIZE,UVMAX,chanbw,lambdaref,freq_axis,freq_axis_fullres,telescope_diameter
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


def image_main(args):
    print("Starting incoherent search of files "+str(args.fnum))
    mingulp=args.mingulp
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
    outriggers = args.outriggers
    nchans_per_node = nchan_per_node = args.nchans_per_node
    ngulps=args.ngulps
    gulpsize=args.gulpsize
    image_size = gridsize = args.image_size
    fnum = args.fnum
    sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
    corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
    fobs = np.reshape(freq_axis_fullres,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1)/1000
    print(fobs)
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

    if len(args.testpulsar)==3:
        pass
    else:
        try:
            assert(args.usecache)
            image_cube = np.load(args.path + "/"+str(args.fnum) + "_periodicity_imagecube.npy")

            sb,mjd,dec = pipeline.read_raw_vis(datadirs[0] + "/nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=mingulp,headersize=16,get_header=True)
            bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
            dat_i, bname_, blen_, UVW_, antenna_order_ = flag_vis(np.zeros((25,4656,8,2)), bname, blen, UVW, antenna_order,
                                                                                list(bad_antennas if outriggers else flagged_antennas) + list(args.flagants),
                                                                                                                            bmin=args.bmin,flagged_corrs=list(flagged_corrs)+list(args.flagcorrs),
                                                                                                                                                                        flagged_chans=list(args.flagchans))
            U = UVW_[0,:,1]
            V = UVW_[0,:,0]

            uv_diag=np.max(np.sqrt(U**2 + V**2))
            ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + ((mingulp)*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)
            print("Loading image cube from " + args.path + "/"+str(args.fnum) + "_periodicity_imagecube.npy")
        except Exception as exc:
            print(exc)
            image_cube_list = []
            racutoffs = []
            for j in range(ngulps):
                print("Reading gulp ",j,"...")
                for i in range(16):
                    print(">sb",i,"...",end="")
                    print(datadirs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out")
                    if i ==0 :
                        tmp_dat,sb,mjd,dec = pipeline.read_raw_vis(datadirs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=mingulp+j,headersize=16)
                    else:
                        tmp_dat = np.concatenate([tmp_dat,pipeline.read_raw_vis(datadirs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=mingulp+j,headersize=16)[0]],2)

                if j ==0:
                    bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
                    uv_diag=np.max(np.sqrt(UVW[0,:,1]**2 + UVW[0,:,0]**2))
                    pixel_resolution = (lambdaref/uv_diag/3)
                dat_i, bname_, blen_, UVW_, antenna_order_ = flag_vis(tmp_dat, bname, blen, UVW, antenna_order,
                                            list(bad_antennas if outriggers else flagged_antennas) + list(args.flagants),
                                            bmin=args.bmin,flagged_corrs=list(flagged_corrs)+list(args.flagcorrs),flag_channel_templates=fcts,
                                            flagged_chans=list(args.flagchans))
                U = UVW_[0,:,1]
                V = UVW_[0,:,0]
                tmpimg = np.zeros((image_size,image_size,gulpsize,len(corrs)))
                for ig in range(gulpsize):
                    print("imaging timestep ",ig,"...")
                    for jg in range(len(corrs)):
                        for jjg in range(nchans_per_node):
                            tmpimg[:,:,ig,jg] = np.nansum([tmpimg[:,:,ig,jg],
                                revised_robust_image(dat_i[ig:ig+1,:,(jg*nchans_per_node) + jjg,:].mean(2),
                                               U/(ct.C_GHZ_M/fobs[(jg*nchans_per_node) + jjg]),
                                               V/(ct.C_GHZ_M/fobs[(jg*nchans_per_node) + jjg]),
                                               image_size,robust=-2)],axis=0)
                                               #pixel_resolution=pixel_resolution)],axis=0)
                            #print(tmpimg[:,:,ig,jg])
                racutoffs.append(0 if j==0 else get_RA_cutoff(dec,T=T*j,usefit=True))
                medimg = np.nanmedian(tmpimg,2,keepdims=True)
                stdimg = np.nanstd(tmpimg,2,keepdims=True)
                image_cube_list.append(np.nanmean((tmpimg-medimg)/stdimg,3))
                
                if j ==0:
                    ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + ((mingulp+j)*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)
            #stack images
            image_cube_list,ra_grid_2D,dec_grid_2D,min_gridsize = stack_images(image_cube_list,racutoffs,
                                                                         ra_grid_2D,dec_grid_2D)
            image_cube = np.concatenate(image_cube_list,2)
            np.save(args.path + "/"+str(args.fnum) + "_periodicity_imagecube.npy",image_cube)

    if args.imagegif:
        def update(ii):
            plt.clf()
            #plt.imshow(image_cube[:,:,ii],aspect='auto',interpolation='none',vmin=0,vmax=1)
            plt.scatter(ra_grid_2D.flatten(),dec_grid_2D.flatten(),image_cube[:,:,ii].flatten(),vmin=0,vmax=1)
            plt.xlabel("RA")
            plt.ylabel("Dec")
            plt.title(Time(mjd + ((mingulp+ii)*tsamp_ms*gulpsize/86400/1000),format='mjd').isot)
            return
        fig=plt.figure(figsize=(12,12))
        animation_fig = animation.FuncAnimation(fig,update,frames=image_cube.shape[2],interval=1)
        animation_fig.save(args.path + "/"+str(args.fnum) + "_periodicity_imagegif.gif")

    #fold on provided period
    if args.imageperiod>image_cube.shape[2]:
        print("No folding, not enough samples for provided period")
        return

    print("Folding to period ",args.imageperiod*tsamp_ms/1000,"s")
    fold_cube = np.nansum(image_cube[:,:,:args.imageperiod*(image_cube.shape[2]//args.imageperiod)].reshape((image_cube.shape[0],image_cube.shape[1],image_cube.shape[2]//args.imageperiod,args.imageperiod)),3)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_foldedcube.npy",fold_cube)
    return



    

def bf_search_main(args):

    print("Starting coherent search of files "+str(args.fnum))
    mingulp=args.mingulp
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
    outriggers = args.outriggers
    nchans_per_node = nchan_per_node = args.nchans_per_node
    ngulps=args.ngulps
    gulpsize=args.gulpsize
    image_size=gridsize=args.image_size
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

    #grid properties
    gridstep = args.gridstep
    nrows = args.gridrows
    ncols = args.gridcols
    nbeams = nrows*ncols
    gridorder=[]
    for i in range(-nrows//2,nrows-(nrows//2)):
        for j in range(-ncols//2,ncols-(ncols//2)):
            print(i+(0.5 if nrows%2==0 else 0),j + (0.5 if ncols%2==0 else 0))
            gridorder.append((i+(0.5 if nrows%2==0 else 0),j + (0.5 if ncols%2==0 else 0)))
    gridcoords = []
    allpix_all = []

    if len(args.testpulsar)==3:
        #dspec_incoherent = (norm.rvs(loc=0,scale=5,size=(ngulps*25,16*nchans_per_node)))
        dspec_beams = (norm.rvs(loc=0,scale=5,size=(nbeams,ngulps*25,16*nchans_per_node)))
        testwidth_s,testP_s,testsnr =args.testpulsar
        testbeam = np.random.choice(np.arange(nbeams,dtype=int),size=1)
        npulse = int((ngulps*25*tsamp_ms/1000)/testP_s)
        print(npulse)
        for n in range(npulse):
            print(n*testP_s)
            pulse = norm.pdf(np.arange(ngulps*25)*tsamp_ms/1000,loc=(testP_s/2) + n*testP_s,scale=testwidth_s)
            pulse *= (testsnr)/np.nanmax(pulse)
            for j in range(16*nchans_per_node):
                dspec_beams[i,:,j] += pulse
        dspec_beams = np.abs(dspec_beams)
        mjd = Time.now().mjd
        dec = 71.6
    else:
        if args.usecache and len(glob.glob(args.path + "/"+str(args.fnum) + "_periodicity_beamform_dspecallbeams.npy"))>0:
            dspec_beams = np.load(args.path + "/"+str(args.fnum) + "_periodicity_beamform_dspecallbeams.npy")
            print(dspec_beams.shape)

            sb,mjd,dec = pipeline.read_raw_vis(datadirs[0] + "/nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=mingulp,headersize=16,get_header=True)
            print("Loading dynamic spectrum from " + args.path + "/"+str(args.fnum) + "_periodicity_dspec.npy")
            for ib in range(nbeams):
                racntr = get_ra(mjd + ((mingulp)*tsamp_ms*gulpsize/86400/1000),dec)
                gridcoords.append(SkyCoord(ra=(racntr + gridorder[ib][1]*gridstep)*u.deg,dec=(dec + gridorder[ib][0]*gridstep)*u.deg,frame='icrs'))
        else: #except Exception as exc:
            #print(exc)
            dspec_beams = np.zeros((nbeams,ngulps*gulpsize,nchans_per_node*16))
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
                    uv_diag=np.max(np.sqrt(UVW[0,:,1]**2 + UVW[0,:,0]**2))
                    pixel_resolution = (lambdaref / uv_diag) / pixperFWHM
                    ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + ((mingulp)*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag)
                    for ib in range(nbeams):
                        racntr = get_ra(mjd + ((mingulp)*tsamp_ms*gulpsize/86400/1000),dec)
                        gridcoords.append(SkyCoord(ra=(racntr + gridorder[ib][1]*gridstep)*u.deg,dec=(dec + gridorder[ib][0]*gridstep)*u.deg,frame='icrs'))
                dat_i, bname_, blen_, UVW_, antenna_order_ = flag_vis(tmp_dat, bname, blen, UVW, antenna_order,
                                            list(bad_antennas if outriggers else flagged_antennas) + list(args.flagants),
                                            bmin=args.bmin,flagged_corrs=list(flagged_corrs)+list(args.flagcorrs),flag_channel_templates=fcts,
                                            flagged_chans=list(args.flagchans))
                U = UVW_[0,:,1]
                V = UVW_[0,:,0]
                
                #beamform, DM=0
                for ib in range(nbeams):
                    for sb in range(16):

                        ret = single_pix_image(dat_i[:,:,sb*nchans_per_node:(sb+1)*nchans_per_node,:],U,V,fobs,sb,dec,mjd+((mingulp+j)*tsamp_ms*gulpsize/86400/1000),1,1,gridcoords[ib],tsamp_ms,pixel_resolution,pixperFWHM,uv_diag,nchans_per_node,gulpsize,image_size,[],0,-2)
                        dspec_beams[ib,j*gulpsize:(j+1)*gulpsize,sb*nchans_per_node:(sb+1)*nchans_per_node] =(ret[0]-np.nanmedian(ret[0],0,keepdims=True))/np.nanstd(ret[0],0,keepdims=True)
                        allpix_all.append(ret[1])
                    """
                    executor = ThreadPoolExecutor(16)
                    print("forming beam",ib)
                    batches = 8
                    for sbi in range(batches):
                        alltasks=[]
                        for sb in range((16//batches)*sbi,(16//batches)*(sbi+1)):
                            alltasks.append(executor.submit(single_pix_image,dat_i[:,:,sb*nchans_per_node:(sb+1)*nchans_per_node,:],U,V,fobs,sb,dec,mjd+((mingulp+j)*tsamp_ms*gulpsize/86400/1000),1,1,gridcoords[ib],tsamp_ms,pixel_resolution,pixperFWHM,uv_diag,nchans_per_node,gulpsize,image_size,[],0,-2))
                        wait(alltasks)
                        for sb in range((16//batches)*sbi,(16//batches)*(sbi+1)):
                            ret = alltasks[sb - (16//batches)*sbi].result()
                            #ret = single_pix_image(dat_i[:,:,sb*nchans_per_node:(sb+1)*nchans_per_node,:],U,V,fobs,sb,dec,mjd+((mingulp+j)*tsamp_ms*gulpsize/86400),1,1,gridcoords[ib],tsamp_ms,pixel_resolution,pixperFWHM,uv_diag,nchans_per_node=nchans_per_node,DM=0)
                            dspec_beams[ib,j*gulpsize:(j+1)*gulpsize,sb*nchans_per_node:(sb+1)*nchans_per_node] = (ret[0]-np.nanmedian(ret[0],0,keepdims=True))/np.nanstd(ret[0],0,keepdims=True)
                            allpix_all.append(ret[1])
                    executor.shutdown()
                    """
                print("done")
            np.save(args.path + "/"+str(args.fnum) + "_periodicity_beamform_dspecallbeams.npy",dspec_beams)

    #search each beam
    for ib in range(nbeams):
        print("searching beam ",ib)
        dspec_ib = dspec_beams[ib,:,:]
        tseries = np.nanmean(dspec_ib,1)
        tseries -= np.nanmedian(tseries)
        noiseest = np.nanstd(tseries)

        print("starting folding search...")
        allfoldoutput = np.zeros((len(args.widthtrials),len(args.periods)))
        for i in range(len(args.widthtrials)):
            w = args.widthtrials[i]
            boxcar=np.zeros_like(tseries)
            boxcar[:w] = 1/w
            tseries_w = np.convolve(tseries,boxcar,mode='same')
            allfoldoutput[i,:] = periodicity.ffa_slow(tseries_w,args.periods)
        if not args.ignoreSNR and not np.any(allfoldoutput>args.SNRthresh):
            print("No candidates found")
            continue #return
        bestidx = np.unravel_index(np.argmax(allfoldoutput),allfoldoutput.shape)
        bestW = args.widthtrials[bestidx[0]]
        print("Best width:",args.widthtrials[bestidx[0]])
        foldoutput = allfoldoutput[bestidx[0],:]
        print("Trial periods:",args.periods)
        print("Signal-to-noise:",foldoutput)
        print("Best period:",args.periods[np.argmax(foldoutput)])
        bestP = args.periods[np.argmax(foldoutput)]
        boxcar=np.zeros_like(tseries)
        boxcar[:bestW] = 1/bestW
        tseries_w = tseries#np.convolve(tseries,boxcar,mode='same')
        print(">>>",int((bestP*args.showprofiles)*(len(tseries_w)//(bestP*args.showprofiles))))
        pgram = tseries_w[:int((bestP*args.showprofiles)*(len(tseries_w)//(bestP*args.showprofiles)))].reshape((len(tseries_w)//(bestP*args.showprofiles),bestP*args.showprofiles))
        print(pgram.shape)

        print("starting timing search...")
        finePtrials = np.linspace(max([0,bestP-args.finePrange]),min([bestP+args.finePrange,len(tseries)//2]),args.nfinePtrials)
        resid = periodicity.ffa_timing(tseries,finePtrials)
        print("Trial periods:",finePtrials)
        print(np.min(resid,1))
        fineP = finePtrials[np.argmin(np.min(resid,1))]

        print("done")

        #get nearby pulsars
        print("Looking for nearby pulsars within",1.22*lambdaref*(180/np.pi)/telescope_diameter/2,"deg...")
        psr_coords,psr_names,psr_ps,psr_dms,psr_ws,psr_fs = atnf_cat(mjd,dec,sep=1.22*lambdaref*(180/np.pi)*u.deg/telescope_diameter/2)
        print("Done, found",len(psr_coords),"pulsars")
        plt.figure(figsize=(32,12))
        plt.subplot(2,4,1)
        plt.plot(np.arange(len(tseries)//bestW)*tsamp_ms*bestW/1000,np.nanmean(tseries[:bestW*(len(tseries)//bestW)].reshape((len(tseries)//bestW,bestW)),1))
        plt.xlim(0,len(tseries)*tsamp_ms/1000)

        plt.subplot(2,4,5)
        plt.imshow(np.nanmean(dspec_ib[:bestW*(len(tseries)//bestW)].reshape((dspec_ib.shape[0]//bestW,bestW,16 if args.plotcorrs else 16*nchans_per_node,nchans_per_node if args.plotcorrs else 1)),(1,3)).transpose(),aspect='auto',interpolation='none',vmin=0,vmax=np.sqrt(128)*noiseest*1.5,
                extent=(0,len(tseries)*tsamp_ms/1000,np.nanmin(freq_axis),np.nanmax(freq_axis)))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (MHz)")

        plt.subplot(2,4,2)
        plt.plot(np.linspace(0,args.showprofiles,pgram.shape[1]),np.nanmean(pgram,0))
        plt.xlim(0,args.showprofiles)
        for i in range(args.showprofiles):
            plt.axvline(i+1,color='red',linestyle='--')
        plt.subplot(2,4,6)
        plt.imshow(pgram,vmin=0,vmax=3*noiseest,aspect='auto',interpolation='none',
            extent=(0,args.showprofiles,0,bestP*args.showprofiles*(pgram.shape[0]+1)*tsamp_ms/1000),origin='lower')
        plt.xlabel("Phase (P={:.02f}".format(bestP*tsamp_ms/1000)+"s)")
        plt.ylabel("Time (s)")
        for i in range(args.showprofiles):
            plt.axvline(i+1,color='red',linestyle='--')

        plt.subplot(2,4,3)
        plt.plot(np.linspace(0,1,resid.shape[1]),resid[np.argmin(np.min(resid,1)),:])
        plt.xlim(0,1)
        plt.subplot(2,4,7)
        plt.imshow(resid,aspect='auto',
            extent=(0,1,finePtrials[0]*tsamp_ms/1000,finePtrials[-1]*tsamp_ms/1000))
        plt.axhline(fineP*tsamp_ms/1000,color='red')
        plt.text(0,fineP*tsamp_ms/1000 - 0.4,"P={:.02f}".format(fineP*tsamp_ms/1000)+"s",color='red',fontsize=25)
        plt.xlabel("Phase")
        plt.ylabel("Trial Period (s)")


        plt.subplot(2,4,4)
        plt.plot(np.array(args.periods)*tsamp_ms/1000,foldoutput)
        plt.axvline(bestP*tsamp_ms/1000,color='red',linestyle='--')
        plt.axvline(fineP*tsamp_ms/1000,color='red')
        plt.xlim(np.nanmin(args.periods)*tsamp_ms/1000,np.nanmax(args.periods)*tsamp_ms/1000)
        plt.ylabel("S/N")
        plt.subplot(2,4,8)
        plt.imshow(allfoldoutput,extent=(np.nanmin(args.periods)*tsamp_ms/1000,np.nanmax(args.periods)*tsamp_ms/1000,np.nanmin(args.widthtrials)*tsamp_ms/1000,np.nanmax(args.widthtrials)*tsamp_ms/1000),aspect='auto',interpolation='none',origin='lower')
        plt.ylabel("Trial Width (s)")
        plt.xlabel("Trial Period (s)")
        plt.axvline(bestP*tsamp_ms/1000,color='red',linestyle='--')
        plt.axvline(fineP*tsamp_ms/1000,color='red')
        plt.axhline(bestW*tsamp_ms/1000,color='red')
        plt.subplots_adjust(hspace=0)

        print(str(args.fnum)+", beam="+str(ib)+",W={:.02f}s".format(bestW*tsamp_ms/1000) + ",P={:.02f}s".format(fineP*tsamp_ms/1000) + "\nNearby Pulsars:\n" + "\n".join([str(psr_names[i]) + ", P={:.02f}".format(psr_ps[i]) + "s, W={:.02f}".format(psr_ws[i]) + "ms, DM={:.02f}".format(psr_dms[i]) + "pc/cc, S={:.02f}".format(psr_fs[i])+"mJy" for i in range(len(psr_coords))]))
        plt.suptitle(str(args.fnum)+", beam="+str(ib)+",W={:.02f}s".format(bestW*tsamp_ms/1000)+",P={:.02f}s".format(fineP*tsamp_ms/1000) + "\nNearby Pulsars:\n" + "\n".join([str(psr_names[i]) + ", P={:.02f}".format(psr_ps[i]) + "s, W={:.02f}".format(psr_ws[i]) + "ms, DM={:.02f}".format(psr_dms[i]) + "pc/cc, S={:.02f}".format(psr_fs[i])+"mJy" for i in range(len(psr_coords))]))


        plt.savefig(args.path + "/"+str(args.fnum) + "_periodicity_beamform_"+str(ib)+"_dspec"+str("_test" if args.testpulsar else "") + ".png")
        plt.close()

        np.save(args.path + "/"+str(args.fnum) + "_periodicity_beamform_"+str(ib)+"_dspec"+str("_test" if args.testpulsar else "") + ".npy",dspec_ib)
        np.save(args.path + "/"+str(args.fnum) + "_periodicity_beamform_"+str(ib)+"_tseries"+str("_test" if args.testpulsar else "") + ".npy",tseries)
        np.save(args.path + "/"+str(args.fnum) + "_periodicity_beamform_"+str(ib)+"_pgram_P"+str(bestP)+str("_test" if args.testpulsar else "") + ".npy",pgram)
        np.save(args.path + "/"+str(args.fnum) + "_periodicity_beamform_"+str(ib)+"_resid"+str("_test" if args.testpulsar else "") + ".npy",resid)
        np.save(args.path + "/"+str(args.fnum) + "_periodicity_beamform_"+str(ib)+"_fine_Ptrials"+str("_test" if args.testpulsar else "") + ".npy",finePtrials)
        results = dict()
        results["BestFFAPeriod_s"] = bestP*tsamp_ms/1000
        results["BestTIMEPeriod_s"] = fineP*tsamp_ms/1000
        results["RA_beam"] = gridcoords[ib].ra.value
        results["DEC_beam"] = gridcoords[ib].dec.value
        results["beamnumber"]= ib
        print(args.path + "/"+str(args.fnum) + "_periodicity_"+str(ib)+"_results"+str("_test" if args.testpulsar else "") + ".json")
        print(results)
        f=open(args.path + "/"+str(args.fnum) + "_periodicity_"+str(ib)+"_results"+str("_test" if args.testpulsar else "") + ".json","w")
        json.dump(results,f)
        f.close()
    return


def search_main(args):

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

    if len(args.testpulsar)==3:
        dspec_incoherent = (norm.rvs(loc=0,scale=5,size=(ngulps*25,16*nchans_per_node)))
        testwidth_s,testP_s,testsnr =args.testpulsar
        npulse = int((ngulps*25*tsamp_ms/1000)/testP_s)
        print(npulse)
        for n in range(npulse):
            print(n*testP_s)
            pulse = norm.pdf(np.arange(ngulps*25)*tsamp_ms/1000,loc=(testP_s/2) + n*testP_s,scale=testwidth_s)
            pulse *= (testsnr)/np.nanmax(pulse)
            for j in range(16*nchans_per_node):
                dspec_incoherent[:,j] += pulse
        dspec_incoherent = np.abs(dspec_incoherent)
        mjd = Time.now().mjd
        dec = 71.6
    else:
        try:
            assert(args.usecache)
            dspec_incoherent = np.load(args.path + "/"+str(args.fnum) + "_periodicity_dspec.npy")
            print(dspec_incoherent.shape)
            dspec_incoherent -= np.nanmedian(dspec_incoherent.reshape((dspec_incoherent.shape[0]//gulpsize,gulpsize,dspec_incoherent.shape[1])),1).repeat(gulpsize,0)
            dspec_incoherent /= np.nanstd(dspec_incoherent.reshape((dspec_incoherent.shape[0]//gulpsize,gulpsize,dspec_incoherent.shape[1])),1).repeat(gulpsize,0)

            sb,mjd,dec = pipeline.read_raw_vis(datadirs[0] + "/nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=mingulp,headersize=16,get_header=True)
            print("Loading dynamic spectrum from " + args.path + "/"+str(args.fnum) + "_periodicity_dspec.npy")
        except Exception as exc:
            print(exc)
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
                dspec_incoherent[j*gulpsize:(j+1)*gulpsize,:] /= np.nanstd(dspec_incoherent[j*gulpsize:(j+1)*gulpsize,:],0)
    
    tseries = np.nanmean(dspec_incoherent,1)
    tseries -= np.nanmedian(tseries)
    noiseest = np.nanstd(tseries)

    print("starting folding search...")
    allfoldoutput = np.zeros((len(args.widthtrials),len(args.periods)))
    for i in range(len(args.widthtrials)):
        w = args.widthtrials[i]
        boxcar=np.zeros_like(tseries)
        boxcar[:w] = 1/w
        tseries_w = np.convolve(tseries,boxcar,mode='same')
        allfoldoutput[i,:] = periodicity.ffa_slow(tseries_w,args.periods)
    if not args.ignoreSNR and not np.any(allfoldoutput>args.SNRthresh):
        print("No candidates found")
        return
    bestidx = np.unravel_index(np.argmax(allfoldoutput),allfoldoutput.shape)
    bestW = args.widthtrials[bestidx[0]]
    print("Best width:",args.widthtrials[bestidx[0]])
    foldoutput = allfoldoutput[bestidx[0],:]
    print("Trial periods:",args.periods)
    print("Signal-to-noise:",foldoutput)
    print("Best period:",args.periods[np.argmax(foldoutput)])
    bestP = args.periods[np.argmax(foldoutput)]
    boxcar=np.zeros_like(tseries)
    boxcar[:bestW] = 1/bestW
    tseries_w = tseries#np.convolve(tseries,boxcar,mode='same')
    print(">>>",int((bestP*args.showprofiles)*(len(tseries_w)//(bestP*args.showprofiles))))
    pgram = tseries_w[:int((bestP*args.showprofiles)*(len(tseries_w)//(bestP*args.showprofiles)))].reshape((len(tseries_w)//(bestP*args.showprofiles),bestP*args.showprofiles))
    print(pgram.shape)

    print("starting timing search...")
    finePtrials = np.linspace(max([0,bestP-args.finePrange]),min([bestP+args.finePrange,len(tseries)//2]),args.nfinePtrials)
    resid = periodicity.ffa_timing(tseries,finePtrials)
    print("Trial periods:",finePtrials)
    print(np.min(resid,1))
    fineP = finePtrials[np.argmin(np.min(resid,1))]

    print("done")

    #get nearby pulsars
    print("Looking for nearby pulsars within",1.22*lambdaref*(180/np.pi)/telescope_diameter/2,"deg...")
    psr_coords,psr_names,psr_ps,psr_dms,psr_ws,psr_fs = atnf_cat(mjd,dec,sep=1.22*lambdaref*(180/np.pi)*u.deg/telescope_diameter/2)
    print("Done, found",len(psr_coords),"pulsars")
    plt.figure(figsize=(32,12))
    plt.subplot(2,4,1)
    plt.plot(np.arange(len(tseries)//bestW)*tsamp_ms*bestW/1000,np.nanmean(tseries[:bestW*(len(tseries)//bestW)].reshape((len(tseries)//bestW,bestW)),1))
    plt.xlim(0,len(tseries)*tsamp_ms/1000)
    
    plt.subplot(2,4,5)
    plt.imshow(np.nanmean(dspec_incoherent[:bestW*(len(tseries)//bestW)].reshape((dspec_incoherent.shape[0]//bestW,bestW,16 if args.plotcorrs else 16*nchans_per_node,nchans_per_node if args.plotcorrs else 1)),(1,3)).transpose(),aspect='auto',interpolation='none',vmin=0,vmax=np.sqrt(128)*noiseest*1.5,
            extent=(0,len(tseries)*tsamp_ms/1000,np.nanmin(freq_axis),np.nanmax(freq_axis)))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    
    plt.subplot(2,4,2)
    plt.plot(np.linspace(0,args.showprofiles,pgram.shape[1]),np.nanmean(pgram,0))
    plt.xlim(0,args.showprofiles)
    for i in range(args.showprofiles):
        plt.axvline(i+1,color='red',linestyle='--')
    plt.subplot(2,4,6)
    plt.imshow(pgram,vmin=0,vmax=3*noiseest,aspect='auto',interpolation='none',
            extent=(0,args.showprofiles,0,bestP*args.showprofiles*(pgram.shape[0]+1)*tsamp_ms/1000),origin='lower')
    plt.xlabel("Phase (P={:.02f}".format(bestP*tsamp_ms/1000)+"s)")
    plt.ylabel("Time (s)")
    for i in range(args.showprofiles):
        plt.axvline(i+1,color='red',linestyle='--')

    plt.subplot(2,4,3)
    plt.plot(np.linspace(0,1,resid.shape[1]),resid[np.argmin(np.min(resid,1)),:])
    plt.xlim(0,1)
    plt.subplot(2,4,7)
    plt.imshow(resid,aspect='auto',
            extent=(0,1,finePtrials[0]*tsamp_ms/1000,finePtrials[-1]*tsamp_ms/1000))
    plt.axhline(fineP*tsamp_ms/1000,color='red')
    plt.text(0,fineP*tsamp_ms/1000 - 0.4,"P={:.02f}".format(fineP*tsamp_ms/1000)+"s",color='red',fontsize=25)
    plt.xlabel("Phase")
    plt.ylabel("Trial Period (s)")


    plt.subplot(2,4,4)
    plt.plot(np.array(args.periods)*tsamp_ms/1000,foldoutput)
    plt.axvline(bestP*tsamp_ms/1000,color='red',linestyle='--')
    plt.axvline(fineP*tsamp_ms/1000,color='red')
    plt.xlim(np.nanmin(args.periods)*tsamp_ms/1000,np.nanmax(args.periods)*tsamp_ms/1000)
    plt.ylabel("S/N")
    plt.subplot(2,4,8)
    plt.imshow(allfoldoutput,extent=(np.nanmin(args.periods)*tsamp_ms/1000,np.nanmax(args.periods)*tsamp_ms/1000,np.nanmin(args.widthtrials)*tsamp_ms/1000,np.nanmax(args.widthtrials)*tsamp_ms/1000),aspect='auto',interpolation='none',origin='lower')
    plt.ylabel("Trial Width (s)")
    plt.xlabel("Trial Period (s)")
    plt.axvline(bestP*tsamp_ms/1000,color='red',linestyle='--')
    plt.axvline(fineP*tsamp_ms/1000,color='red')
    plt.axhline(bestW*tsamp_ms/1000,color='red')
    plt.subplots_adjust(hspace=0)
    
    print(str(args.fnum)+",W={:.02f}s".format(bestW*tsamp_ms/1000) + ",P={:.02f}s".format(fineP*tsamp_ms/1000) + "\nNearby Pulsars:\n" + "\n".join([str(psr_names[i]) + ", P={:.02f}".format(psr_ps[i]) + "s, W={:.02f}".format(psr_ws[i]) + "ms, DM={:.02f}".format(psr_dms[i]) + "pc/cc, S={:.02f}".format(psr_fs[i])+"mJy" for i in range(len(psr_coords))]))
    plt.suptitle(str(args.fnum)+",W={:.02f}s".format(bestW*tsamp_ms/1000)+",P={:.02f}s".format(fineP*tsamp_ms/1000) + "\nNearby Pulsars:\n" + "\n".join([str(psr_names[i]) + ", P={:.02f}".format(psr_ps[i]) + "s, W={:.02f}".format(psr_ws[i]) + "ms, DM={:.02f}".format(psr_dms[i]) + "pc/cc, S={:.02f}".format(psr_fs[i])+"mJy" for i in range(len(psr_coords))]))
    
    
    plt.savefig(args.path + "/"+str(args.fnum) + "_periodicity"+str("_test" if args.testpulsar else "") + ".png")
    plt.close()
    
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_dspec"+str("_test" if args.testpulsar else "") + ".npy",dspec_incoherent)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_tseries"+str("_test" if args.testpulsar else "") + ".npy",tseries)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_pgram_P"+str(bestP)+str("_test" if args.testpulsar else "") + ".npy",pgram)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_resid"+str("_test" if args.testpulsar else "") + ".npy",resid)
    np.save(args.path + "/"+str(args.fnum) + "_periodicity_fine_Ptrials"+str("_test" if args.testpulsar else "") + ".npy",finePtrials)
    results = dict()
    results["BestFFAPeriod_s"] = bestP*tsamp_ms/1000
    results["BestTIMEPeriod_s"] = fineP*tsamp_ms/1000
    results["RA_point"] = get_ra(mjd,dec)
    results["DEC_point"] = dec
    print(args.path + "/"+str(args.fnum) + "_periodicity_results"+str("_test" if args.testpulsar else "") + ".json")
    print(results)
    f=open(args.path + "/"+str(args.fnum) + "_periodicity_results"+str("_test" if args.testpulsar else "") + ".json","w")
    json.dump(results,f)
    f.close()
    return

def main(args):
    if args.mode == 'incoherent_search':
        search_main(args)
    elif args.mode == 'beamform_search':
        bf_search_main(args)
        return
    elif args.mode=='image_fold' and args.imageperiod>0 and args.imagewidth>0:
        image_main(args)
    elif args.mode=='image_fold':
        print('Must provide --imageperiod and --imagewidth in \'image_fold\' mode')
        return 
    else:
        print("Invalide --mode")
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
    parser.add_argument('--widthtrials',nargs='+',type=int,help='width trials (in samples) to search',default=[])
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per node',default=8)
    parser.add_argument('--mingulp',type=int,help='First gulp to read',default=0)
    parser.add_argument('--ngulps',type=int,help='Number of gulps to read',default=90)
    parser.add_argument('--gulpsize',type=int,help='Number of time samples in each gulp',default=25)
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold',default=3)
    parser.add_argument('--showprofiles',type=int,help='Number of folded profiles to show',default=2)
    parser.add_argument('--ignoreSNR',action='store_true',help='ignore the SNR threshold and always save the periodogram')
    parser.add_argument('--usecache',action='store_true',help='use cached data')
    parser.add_argument('--plotbin',type=int,help='bin factor for plotting dynamic spectrum',default=1)
    parser.add_argument('--plotcorrs',action='store_true',help='bin dynamic spectrum by corr nodes')
    parser.add_argument('--testpulsar',type=float,nargs='+',help='width and period in seconds of test pulsar injection',default=[])
    parser.add_argument('--finePrange',type=float,help='range in samples around initial period to run timing analysis',default=50)
    parser.add_argument('--nfinePtrials',type=int,help='number of fine trials',default=50)
    parser.add_argument('--mode',type=str,choices=['incoherent_search','image_fold','beamform_search'],default='incoherent_search',help='incoherent_search: runs fast folding search pipeline\nimage_fold: forms and folds images at a specified period\nbeamform_search: runs fast folding search on grid of 25 beams')
    parser.add_argument('--imageperiod',type=int,help='period in samples to use in \'image\' mode',default=-1)
    parser.add_argument('--imagewidth',type=int,help='width in samples to use in \'image\' mode',default=-1)
    parser.add_argument('--image_size',type=int,help='image size, default=301',default=301)

    parser.add_argument('--imagegif',action='store_true',help='output gif')
    parser.add_argument('--gridrows',type=int,help='number of rows for beamforming search, default=5',default=5)
    parser.add_argument('--gridcols',type=int,help='number of cols for beamforming search, default=5',default=5)
    parser.add_argument('--gridstep',type=float,help='stepsize between beams in degrees, default=0.25',default=0.25)
    args = parser.parse_args()
    main(args)


