import argparse
from nsfrb import outputlogging
import time
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
plt.rcParams.update({'font.size': 30})
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
from nsfrb.config import IMAGE_SIZE,bmin,flagged_antennas,bad_antennas,pixperFWHM,NUM_CHANNELS,ngulps_per_file,nsamps
import json
from periodicity import ffa_slow,ffa_faster,gen_psamp_trials,ffa_timing

sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]

def main(args):
    t__ = time.time()
    tstamp = str(Time.now().isot if args.tstamp else "")
    #find files to search
    if len(args.GPplan)>0:
        files_ = glob.glob(vis_dir + args.GPplan + "/*sb00*.out")
       
        if len(files_)==0:
            print("No files found")
            return -1
        if not args.GPoverwrite:
            files =[]
            for f in files_:
                fn = int(os.path.basename(f)[11:-4]) 
                if len(glob.glob(vis_dir + args.GPplan + "/*"+str(fn)+"*snrs.npy"))==0:
                    files.append(f)
                else:
                    print("already searched",fn)
        else:
            files = files_

        decs = []
        mjds = []
        for f in files:
            sb,mjd,dec = pipeline.read_raw_vis(f,nchan=args.nchans_per_node,nsamps=nsamps,gulp=0,headersize=16,get_header=True)
            decs.append(dec)
            mjds.append(mjd)
        fnums = np.array([int(os.path.basename(f)[11:-4]) for f in files],dtype=int)
        print(fnums)
    elif args.fnum == -1:
        print("Need to provide either --fnum or --GPplan")
        return -1
    else:
        if len(args.imgfile)>0:
            fnums = np.array([int(os.path.basename(args.imgfile)[6:os.path.basename(args.imgfile).index("_period")])])
        else:
            fnums = np.array([args.fnum])
        
        if len(args.path)>0:
            files = glob.glob(args.path + "/nsfrb_sb00_" + str(fnums[0]) + ".out")
        else:
            files = glob.glob(vis_dir + "/lxd110" + corrs[0] + "/nsfrb_sb00_" + str(fnums[0]) + ".out")
        if len(files)==0:
            print("No files found")
            return -1
        sb,mjd,dec = pipeline.read_raw_vis(files[0],nchan=args.nchans_per_node,nsamps=nsamps,gulp=0,headersize=16,get_header=True)
        decs = [dec]
        mjds = [mjd]


    #figure out number of gulps
    gridsize = image_size = args.image_size
    outriggers = args.outriggers
    gulpsize = nsamps
    nchans_per_node = nchan_per_node = args.nchans_per_node
    gulps = np.arange(args.gulpoffset,min([args.gulpoffset+args.ngulps,ngulps_per_file]),dtype=int)
    if len(gulps)<args.ngulps:
        gulps = np.concatenate([np.arange(max([0,args.gulpoffset - (args.ngulps-len(gulps))]),args.gulpoffset),gulps])
        gulps = np.unique(gulps)
    ngulps = len(gulps)
    print(ngulps,gulps)

    for it in range(len(fnums)):
        fnum = fnums[it]
        dec = decs[it]
        mjd = mjds[it]


        if ngulps>1:
            #get RA cutoffs for each gulp
            racutoffs = []
            for g in range(ngulps):
                racutoffs.append(get_RA_cutoff(dec,usefit=True,asint=True,offset_s=g*tsamp_ms*gulpsize/1000,pixperFWHM=args.pixperFWHM))
                print("RA CUTOFFS:",racutoffs)
        else:
            racutoffs = [0]


        if len(args.imgfile)>0:
            print("loading image from "+args.imgfile)
            full_img = np.load(args.imgfile)
            image_size,min_gridsize,totsamps = full_img.shape
           
            test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
            pt_dec = dec*np.pi/180.
            bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
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

            dat, bname, blen, UVW, antenna_order = flag_vis(np.zeros((gulpsize,len(bname),nchans_per_node,2),dtype=complex), bname, blen, UVW, antenna_order, (list(bad_antennas) + list(args.flagants) if outriggers else list(flagged_antennas) + list(args.flagants)), bmin, list(flagged_corrs) + list(args.flagcorrs), flag_channel_templates = fcts)
            U = UVW[0,:,1]
            V = UVW[0,:,0]
            W = UVW[0,:,2]


            uv_diag=np.max(np.sqrt(U**2 + V**2))
            pixel_resolution = (lambdaref/uv_diag/args.pixperFWHM)
            ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + (gulps[0]*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag,pixperFWHM=args.pixperFWHM)
            ra_grid_2D = ra_grid_2D[:,-min_gridsize:]
            dec_grid_2D = dec_grid_2D[:,-min_gridsize:]
        else:
            print(fnum,gulps)
            g=0
            min_gridsize = image_size
            full_img = np.zeros((image_size,image_size,int(gulpsize*ngulps/args.timebin)),dtype=float) #,gulpsize,16*nchan_per_node))
            print("Image shape:",full_img.shape)
            
            
            print("Getting UVW params...")
            if len(args.path)==0:
                sb,mjd,dec = pipeline.read_raw_vis(vis_dir + "/lxd110" + corrs[0] + "/nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=0,headersize=16,get_header=True)
            else:
                sb,mjd,dec = pipeline.read_raw_vis(args.path + "nsfrb_sb" + sbs[0] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=0,headersize=16,get_header=True)
            test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
            pt_dec = dec*np.pi/180.
            bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
            fobs = (1e-3)*(np.reshape(freq_axis_fullres,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1))
            init_dsa = dict()
            init_dsa['test'] = test
            init_dsa['key_string'] = key_string
            init_dsa['nant'] = nant
            init_dsa['nchan'] = nchan
            init_dsa['npol'] = npol
            init_dsa['fobs'] = fobs
            init_dsa['samples_per_frame'] = samples_per_frame
            init_dsa['samples_per_frame_out'] = samples_per_frame_out
            init_dsa['nint'] = nint
            init_dsa['nfreq_int'] = nfreq_int
            init_dsa['antenna_order'] = antenna_order
            init_dsa['pt_dec'] = pt_dec
            init_dsa['tsamp'] = tsamp
            init_dsa['fringestop'] = fringestop
            init_dsa['filelength_minutes'] = filelength_minutes
            init_dsa['outrigger_delays'] = outrigger_delays
            init_dsa['refmjd'] = refmjd
            init_dsa['subband'] = subband
            init_dsa['bname'] = bname
            init_dsa['blen'] = blen
            init_dsa['UVW'] = UVW

            for gulp in gulps:#[77,78,79,80,81]:##0,45,75]:#range(3):

                dat = None
                for i in range(16):
                    try:
                        if len(args.path)==0:
                            dat_i,sb,mjd,dec = pipeline.read_raw_vis(vis_dir + "/lxd110" + corrs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=gulp,headersize=16)
                        else:
                            dat_i,sb,mjd,dec = pipeline.read_raw_vis(args.path + "nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan_per_node,nsamps=gulpsize,gulp=gulp,headersize=16)

                        print(mjd,dec,sb)

                        if dat is None:
                            dat = np.nan*np.ones(dat_i.shape,dtype=dat_i.dtype).repeat(len(corrs),axis=2)
                        dat[:,:,i*nchans_per_node:(i+1)*nchans_per_node,:] = dat_i

                    except Exception as exc:
                        print(exc)


                """
                print("Getting UVW params...")
                test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
                pt_dec = dec*np.pi/180.
                bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
                fobs = (1e-3)*(np.reshape(freq_axis_fullres,(len(corrs)*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1))
                """
                
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

                dat, bname, blen, UVW, antenna_order = flag_vis(dat, init_dsa['bname'], init_dsa['blen'], init_dsa['UVW'], init_dsa['antenna_order'], (list(bad_antennas) + list(args.flagants) if outriggers else list(flagged_antennas) + list(args.flagants)), bmin, list(flagged_corrs) + list(args.flagcorrs), flag_channel_templates = fcts)
                U = UVW[0,:,1]
                V = UVW[0,:,0]
                W = UVW[0,:,2]


                uv_diag=np.max(np.sqrt(U**2 + V**2))
                pixel_resolution = (lambdaref/uv_diag/args.pixperFWHM)
                dat[np.isnan(dat)] = 0
                if g == 0:
                    ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd + (gulps[0]*tsamp_ms*gulpsize/86400/1000),image_size,DEC=dec,two_dim=True,manual=False,uv_diag=uv_diag,pixperFWHM=args.pixperFWHM)

                for i in range(int(dat.shape[0]/args.timebin)):
                    for j in range(len(corrs)):
                        for k in range(dat.shape[-1]):
                            for jj in range(nchans_per_node):
                                if ngulps>1:
                                    full_img[:,:,int(g*gulpsize/args.timebin) + i] += revised_robust_image(dat[i*args.timebin:(i+1)*args.timebin,:,(j*nchans_per_node) + jj,k],
                                                   U/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   V/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   image_size,robust=args.robust,
                                                   pixel_resolution=pixel_resolution,
                                                   pixperFWHM=args.pixperFWHM)
                                else:
                                    full_img[:,:,int(g*gulpsize/args.timebin) + i] += revised_robust_image(dat[i*args.timebin:(i+1)*args.timebin,:,(j*nchans_per_node) + jj,k],
                                                   U/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   V/(ct.C_GHZ_M/fobs[(j*nchans_per_node) + jj]),
                                                   image_size,robust=args.robust,
                                                   pixel_resolution=pixel_resolution,
                                                   pixperFHWM=args.pixperFWHM)

                print(full_img[:,:,int(g*gulpsize/args.timebin):int((g+1)*gulpsize/args.timebin)].shape)
                if args.timebin < gulpsize:
                    full_img[:,:,int(g*gulpsize/args.timebin):int((g+1)*gulpsize/args.timebin)] -= np.nanmedian(full_img[:,:,int(g*gulpsize/args.timebin):int((g+1)*gulpsize/args.timebin)],axis=2,keepdims=True)
                g += 1
   
            full_img -= np.nanmedian(full_img,axis=2,keepdims=True)
            if ngulps>1:
                all_full_imgs = [full_img[:,:,g*int(gulpsize/args.timebin):(g+1)*int(gulpsize/args.timebin)] for g in range(ngulps)]
                stacked_full_imgs,ra_grid_2D,dec_grid_2D,min_gridsize = stack_images(all_full_imgs,racutoffs,ra_grid_2D,dec_grid_2D)
                full_img = np.zeros((gridsize,min_gridsize,full_img.shape[2]))
                for g in range(len(stacked_full_imgs)):
                    full_img[:,:,g*int(gulpsize/args.timebin):(g+1)*int(gulpsize/args.timebin)] = stacked_full_imgs[g]
                print("new img shape:",full_img.shape)
            np.save(str(args.path if len(args.path)>0 else img_dir) + "/nsfrb_"+str(fnum) + "_periodicity_search_data"+tstamp+".npy",full_img)
            
        #periodicity search
        print("Searching periods (s):",np.array(args.periods)*tsamp_ms*args.timebin/1000)
        noise_est = np.nanmedian(np.nanstd(full_img,axis=2))
        if noise_est == 0: noise_est = 1
        print("Noise estimate:",noise_est)
        #idxs_full,bool_idxs_full = gen_psamp_trials(args.periods,full_img.shape[2])
        #print(idxs_full.shape,bool_idxs_full.shape,full_img.shape)
        t_ = time.time()
        snrs = ffa_slow(full_img,args.periods)#snrs,tmp = ffa_faster(full_img,idxs_full.astype(int),bool_idxs_full.astype(bool),periodogram=False)
        print("slow search time:",time.time()-t_)
        #snrs /= noise_est
        np.save(str(args.path if len(args.path)>0 else img_dir) + "/nsfrb_"+str(fnum) + "_periodiciity_search_snrs"+tstamp+".npy",snrs)
        print("Peak S/N:",np.nanmax(snrs))
        if np.nanmax(snrs)>args.SNRthresh or args.alwaysplot:
            peakidx = np.unravel_index(np.nanargmax(snrs),snrs.shape)
            print("Candidate found at RA=",ra_grid_2D[peakidx[0],peakidx[1]],"DEC=",dec_grid_2D[peakidx[0],peakidx[1]],"P=",args.periods[peakidx[2]]*tsamp_ms*args.timebin/1000,"s")
            print(snrs[peakidx[0],peakidx[1],:])

            i = peakidx[2]
            ii = np.nanargmax(snrs[:,:,i])
            peak_timeseries = full_img.reshape((full_img.shape[0]*full_img.shape[1],full_img.shape[2]))[ii,:]
            minp = max([i-1,0])
            maxp = min([i+1,len(args.periods)-1])
            trial_p_fine = np.linspace(args.periods[minp],args.periods[maxp],args.nfine)
            #trial_p_fine = np.linspace((args.periods[i-1] if i>0 else args.periods[0]),(args.periods[i+1] if i<len(args.periods)-1 else len(args.periods)-1),args.nfine)
            print(trial_p_fine)
            t_ = time.time()
            resids = ffa_timing(peak_timeseries,trial_p_fine,args.periods[i])
            print("fine search time:",time.time()-t_)
            print(resids.shape)
            minresid = np.unravel_index(np.argmin(resids),resids.shape)
            print("Refined period P=",trial_p_fine[minresid[0]]*tsamp_ms*args.timebin/1000,"s")

            d = dict()
            d['ra'] = ra_grid_2D[peakidx[0],peakidx[1]]
            d['dec'] = dec_grid_2D[peakidx[0],peakidx[1]]
            d['P'] = trial_p_fine[minresid[0]]*tsamp_ms*args.timebin/1000
            f = open(str(args.path if len(args.path)>0 else img_dir) + "/nsfrb_"+str(fnum) + "_periodiciity_search_cand"+tstamp+".json","w")
            json.dump(d,f)
            f.close()

            if args.plot:
                vmax = np.nanpercentile(snrs,95)
                aspect = 0.25*(np.max(ra_grid_2D)-np.min(ra_grid_2D))/(np.max(dec_grid_2D)-np.min(dec_grid_2D))
                plt.figure(figsize=(36,12*(1 if args.candplot else (len(args.periods)-2)))) #*(len(args.periods))))

                #periodograms
                #tseries = np.nansum(full_img,(0,1))
                alphas = np.clip(full_img/(np.nanmax(full_img)/2-np.nanmin(full_img)),0.05,1).reshape((full_img.shape[0]*full_img.shape[1],full_img.shape[2]))
                msizes = (full_img/(np.nanmax(full_img)/3-np.nanmin(full_img))).reshape((full_img.shape[0]*full_img.shape[1],full_img.shape[2]))
                msizes[msizes<0]=0
                taxis = np.arange(full_img.shape[-1])

                if args.candplot:
                    plt.subplot(1,3,3)
                else:
                    plt.subplot(len(args.periods),3,3*i + 3)
                plt.imshow(resids,aspect='auto',extent=(0,1,np.nanmin(trial_p_fine)*(tsamp_ms*args.timebin/1000),np.nanmax(trial_p_fine)*(tsamp_ms*args.timebin/1000)),origin='lower')
                plt.axhline(trial_p_fine[minresid[0]]*(tsamp_ms*args.timebin/1000),color='red',linewidth=4)
                plt.axvline(minresid[1]/args.periods[i],color='red',linewidth=4)
                plt.title("Residuals (samples)")
                plt.colorbar()
                plt.xlabel("Phase")
                plt.ylabel("Trial Period (s)")
                if args.candplot:
                    plt.subplot(1,3,1)
                    plt.scatter((taxis%args.periods[i])/args.periods[i],(tsamp_ms*args.timebin*args.periods[i]/1000)*(taxis//args.periods[i]),alpha=alphas[ii,:],s=msizes[ii,:]*1000,label="P="+str(np.around(args.periods[i]*tsamp_ms*args.timebin/1000,2)) + " s")
                    if args.testP != 0:
                        testpoints = np.array([0,args.testP])/(tsamp_ms*args.timebin/1000) #args.testP*np.arange(int((full_img.shape[-1]*tsamp_ms*args.timebin/1000)/args.testP))/(tsamp_ms*args.timebin/1000)
                        for a in range(0,int(full_img.shape[-1]//args.periods[i]),4):
                            plt.plot(testpoints/args.periods[i],(tsamp_ms*args.timebin/1000)*(a*args.periods[i] + testpoints),color='red',alpha=0.5)
                    plt.ylabel("Time (s)")
                    plt.legend(loc='upper right')
                    plt.xlabel("Phase")

                    plt.subplot(1,3,2)
                    plt.scatter(ra_grid_2D.flatten(),dec_grid_2D.flatten(),c=snrs[:,:,i].flatten(),s=1*(100*(3/args.pixperFWHM)),alpha=1,vmin=0,vmax=vmax,marker='s',label="P="+str(np.around(args.periods[i]*tsamp_ms*args.timebin/1000,2)) + " s")
                    plt.legend(loc='upper right')
                    plt.ylabel("DEC")
                    #plt.suptitle("RA="+str(np.around(d['ra'],2)) + " deg,DEC="+ str(np.around(d['dec'],2)) + " deg,P="+str(np.around(args.periods[i]*tsamp_ms*args.timebin/1000,2)) + " s")
                    plt.xlabel("RA")
                else:
                    for i in range(len(args.periods)):
                        plt.subplot(len(args.periods),3,1+3*i)
                        plt.scatter((taxis%args.periods[i])/args.periods[i],(tsamp_ms*args.timebin*args.periods[i]/1000)*(taxis//args.periods[i]),alpha=alphas[ii,:],s=msizes[ii,:]*1000,label="P="+str(np.around(args.periods[i]*tsamp_ms*args.timebin/1000,2)) + " s")
                        if args.testP != 0:
                            testpoints = np.array([0,args.testP])/(tsamp_ms*args.timebin/1000) #args.testP*np.arange(int((full_img.shape[-1]*tsamp_ms*args.timebin/1000)/args.testP))/(tsamp_ms*args.timebin/1000)
                            for a in range(0,int(full_img.shape[-1]//args.periods[i]),4):
                                plt.plot(testpoints/args.periods[i],(tsamp_ms*args.timebin/1000)*(a*args.periods[i] + testpoints),color='red',alpha=0.5)
                        plt.ylabel("Time (s)")
                        plt.legend(loc='upper right')
                        plt.xlabel("Phase")

                        plt.subplot(len(args.periods),3,2+3*i)
                        plt.scatter(ra_grid_2D.flatten(),dec_grid_2D.flatten(),c=snrs[:,:,i].flatten(),s=1*(100*(3/args.pixperFWHM)),alpha=1,vmin=0,vmax=vmax,marker='s',label="P="+str(np.around(args.periods[i]*tsamp_ms*args.timebin/1000,2)) + " s")
                        plt.legend(loc='upper right')
                        plt.ylabel("DEC")
                        #plt.suptitle("RA="+str(np.around(d['ra'],2)) + " deg,DEC="+ str(np.around(d['dec'],2)) + " deg,P="+str(np.around(args.periods[i]*tsamp_ms*args.timebin/1000,2)) + " s")
                        plt.xlabel("RA")
                plt.suptitle("RA="+str(np.around(d['ra'],2)) + " deg,DEC="+ str(np.around(d['dec'],2)) + " deg,P="+str(np.around(trial_p_fine[minresid[0]]*tsamp_ms*args.timebin/1000,2)) + " s,S/N="+str(np.around(snrs[peakidx[0],peakidx[1],peakidx[2]],2)))
                plt.savefig(str(args.path if len(args.path)>0 else img_dir) + "/nsfrb_"+str(fnum)+"_periodicity_search" + tstamp + ".png")
                if args.show:
                    plt.show()
                else:
                    plt.close()
                if args.toslack:
                    #outputlogging.send_candidate_slack("nsfrb_"+str(fnum)+"_periodicity_search" + tstamp + ".png",filedir=str(args.path if len(args.path)>0 else img_dir) + "/")
                    print("sending notification via x11...")
                    os.system("cp " + str(args.path if len(args.path)>0 else img_dir) + "/" + "nsfrb_"+str(fnum)+"_periodicity_search" + tstamp + ".png" + " " + os.environ["NSFRBDIR"] + "/scripts/x11display.png")
                    os.system("echo 1 > "+ os.environ["NSFRBDIR"] + "/scripts/x11size.txt")
                    os.system("echo " + "nsfrb_"+str(fnum)+"_periodicity_search" + tstamp + ".png" + " > "+ os.environ["NSFRBDIR"] + "/scripts/x11alertmessage.txt")
                    print("done!")
    print("Full time:",time.time()-t__)
    return 0



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('--fnum',type=int,help='Number of fast visibility file to search',default=-1)
    parser.add_argument('--periods',nargs='+',type=int,help='periods (in samples) to search',default=[])
    parser.add_argument('--GPplan',type=str,help='Name of the GP plan so search all fast vis files for',default='')
    parser.add_argument('--path',type=str,help='Path of fast visibility data; if omitted, assumes data is in corr node directories',default='')
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
    parser.add_argument('--outriggers',action='store_true',help='Includes outrigger antennas in imaging')
    parser.add_argument('--ngulps',type=int,help='Number of gulps of 25 samples (3.25 s) to integrate, default=1',default=1)
    parser.add_argument('--gulpoffset',type=int,help='Gulp to start reading data, default=0',default=0)
    parser.add_argument('--timebin',type=int,help='Number of time samples to bin by, default=25 (each gulp binned into single sample)',default=25)
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging,default=-2 for uniform weighting',default=-2)
    parser.add_argument('--plot',action='store_true',help='Plot S/N vs position and trial period')
    parser.add_argument('--show',action='store_true',help='Show plot')
    parser.add_argument('--tstamp',action='store_true',help='Timestamp output files')
    parser.add_argument('--pixperFWHM',type=float,help='Pixels per FWHM, default 3',default=pixperFWHM)
    parser.add_argument('--imgfile',type=str,help='path to .npy full image file',default='')
    parser.add_argument('--SNRthresh',type=float,help='SNR Threshold',default=3)
    parser.add_argument('--testP',type=float,help='Trial period to overplot on periodogram; need not be an integer number of samples',default=0)
    parser.add_argument('--GPoverwrite',action='store_true',help='Checks for existing search files and overwrites if set, otherwise skips')
    parser.add_argument('--nfine',type=int,help='Number of fine period trials around the initial period,default=10',default=10)
    parser.add_argument('--candplot',action='store_true',help='Only plot periodogram for the candidate')
    parser.add_argument('--alwaysplot',action='store_true',help='Plot even if no candidates')
    parser.add_argument('--toslack',action='store_true',help='Send candidate plot to slack')
    args = parser.parse_args()
    main(args)
