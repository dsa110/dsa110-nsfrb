import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait
import glob
import csv
from matplotlib import pyplot as plt
from nsfrb.simulating import compute_uvw,get_core_coordinates,get_all_coordinates
#from inject import injecting
import h5py
from casatools import table
import numpy as np
from astropy.time import Time
import astropy.units as u
import sys
from dsamfs import utils as pu
from dsautils import cnf
from collections import OrderedDict
my_cnf = cnf.Conf(use_etcd=True)

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()

#sys.path.append(cwd+"/nsfrb/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/nsfrb/")
#sys.path.append(cwd+"/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,pixperFWHM
from nsfrb.imaging import inverse_uniform_image,uniform_image,inverse_revised_uniform_image,revised_uniform_image, uv_to_pix, revised_robust_image,get_ra,briggs_weighting,uniform_grid
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST
from nsfrb.TXclient import send_data,ipaddress
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images
from tqdm import tqdm
import time
from scipy.stats import norm,multivariate_normal
#import nsfrb.searching as sl
from nsfrb.outputlogging import numpy_to_fits
#from nsfrb import calibration as cal
from nsfrb import pipeline
import os
#vispath = os.environ["NSFRBDATA"] + "dsa110-nsfrb-fast-visibilities" #cwd + "-fast-visibilities"
#imgpath = cwd + "-images"
#inject_file = cwd + "-injections/injections.csv"

from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,flagged_antennas,Lon,Lat,maxrawsamps,flagged_corrs,timelogfile


"""
This script reads raw fast visibility data from a file on disk, applies fringe-stopping from a pre-made table,
applies calibration, and images. If specified, the resulting image is transmitted to the process server.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m

#flagged antennas/
TXtask_list = []
def offline_image_task(dat, U, V, gridsize,  pixel_resolution, nchans_per_node, fobs_j, j, briggs=False, robust= 0.0, return_complex=False, inject_img=None, inject_flat=False, wstack=False, W=None, Nlayers_w=18,pixperFWHM=pixperFWHM,
                        port=-1,ipaddress="",time_start_isot="", uv_diag=-1, Dec=-1, TXexecutor=None, stagger=0):

    outimage = np.nan*np.ones((args.gridsize,args.gridsize,args.num_time_samples,1)) 

    #for j in range(args.num_chans):

    for jj in range(nchans_per_node):
        chanidx = (nchans_per_node*j)+jj
        U_wav = U/(2.998e8/fobs_j[jj]/1e9)
        V_wav = V/(2.998e8/fobs_j[jj]/1e9)
        W_wav = None if not wstack else W/(2.998e8/fobs_j[jj]/1e9)
        uniform_grid(U_wav, V_wav, gridsize, pixel_resolution, pixperFWHM, w=W_wav, wstack=wstack)
        if briggs:
            if wstack:
                i_indices,j_indices,k_indices,i_conj_indices,j_conj_indices,k_conj_indices = uniform_grid(U_wav, V_wav, gridsize, pixel_resolution, pixperFWHM, w=W_wav, wstack=wstack)
            else:
                i_indices,j_indices,i_conj_indices,j_conj_indices = uniform_grid(U_wav, V_wav, gridsize, pixel_resolution, pixperFWHM, w=W_wav, wstack=wstack)
                #print("indices:",i_indices,j_indices,i_conj_indices,j_conj_indices)
                bweights = briggs_weighting(U_wav, V_wav, args.gridsize, robust=args.robust,pixel_resolution=pixel_resolution)

        for i in range(dat.shape[0]):
            for k in range(dat.shape[-1]):
                if briggs:
                    if k == 0 and jj == 0:
                        outimage[:,:,i,0] = revised_robust_image(dat[i:i+1, :, jj, k],
                                            U_wav,
                                            V_wav,
                                            gridsize,
                                            inject_img=None if np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/nchans_per_node,
                                            robust=robust,
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,
                                            wstack=wstack,
                                            w=W_wav,
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM,
                                            i_indices=i_indices,
                                            j_indices=j_indices,
                                            k_indices=None if not wstack else k_indices,
                                            i_conj_indices=i_conj_indices,
                                            j_conj_indices=j_conj_indices,
                                            k_conj_indices=None if not wstack else k_conj_indices)
                    else:
                        outimage[:,:,i,0] += revised_robust_image(dat[i:i+1, :, jj, k],
                                            U_wav,
                                            V_wav,
                                            gridsize,
                                            inject_img=None if np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/nchans_per_node,
                                            robust=args.robust,
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,
                                            wstack=wstack,
                                            w=W_wav,
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM,
                                            i_indices=i_indices,
                                            j_indices=j_indices,
                                            k_indices=None if not wstack else k_indices,
                                            i_conj_indices=i_conj_indices,
                                            j_conj_indices=j_conj_indices,
                                            k_conj_indices=None if not wstack else k_conj_indices)
                else:
                    if k == 0 and jj == 0:
                        outimage[:,:,i,0] = revised_uniform_image(dat[i:i+1, :, jj, k],
                                            U_wav,
                                            V_wav,
                                            gridsize,
                                            inject_img=None if np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/nchans_per_node,
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,wstack=wstack,
                                            w=W_wav,
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM)
                    else:
                        outimage[:,:,i,0] += revised_uniform_image(dat[i:i+1, :, jj, k],
                                            U_wav,
                                            V_wav,
                                            gridsize,
                                            inject_img=None if np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/nchans_per_node,
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,wstack=wstack,
                                            w=W_wav,
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM)

    print("it's sending from here")
    if port > 0 and len(ipaddress)>0:
        if TXexecutor is not None:
            TXtask_list.append(TXexecutor.submit(send_data,time_start_isot, uv_diag, Dec, outimage[:,:,:,0] ,None,23,'',128,True,5,10,port,ipaddress))
            if stagger>0: time.sleep(stagger)
        else:
            msg=send_data(time_start_isot, uv_diag, Dec, outimage ,verbose=True,retries=5,keepalive_time=10,port=port,ipaddress=ipaddress)
            print(msg)
    return outimage,j




#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
def main(args):

    verbose = args.verbose
    #send in sub-gulps
    
    num_gulps = 1#int(dat_all.shape[0]//args.num_time_samples)
    if args.num_gulps != -1:
        num_gulps = args.num_gulps#np.min([args.num_gulps,num_gulps])
    #num_chans = int(NUM_CHANNELS//AVERAGING_FACTOR)

    #randomly choose which gulp to inject burst in
    if args.inject:
        num_inject = args.num_inject
        if args.num_inject > num_gulps:
            num_inject = num_gulps
        inject_gulps = np.linspace(args.gulp_offset,args.gulp_offset + num_gulps,num_inject,dtype=int)
        #inject_gulps = np.random.choice(np.arange(args.gulp_offset, args.gulp_offset + num_gulps,dtype=int),replace=False,size=num_inject)

    #parameters from etcd
    #test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)

    #if verbose: print("TIMESTAMP:",tsamp)
    #get timestamp
    if len(args.timestamp) != 0:
        timestamp = args.timestamp

    if args.multiimage:
        print("Using multi-threaded imaging with ",args.maxProcesses,"threads")
        executor = ThreadPoolExecutor(args.maxProcesses)
    else: executor = None
    if args.multisend and len(args.multiport)>0:
        print("Using multi-threaded TX client ",args.maxProcesses,"threads and " + str(len(args.multiport)) + " ports")
        TXexecutor = ThreadPoolExecutor(args.maxProcesses)
        global TXtask_list
    else: TXexecutor = None

    dirty_img = np.nan*np.ones((args.gridsize,args.gridsize,args.num_time_samples,args.num_chans))
    #dirty_img_init = dict()
    #for i in range(args.num_chans):
    #    dirty_img_init[i] = False
    def image_future_callback(future):
        print("Callback ",future.result()[1])#,future.result()[2])
        dirty_img[:,:,:,future.result()[1]] = future.result()[0][:,:,:,0] #np.nansum(np.concatenate([dirty_img[:,:,:,future.result()[1],np.newaxis],future.result()[0][:,:,:,np.newaxis]],3),axis=3)
        """
        if not dirty_img_init[future.result()[1]]: 
            print(">>initializing")
            dirty_img[:,:,:,future.result()[1]] = future.result()[0]
            dirty_img_init[future.result()[1]] = True
            print(">>",dirty_img_init)
        else: dirty_img[:,:,:,future.result()[1]] += future.result()[0]
        """
        return

    for gulp in range(args.gulp_offset - (1 if args.gulp_offset>0 and args.search else 0),args.gulp_offset + num_gulps):
        
        #if searching, also need to find the previous integration set so we can initialize previous frame
        filelabels = [args.filelabel]
        if args.search and gulp==0:
            fnum = int(args.filelabel[1:])-1
            #look for previous label
            while len(glob.glob(args.path + "/lxd110h03/" + ("nsfrb_sb00" if args.sb else "h03") + "_" + str(fnum) + ".out")) == 0 and (int(args.filelabel[1:])-fnum)<10:
                fnum -=1
                continue
            if len(glob.glob(args.path + "/lxd110h03/" + ("nsfrb_sb00" if args.sb else "h03") + "_" + str(fnum) + ".out")) > 0:
                print("Using _" + str(fnum) + " for last frame initialization") 
                filelabels = ["_" + str(fnum)] + filelabels
                #if len(args.filedir) > 0:
                #    print("Copying files " + "_" + str(fnum) + " to " + args.filedir) 
                #    os.system("cp " + args.path + "/lxd110h*/" + ("nsfrb_sb*" if args.sb else "h*") + "_" + str(fnum) + ".out " + args.filedir)
            else:
                print("Couldn't find previous file, cannot initialize last frame")
        
        #read raw data for each corr node
        for g in range(len(filelabels)):

            #parameters from etcd
            test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
            ff = 1.53-np.arange(8192)*0.25/8192
            fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
            fobs = np.reshape(fobs,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1)
            #dat = dat_all[gulp*args.num_time_samples:(gulp+1)*args.num_time_samples,:,:,:]


            if filelabels[g] != args.filelabel:
                print("Making image to initialize last frame")
            dat = None
            Dec = None
            for i in range(len(corrs)):
                corr = corrs[i]
                sb = sbs[i]

                if len(args.filedir) == 0:
                    fname = args.path + "/lxd110"+ corr + "/" + ("nsfrb_" + sb if args.sb else corr) + filelabels[g] + ".out"
                else:
                    fname =  args.filedir + "/" + ("nsfrb_" + sb if args.sb else corr) + filelabels[g] + ".out"
                print(fname)
                #fname = args.path + "/lxd110"+ corr + "/" + corr + args.filelabel + ".out"
                #fname = args.path + "/3C286_vis/" + corr + args.filelabel + ".out"
                try:
                    #tmp = cal.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples)
                    #print("tmp",tmp)

                    dat_corr,sbnum,tstamp_mjd,Dec = pipeline.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples,gulp=(gulp if filelabels[g]==args.filelabel else (maxrawsamps//args.num_time_samples)-1),nchan=int(args.nchans_per_node),headersize=16)
                    if len(args.timestamp) == 0: 
                        timestamp = Time(tstamp_mjd,format='mjd').isot
            
                    #dat_corr = np.nanmean(dat_corr,axis=2,keepdims=True)
                    if verbose: print(dat_corr.shape)
                    if dat is None:
                        dat = np.nan*np.ones(dat_corr.shape,dtype=dat_corr.dtype).repeat(len(corrs),axis=2)
                    #print(dat_all.shape,dat_corr.shape)
                    dat[:,:,i*args.nchans_per_node:(i+1)*args.nchans_per_node,:] = dat_corr
                    #print("tmp2",dat_all[:,:,i,:],dat_corr)
                except Exception as exc:
                    if verbose: print("No data for " + corr)
                    if verbose: print(exc)
        

            print("Are any values nan?:",np.any(np.isnan(dat))) 
            #print(list(np.isnan(dat.mean((0,1,3)))))
            if verbose: print("Gulp size:",dat.shape)

        
            #use MJD to get pointing
            mjd = Time(timestamp,format='isot').mjd + ((gulp if filelabels[g]==args.filelabel else (maxrawsamps//args.num_time_samples)-1)*args.num_time_samples*tsamp/86400)
            time_start_isot = Time(mjd,format='mjd').isot
            #LST = Time(mjd,format='mjd').sidereal_time("mean",longitude=Lon).to(u.hourangle).value
            print("DEC from file:",Dec)



            pt_dec = Dec*np.pi/180.
            if verbose: print("Pointing dec (deg):",pt_dec*180/np.pi)
            bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)

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
            dat, bname, blen, UVW, antenna_order = flag_vis(dat, bname, blen, UVW, antenna_order, list(flagged_antennas) + list(args.flagants), bmin, list(flagged_corrs) + list(args.flagcorrs), flag_channel_templates = fcts)
            
            U = UVW[0,:,0]
            V = UVW[0,:,1]
            W = UVW[0,:,2]
            uv_diag=np.max(np.sqrt(U**2 + V**2))
            pixel_resolution = (0.20 / uv_diag) / args.pixperFWHM
            if verbose: print(antenna_order,len(antenna_order))#x_m.shape,y_m.shape,z_m.shape)
            if verbose: print(UVW.shape,U.shape,V.shape,W.shape)
            if verbose: print(UVW)

            print("Print bad channels:",np.isnan(dat.mean((0,1,3))))



            #pt_RA = LST*15*np.pi/180
            if verbose: print("Time:",time_start_isot)
            #if verbose: print("LST (hr):",LST)
            if Dec is None:

                RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag)
                #HA_axis = (LST*15) - RA_axis
                HA_axis = RA_axis[int(len(RA_axis)//2)] - RA_axis
                print(HA_axis)
                #HA_axis = RA_axis - RA_axis[int(len(RA_axis)//2)] #want to image the central RA, so the hour angle should be 0 here, right?
                RA = RA_axis[int(len(RA_axis)//2)]
                HA = HA_axis[int(len(HA_axis)//2)]
                Dec = Dec_axis[int(len(Dec_axis)//2)]
            else:
                #RA = get_ra(mjd,Dec) #LST*15
                #HA = 0
                RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,DEC=Dec)
                #HA_axis = (LST*15) - RA_axis
                HA_axis = RA_axis[int(len(RA_axis)//2)] - RA_axis
                RA = RA_axis[int(len(RA_axis)//2)]
                HA = HA_axis[int(len(HA_axis)//2)]
                print(HA_axis[len(HA_axis)//2-10:len(HA_axis)//2+10])
            if verbose: print("Coordinates (deg):",RA,Dec)
            if verbose: print("Hour angle (deg):",HA)



            #creating injection
            if args.inject and (gulp in inject_gulps) and filelabels[g]==args.filelabel:
                print("Injecting pulse in gulp",gulp)
                from inject import injecting
                offsetRA,offsetDEC,SNR,width,DM,maxshift = injecting.draw_burst_params(time_start_isot,RA_axis=RA_axis,DEC_axis=Dec_axis,gridsize=args.gridsize,nsamps=dat.shape[0],nchans=args.num_chans,tsamp=tsamp*1000,SNRmin=args.snr_min_inject,SNRmax=args.snr_max_inject)
                #offsetRA = offsetDEC = 0

                if args.snr_inject > 0:
                    SNR = args.snr_inject
                if args.dm_inject != -1 and args.dm_inject >= 0:
                    DM = args.dm_inject
                if args.width_inject > 0:
                    width = args.width_inject
                offsetRA = args.offsetRA_inject
                offsetDEC = args.offsetDEC_inject
                print("PARAMSFROM OFFLINE IMAGER:",offsetRA,offsetDEC,SNR,width,DM,maxshift,tsamp)
                print("OFFSET HOUR ANGLE:",HA_axis[int(len(HA_axis)//2 + offsetRA)])
                noiseless=False
                if args.solo_inject or args.flat_field or args.gauss_field:
                    #noiseless=False
                    dat[:,:,:,:] = 0
                if args.inject_noiseless:
                    noiseless=True
                #noiseless = True
                #DM = 0
                #SNR = 10000
                #width = 2
                #offsetRA = offsetDEC = 0
                inject_img = injecting.generate_inject_image(time_start_isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,snr=SNR,width=width,loc=0.5,gridsize=args.gridsize,nchans=args.num_chans,nsamps=dat.shape[0],DM=DM,maxshift=maxshift,offline=args.offline,noiseless=noiseless,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=args.inject_noiseonly,bmin=args.bmin,robust=args.robust if args.briggs else -2)

                if args.flat_field:
                    inject_img = np.ones_like(inject_img)
                elif args.gauss_field:
                    xx,yy = np.meshgrid(np.linspace(-2,2,args.gridsize),np.linspace(-2,2,args.gridsize))
                    inject_img = multivariate_normal(mean=[0,0],cov=0.5).pdf(np.dstack((xx,yy)))
                    inject_img = inject_img[:,:,np.newaxis,np.newaxis].repeat(dat.shape[0],2).repeat(args.num_chans,3)
                elif args.point_field:
                    inject_img = np.zeros_like(inject_img)
                    inject_img[int(args.gridsize//2)+offsetDEC,int(args.gridsize//2)+offsetRA] = 1
                #report injection in log file
                with open(inject_file,"a") as csvfile:
                    wr = csv.writer(csvfile,delimiter=',')
                    wr.writerow([time_start_isot,DM,width,SNR])
                csvfile.close()


            else:
                inject_img = np.zeros((args.gridsize,args.gridsize,dat.shape[0],args.num_chans))
            dat[np.isnan(dat)]= 0 
        
            #imaging
            print("Start imaging")
            if args.wstack: print("W-stacking with ",args.Nlayers," layers")
            dirty_img[:,:,:,:] = np.nan#*np.ones((args.gridsize,args.gridsize,dat.shape[0],args.num_chans))
            timage = time.time()
            if args.multiimage:
                task_list = []
                for j in range(args.num_chans):
                    print("submitting task:",j)
                    if args.search and filelabels[g] == args.filelabel and gulp>=args.gulp_offset:
                        if (args.multisend and len(args.multiport)>0):
                            port_j = args.multiport[int(j%len(args.multiport))]
                        elif args.multisend and len(args.multiport)==0:
                            port_j = args.port
                        else:
                            port_j = -1
                    else:
                        port_j = -1
                    task_list.append(executor.submit(offline_image_task,dat[:,:,j*args.nchans_per_node:(j+1)*args.nchans_per_node,:],
                                                    U,
                                                    V,
                                                    args.gridsize,
                                                    pixel_resolution,
                                                    args.nchans_per_node,
                                                    fobs[j*args.nchans_per_node:(j+1)*args.nchans_per_node],
                                                    j,
                                                    args.briggs,
                                                    args.robust,
                                                    False,
                                                    inject_img[:,:,:,j],
                                                    (args.point_field or args.gauss_field or args.flat_field),
                                                    args.wstack,
                                                    W,
                                                    args.Nlayers,
                                                    args.pixperFWHM,
                                                    port_j,
                                                    ipaddress,
                                                    time_start_isot, 
                                                    uv_diag, 
                                                    Dec,
                                                    TXexecutor,
                                                    args.stagger_multisend))
                    task_list[-1].add_done_callback(image_future_callback)
                wait(task_list)
            else:
                for j in range(args.num_chans):
                    for jj in range(args.nchans_per_node):
                        chanidx = (args.nchans_per_node*j)+jj
                        U_wav = U/(2.998e8/fobs[chanidx]/1e9)
                        V_wav = V/(2.998e8/fobs[chanidx]/1e9)
                        W_wav = None if not args.wstack else W/(2.998e8/fobs[chanidx]/1e9)
                        uniform_grid(U_wav, V_wav, args.gridsize, pixel_resolution, args.pixperFWHM, w=W_wav, wstack=args.wstack)
                        if args.briggs:
                            if args.wstack:
                                i_indices,j_indices,k_indices,i_conj_indices,j_conj_indices,k_conj_indices = uniform_grid(U_wav, V_wav, args.gridsize, pixel_resolution, args.pixperFWHM, w=W_wav, wstack=args.wstack)
                            else:
                                i_indices,j_indices,i_conj_indices,j_conj_indices = uniform_grid(U_wav, V_wav, args.gridsize, pixel_resolution, args.pixperFWHM, w=W_wav, wstack=args.wstack)
                            #print("indices:",i_indices,j_indices,i_conj_indices,j_conj_indices)
                            bweights = briggs_weighting(U_wav, V_wav, args.gridsize, robust=args.robust,pixel_resolution=pixel_resolution)

                        for i in range(dat.shape[0]):
                            for k in range(dat.shape[-1]):
                                if args.briggs:
                                    if k == 0 and jj == 0:
                                        dirty_img[:,:,i,j] = revised_robust_image(dat[i:i+1, :, chanidx, k],
                                            U_wav,
                                            V_wav,
                                            args.gridsize,
                                            inject_img=None if np.all(inject_img[:,:,i,j]==0) else inject_img[:,:,i,j]/dat.shape[-1]/args.nchans_per_node,
                                            robust=args.robust,
                                            inject_flat=(args.point_field or args.gauss_field or args.flat_field),
                                            pixel_resolution=pixel_resolution,
                                            wstack=args.wstack,
                                            w=W_wav,
                                            Nlayers_w=args.Nlayers,
                                            pixperFWHM=args.pixperFWHM,
                                            i_indices=i_indices,
                                            j_indices=j_indices,
                                            k_indices=None if not args.wstack else k_indices,
                                            i_conj_indices=i_conj_indices,
                                            j_conj_indices=j_conj_indices,
                                            k_conj_indices=None if not args.wstack else k_conj_indices)
                                    else:
                                        dirty_img[:,:,i,j] += revised_robust_image(dat[i:i+1, :, chanidx, k],
                                            U_wav,
                                            V_wav,
                                            args.gridsize,
                                            inject_img=None if np.all(inject_img[:,:,i,j]==0) else inject_img[:,:,i,j]/dat.shape[-1]/args.nchans_per_node,
                                            robust=args.robust,
                                            inject_flat=(args.point_field or args.gauss_field or args.flat_field),
                                            pixel_resolution=pixel_resolution,
                                            wstack=args.wstack,
                                            w=W_wav,
                                            Nlayers_w=args.Nlayers,
                                            pixperFWHM=args.pixperFWHM,
                                            i_indices=i_indices,
                                            j_indices=j_indices,
                                            k_indices=None if not args.wstack else k_indices,
                                            i_conj_indices=i_conj_indices,
                                            j_conj_indices=j_conj_indices,
                                            k_conj_indices=None if not args.wstack else k_conj_indices)
                                else:
                                    if k == 0 and jj == 0:
                                        dirty_img[:,:,i,j] = revised_uniform_image(dat[i:i+1, :, chanidx, k],
                                            U_wav,
                                            V_wav,
                                            args.gridsize,
                                            inject_img=None if np.all(inject_img[:,:,i,j]==0) else inject_img[:,:,i,j]/dat.shape[-1]/args.nchans_per_node,
                                            inject_flat=(args.point_field or args.gauss_field or args.flat_field),
                                            pixel_resolution=pixel_resolution,wstack=args.wstack,
                                            w=W_wav,
                                            Nlayers_w=args.Nlayers,
                                            pixperFWHM=args.pixperFWHM)
                                    else:
                                        dirty_img[:,:,i,j] += revised_uniform_image(dat[i:i+1, :, chanidx, k],
                                            U_wav,
                                            V_wav,
                                            args.gridsize,
                                            inject_img=None if np.all(inject_img[:,:,i,j]==0) else inject_img[:,:,i,j]/dat.shape[-1]/args.nchans_per_node,
                                            inject_flat=(args.point_field or args.gauss_field or args.flat_field),
                                            pixel_resolution=pixel_resolution,wstack=args.wstack,
                                            w=W_wav,
                                            Nlayers_w=args.Nlayers,
                                            pixperFWHM=args.pixperFWHM)
            print("Imaging complete:",time.time()-timage,"s")            
            print(dirty_img)
        
        
        
        
        
        
            #save image to fits, numpy file
            if args.save and filelabels[g] == args.filelabel and gulp>=args.gulp_offset:
                print("SAVING")
                np.save(args.outpath + "/" + time_start_isot + ".npy",dirty_img)
                numpy_to_fits(np.nanmean(dirty_img,(2,3)),args.outpath + "/" + time_start_isot + ".fits")
            
                if args.inject:
                    np.save(args.outpath + "/" + time_start_isot + "_response.npy",dirty_img/inject_img)
                    numpy_to_fits(np.nanmean(dirty_img,(2,3))/np.nanmean(inject_img,(2,3)),args.outpath + "/" + time_start_isot + "_response.fits")        

            #send to proc server
            if args.search and filelabels[g] == args.filelabel and gulp>=args.gulp_offset and not args.multisend:

                #if filelabels[g] == args.filelabel and gulp>=args.gulp_offset:
                #if args.multisend and len(args.multiport)>0:
                #    TXtask_list = []
                for i in range(args.num_chans):
                    #if args.multisend and len(args.multiport)>0:
                    #    TXtask_list.append(TXexecutor.submit(send_data,time_start_isot, uv_diag, Dec, dirty_img[:,:,:,i] ,None,23,'',128,args.verbose,5,10,args.multiport[int(i%len(args.multiport))],ipaddress))
                    #    if args.stagger_multisend>0: time.sleep(args.stagger_multisend)

                    #else:
                    #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
                    print("SENDING IMAGE OF SHAPE:",dirty_img[:,:,:,i].shape)
                    msg=send_data(time_start_isot, uv_diag, Dec, dirty_img[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.port)
                    if args.verbose: print(msg)
                    time.sleep(1)
                #if args.multisend and len(args.multiport)>0:
                    #wait(TXtask_list)
                    #for t in TXtask_list: print(t.result())
            elif args.search and args.multisend and len(args.multiport)>0 and filelabels[g] == args.filelabel and gulp>=args.gulp_offset:
                print(TXtask_list)
                wait(TXtask_list)
                for t in TXtask_list: print(t.result())
                TXtask_list = []
            ftime = open(timelogfile,"a")
            ftime.write("[image/send] " + str(time.time()-timage)+"\n")
            ftime.close()
            """
            if args.search:
                
                if filelabels[g] == args.filelabel and gulp>=args.gulp_offset:
                    if args.multisend and len(args.multiport)>0:
                        TXtask_list = []
                    for i in range(args.num_chans):
                        if args.multisend and len(args.multiport)>0:
                            TXtask_list.append(TXexecutor.submit(send_data,time_start_isot, uv_diag, Dec, dirty_img[:,:,:,i] ,None,23,'',128,args.verbose,5,10,args.multiport[int(i%len(args.multiport))],ipaddress))
                            if args.stagger_multisend>0: time.sleep(args.stagger_multisend)
                            
                        else:
                            #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
                            msg=send_data(time_start_isot, uv_diag, Dec, dirty_img[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.port)
                            if args.verbose: print(msg)
                            time.sleep(1)
                    if args.multisend and len(args.multiport)>0:
                        wait(TXtask_list)
                        for t in TXtask_list: print(t.result())
            """
            if filelabels[g] != args.filelabel or gulp < args.gulp_offset:#else:
                print("Writing to last_frame.npy")
                np.save(frame_dir + "last_frame.npy",dirty_img)
        time.sleep(args.sleeptime)
    if args.multiimage:
        executor.shutdown()
    if args.multisend and len(args.multiport)>0:
        TXexecutor.shutdown()
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('filelabel')           # positional argument
    parser.add_argument('--timestamp',type=str,help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49); if not given, timestamp is retrieved from sb00 file with os.path.getctime() or from time of rsync',default='')
    parser.add_argument('--filedir',type=str,help='Path to fast visibilities; if not given, the /dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/lxd110h**/ paths are used',default='')
    parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=-1)
    parser.add_argument('--gulp_offset',type=int,help='Gulp offset to start from, default = 0', default=0)
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    #parser.add_argument('--fringestop', action='store_true', default=False, help='Fringe stop manually')
    #parser.add_argument('--fringetable',type=str,help='Fringe stop manually with specified table in the dsa110-nsfrb-fast-visibilities dir',default='')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=raw_datasize)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vis_dir[:-1])
    parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--inject',action='store_true',default=False,help='Inject a burst into the gridded visibilities. Unless the --solo_inject flag is set, a noiseless injection will be integrated into the data.')
    parser.add_argument('--solo_inject',action='store_true',default=False,help='If set, visibility data will be zeroed and an injection with simulated noise will overwrite the data')
    parser.add_argument('--snr_inject',type=float,help='SNR of injection; default -1 which chooses a random SNR',default=-1)
    parser.add_argument('--snr_min_inject',type=float,help='Minimum injection S/N, default 1e7',default=1e7)
    parser.add_argument('--snr_max_inject',type=float,help='Maximum injection S/N, default 1e8',default=1e8)
    parser.add_argument('--dm_inject',type=float,help='DM of injection; default -1 which chooses a random DM',default=-1)
    parser.add_argument('--width_inject',type=int,help='Width of injection in samples; default -1 which chooses a random width',default=-1)
    parser.add_argument('--offsetRA_inject',type=int,help='Offset RA of injection in samples; default random', default=int(np.random.choice(np.arange(-IMAGE_SIZE//2,IMAGE_SIZE//2))))
    parser.add_argument('--offsetDEC_inject',type=int,help='Offset DEC of injection in samples; default random', default=int(np.random.choice(np.arange(-IMAGE_SIZE//2,IMAGE_SIZE//2))))
    parser.add_argument('--offline',action='store_true',default=False,help='Initializes previous frame with noise')
    parser.add_argument('--inject_noiseonly',action='store_true',default=False,help='Only inject noise; for use with false positive testing')
    parser.add_argument('--inject_noiseless',action='store_true',default=False,help='Only inject signal')
    parser.add_argument('--num_inject',type=int,help='Number of injections, must be less than number of gulps',default=1)
    parser.add_argument('--sb',action='store_true',default=False,help='Use nsfrb_sbxx names')
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=8)
    parser.add_argument('--flat_field',action='store_true',help='Illuminate all pixels uniformly')
    parser.add_argument('--gauss_field',action='store_true',help='Illuminate a gaussian source')
    parser.add_argument('--point_field',action='store_true',help='Illuminate a point source')
    parser.add_argument('--briggs',action='store_true',help='If set use robust weighted gridding with \'briggs\' weighting')
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging',default=0)
    parser.add_argument('--sleeptime',type=float,help='Time to sleep between processing gulps (seconds)',default=30)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--wstack',action='store_true',help='If set use w-stacking algorithm with --Nlayers layers')
    parser.add_argument('--Nlayers',type=int,help='Number of layers for w-stacking',default=18)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--flagSWAVE',action='store_true',help='Flag channels when SWAVE template RFI is detected, which manifests as a 2 Hz sin wave over ~5 minutes of data')
    parser.add_argument('--flagBPASS',action='store_true',help='Flag channels when BPASS template RFI is detected, which is simpl comparison to bandpass mean in visibilities')
    parser.add_argument('--flagFRCBAND',action='store_true',help='Flag channels in FRC miltiary allocation 1435-1525 MHz')
    parser.add_argument('--flagBPASSBURST',action='store_true',help='Flag channel when BPASS template RFI is detected in any timestep, i.e. should detect pulsed narrowband RFI')
    parser.add_argument('--flagcorrs',type=int,nargs='+',default=[],help='List of sb nodes [0-15] to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagants',type=int,nargs='+',default=[],help='List of antennas to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of processes used for multithreading; only used if --multiimage is set; default=16',default=16)
    parser.add_argument('--multiimage',action='store_true',help='If set, uses multithreading for imaging')
    parser.add_argument('--pixperFWHM',type=float,help='Pixels per FWHM, default 3',default=pixperFWHM)
    #parser.add_argument('--multiimagepol',action='store_true',help='If set with --multiimage flag, runs separate threads for each polarization, otherwise ignored')
    parser.add_argument('--multisend',action='store_true',help='If set, uses multithreading to send data to the process server')
    parser.add_argument('--stagger_multisend',type=float,help='Specifies the time in seconds between sending each subband, default 0 sends all at once',default=0)
    parser.add_argument('--port',type=int,help='Port number for receiving data from subclient, default = 8080',default=8080)
    parser.add_argument('--multiport',nargs='+',default=list(8810 + np.arange(16)),help='List of port numbers to listen on, default using single port specified in --port',type=int)
    args = parser.parse_args()
    main(args)



