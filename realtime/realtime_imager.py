import argparse
from dsacalib import constants as ct
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
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,pixperFWHM,chanbw,freq_axis_fullres,lambdaref,c,NSFRB_PSRDADA_KEY,NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,rtbench_file,nsamps,T
from nsfrb.imaging import inverse_revised_uniform_image,uv_to_pix, revised_robust_image,get_ra,briggs_weighting,uniform_grid
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST
from nsfrb.TXclient import send_data,ipaddress
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images
from tqdm import tqdm
import time
from scipy.stats import norm,multivariate_normal
#import nsfrb.searching as sl
from nsfrb.outputlogging import numpy_to_fits,printlog
#from nsfrb import calibration as cal
from nsfrb import pipeline
import os
#vispath = os.environ["NSFRBDATA"] + "dsa110-nsfrb-fast-visibilities" #cwd + "-fast-visibilities"
#imgpath = cwd + "-images"
#inject_file = cwd + "-injections/injections.csv"

from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,flagged_antennas,Lon,Lat,maxrawsamps,flagged_corrs,timelogfile,NSFRB_PSRDADA_TESTKEYS,local_inject_dir

import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
from dsautils import cnf
from realtime import rtreader
"""
This script reads raw fast visibility data from a file on disk, applies fringe-stopping from a pre-made table,
applies calibration, and images. If specified, the resulting image is transmitted to the process server.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m

#logger
logfile = "realtime_imager_log.txt"

# ETCD interface
from multiprocessing import Process, Queue
ETCD = ds.DsaStore()
ETCDKEY = f'/mon/nsfrb/fastvis'
ETCDKEY_INJECT = f'/mon/nsfrb/inject'
ETCDKEY_TIMING = f'/mon/nsfrb/timing'
ETCDKEY_TIMING_LIST = [f'/mon/nsfrbtiming/'+str(i+1) for i in range(len(corrs))]

#flagged antennas/
TXtask_list = []
def realtime_image_task(dat, tidx, U_wavs, V_wavs, i_indices_all, j_indices_all, i_conj_indices_all, j_conj_indices_all, bweights_all, gridsize,  pixel_resolution, nchans_per_node, fobs_j, jj, briggs=False, robust= 0.0, return_complex=False, inject_img=None, inject_flat=False, wstack=False, W_wavs=None,  Nlayers_w=18,pixperFWHM=pixperFWHM,wstack_parallel=False):#,port=-1,ipaddress="",time_start_isot="", uv_diag=-1, Dec=-1, TXexecutor=None, stagger=0):

    outimage = revised_robust_image(dat.mean(2),#.transpose((0,2,1)),#dat[i:i+1, :, jj, k],
                                            U_wavs,
                                            V_wavs,
                                            gridsize,
                                            inject_img=None if np.all(inject_img==0) else inject_img/dat.shape[-1]/nchans_per_node,
                                            robust=robust,
                                            uniform=(not briggs),
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,
                                            wstack=wstack,
                                            w=None if W_wavs is None else W_wavs,
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM,
                                            briggs_weights=None if (not briggs) else bweights_all,
                                            i_indices=i_indices_all,
                                            j_indices=j_indices_all,
                                            i_conj_indices=i_conj_indices_all,
                                            j_conj_indices=j_conj_indices_all,
                                            clipuv=False,keeptime=True,wstack_parallel=wstack_parallel)
    return outimage,tidx




#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
def main(args):

    #verbose = args.verbose

    if args.inject:
        inject_count = args.inject_interval


    #printlog("Using multi-threaded imaging with " + str(args.maxProcesses) + "threads",output_file=logfile)
    executor = ThreadPoolExecutor(args.maxProcesses)
    if args.multisend and len(args.multiport)>0:
        #printlog("Using multi-threaded TX client ",args.maxProcesses,"threads and " + str(len(args.multiport)) + " ports",output_file=logfile)
        TXexecutor = ThreadPoolExecutor(args.maxProcesses)
        global TXtask_list
    else: TXexecutor = None

    #initialize UVWs...note we MUST restart when declination is changed
    dirty_img = np.nan*np.ones((args.gridsize,args.gridsize,args.num_time_samples))
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
    fobs = (1e-3)*(np.reshape(freq_axis_fullres,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1))
    

    #pt_dec = Dec*np.pi/180.
    #if verbose: printlog("Pointing dec (deg):" + str(pt_dec*180/np.pi),output_file=logfile)
    bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)


    #flagging andd baseline cut
    tmp, bname, blen, UVW, antenna_order,keep = flag_vis(np.zeros((nsamps,UVW.shape[1],args.nchans_per_node,2)), bname, blen, UVW, antenna_order, list(flagged_antennas) + list(args.flagants), bmin, [], flag_channel_templates = [], flagged_chans=[], flagged_baseline_idxs=args.flagbase, bmax=args.bmax, returnidxs=True)
    #keep = np.sqrt(UVW[0,:,1]**2 + UVW[0,:,0]**2)>args.bmin
    U = UVW[0,:,1]
    V = UVW[0,:,0]
    W = UVW[0,:,2]
    uv_diag=np.max(np.sqrt(U**2 + V**2))
    pixel_resolution = (lambdaref / uv_diag) / args.pixperFWHM



    U_wavs = U[:,np.newaxis]/(ct.C_GHZ_M/fobs)
    V_wavs = V[:,np.newaxis]/(ct.C_GHZ_M/fobs)
    if args.wstack or args.wstack_parallel:
        W_wavs = W[:,np.newaxis]/(ct.C_GHZ_M/fobs)
    else:
        W_wavs = np.zeros((len(W),args.nchans_per_node))
    i_indices_all,j_indices_all,i_conj_indices_all,j_conj_indices_all = uniform_grid(U_wavs, V_wavs, args.gridsize, pixel_resolution, args.pixperFWHM)
    bweights_all = np.zeros(U_wavs.shape)
    if args.briggs:
        for jj in range(args.nchans_per_node):
            bweights_all[:,jj] = briggs_weighting(U_wavs[:,jj], V_wavs[:,jj], args.gridsize, robust=args.robust,pixel_resolution=pixel_resolution)


    
    #read and reshape into np array (25 times x 4656 baselines x 8 chans x 2 pols, complex)
    while True:
        dat = None
        while (dat is None) or dat.shape[0]<args.num_time_samples:
            #printlog("Waiting for data in psrdada buffer")
            #if args.testh23:
            #    dat_i,mjd,sb,Dec = rtreader.rtread(key=NSFRB_PSRDADA_TESTKEYS[args.sb],nchan=args.nchans_per_node,nbls=args.nbase,nsamps=args.num_time_samples)
            #else:
            dat_i,mjd,sb,Dec = rtreader.rtread(key=NSFRB_PSRDADA_KEY,nchan=args.nchans_per_node,nbls=args.nbase,nsamps=args.num_time_samples)
            #printlog(str((mjd,sb,Dec)),output_file=logfile)
            assert(sb==args.sb)
            print(Dec,pt_dec*180/np.pi)
            assert(np.abs((Dec*np.pi/180) - pt_dec)<1e-2)
            if dat is None:
                dat = dat_i
            else:
                dat = np.concatenate([dat,dat_i])
            #printlog(dat.shape,output_file=logfile)
        if args.testh23:
            mjd = Time.now().mjd

        timage = time.time()
        
        #manual flagging
        dat = dat[:,keep,:,:]
        if args.sb in list(flagged_corrs) + list(args.flagcorrs):
            dat[:] = np.nan
        fchans = np.array(args.flagchans,dtype=int)[np.logical_and(np.array(args.flagchans)>=args.sb*args.nchans_per_node,np.array(args.flagchans)<args.sb*args.nchans_per_node)]-(args.sb*args.nchans_per_node)
        dat[:,:,fchans,:]=np.nan

        #np.save(img_dir + "2025-02-16T20:36:48.010_rtvis.npy",dat)

        #if verbose: printlog("Collected 25 samples, imaging...",output_file=logfile)
        
        #use MJD to get pointing
        time_start_isot = Time(mjd,format='mjd').isot
        

        #creating injection
        if args.inject and (inject_count>=args.inject_interval):
            inject_count = 0
            #if verbose: printlog("Injecting pulse",output_file=logfile)

            #look for an injection in etcd
            injection_params = ETCD.get_dict(ETCDKEY_INJECT)
            if injection_params is None:
                #if verbose: printlog("Injection not ready, postponing",output_file=logfile)
                inject_count = args.inject_interval
            else:
                #update dict
                if 'ISOT' not in injection_params.keys():
                    injection_params['ISOT'] = time_start_isot
                print(injection_params)
                #acknowledge receipt
                if args.testh23:
                    for sbi in range(16):
                        injection_params["ack"][sbi] = True
                else:
                    injection_params["ack"][args.sb] = True

                #check if correct time
                if time_start_isot == injection_params['ISOT']:
                    if args.testh23:
                        for sbi in range(16):
                            injection_params["injected"][sbi] = True
                    else:
                        injection_params["injected"][args.sb] = True
                ETCD.put_dict(ETCDKEY_INJECT,injection_params)

                if time_start_isot == injection_params['ISOT']:
                    #if verbose: printlog("Injection" + injection_params['ID'] + "found",output_file=logfile)
                    fname = "injection_" + str(injection_params['ID']) + "_sb" +str("0" if args.sb<10 else "")+ str(args.sb) + ".npy"
                    #copy
                    if args.testh23:
                        os.system("cp " + inject_dir + "realtime_staging/" + "injection_" + str(injection_params['ID']) + "_sb" +str("0" if args.sb<10 else "")+ str(args.sb) + ".npy " + local_inject_dir)
                    else:
                        os.system("scp h24.pro.pvt:" + inject_dir + "realtime_staging/" + "injection_" + str(injection_params['ID']) + "_sb" +str("0" if args.sb<10 else "")+ str(args.sb) + ".npy " + local_inject_dir)
                    
                    
                    #read
                    inject_img = np.load(local_inject_dir + fname)
                    #clear data if we only want the injection
                    if injection_params['inject_only']: dat[:,:,:,:] = 0
                    inject_flat = injection_params['inject_flat']
                    if args.verbose: printlog("Done injecting",output_file=logfile)
        else:
            inject_flat = False
            inject_img = np.zeros((args.gridsize,args.gridsize,dat.shape[0]))
        dat[np.isnan(dat)]= 0


        
        #imaging
        #if verbose: printlog("Start imaging",output_file=logfile)
        if args.wstack and verbose: printlog("W-stacking with "+str(args.Nlayers)+" layers",output_file=logfile)
        dirty_img = np.zeros((args.gridsize,args.gridsize,dat.shape[0]))
        #j=args.sb
        
        task_list = []
        #for j in range(args.num_chans):
        tgrid = time.time()
        #if verbose: printlog("gridding in advance...",output_file=logfile)
        #make U,V,Ws in advance

        for jj in range(args.nchans_per_node):
            #if verbose: printlog("submitting task:"+str(jj),output_file=logfile)
            for tidx in range(5):
                task_list.append(executor.submit(realtime_image_task,dat[tidx*5:(tidx+1)*5,:,jj,:],
                                                    tidx,
                                                    U_wavs[:,jj],
                                                    V_wavs[:,jj],
                                                    i_indices_all[:,jj],
                                                    j_indices_all[:,jj],
                                                    i_conj_indices_all[:,jj],
                                                    j_conj_indices_all[:,jj],
                                                    None if not args.briggs else bweights_all[:,jj],
                                                    args.gridsize,
                                                    pixel_resolution,
                                                    args.nchans_per_node,
                                                    fobs[(args.nchans_per_node*args.sb)+jj],
                                                    jj,
                                                    args.briggs,
                                                    args.robust,
                                                    False,
                                                    inject_img[:,:,tidx*5:(tidx+1)*5]/dat.shape[-1]/args.nchans_per_node,
                                                    False,
                                                    (args.wstack or args.wstack_parallel),
                                                    W_wavs,
                                                    #k_indices_all,
                                                    #k_conj_indices_all,
                                                    args.Nlayers,
                                                    args.pixperFWHM,
                                                    args.wstack_parallel))
        wait(task_list)
        for t in task_list:
            m=t.result()
            dirty_img[:,:,m[1]*5:(m[1]+1)*5] += m[0] #t.result()
                                            
        
        #if verbose: printlog(str("Imaging complete:" + str(time.time()-timage) + "s"),output_file=logfile)
        rtime=time.time()-timage
        if args.testh23:
            for sbi in range(len(corrs)):
                timing_dict = ETCD.get_dict(ETCDKEY_TIMING_LIST[sbi])
                if timing_dict is None: timing_dict = dict()
                timing_dict["corr_num"] = sbi
                timing_dict["ISOT"] = time_start_isot
                timing_dict["image_time"] = rtime
                ETCD.put_dict(ETCDKEY_TIMING_LIST[sbi],timing_dict)
                #timing_dict[sbi]["tx_time"] = -1
        else:
            timing_dict = ETCD.get_dict(ETCDKEY_TIMING_LIST[args.sb])
            if timing_dict is None: timing_dict = dict()
            timing_dict["corr_num"] = args.sb
            timing_dict["ISOT"] = time_start_isot
            timing_dict["image_time"] = rtime
            ETCD.put_dict(ETCDKEY_TIMING_LIST[args.sb],timing_dict)
            #timing_dict[args.sb]["tx_time"] = -1
        #ETCD.put_dict(ETCDKEY_TIMING,timing_dict)

        ftime = open(rtbench_file,"a")
        ftime.write(str(rtime)+"\n")
        ftime.close()
        if rtime>T:
            executor.shutdown()
            printlog("Realtime exceeded, shutting down imager",output_file=logfile)
            return 


        #send to proc server
        if args.save:
            np.save(img_dir + time_start_isot + "_rtimage.npy",dirty_img)
        if args.search:
            if args.testh23:
                ttx = time.time()
                for sbi in range(len(corrs)):
                    msg=send_data(time_start_isot, uv_diag, Dec, dirty_img ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.multiport[int(sbi%len(args.multiport))])
            
                    #msg=send_data(time_start_isot, uv_diag, Dec, dirty_img ,verbose=args.verbose,retries=5,keepalive_time=10)
                    #if args.verbose: printlog(msg,output_file=logfile)
                    txtime = time.time()-ttx
                    timing_dict = ETCD.get_dict(ETCDKEY_TIMING_LIST[sbi])
                    if timing_dict is None: timing_dict = dict()
                    timing_dict["tx_time"] = txtime
                    ETCD.put_dict(ETCDKEY_TIMING_LIST[sbi],timing_dict)
            else:
                ttx = time.time()
                msg=send_data(time_start_isot, uv_diag, Dec, dirty_img ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.multiport[int(args.sb%len(args.multiport))])
                txtime = time.time()-ttx
                timing_dict = ETCD.get_dict(ETCDKEY_TIMING_LIST[args.sb])
                if timing_dict is None: timing_dict = dict()
                timing_dict["tx_time"] = txtime
                ETCD.put_dict(ETCDKEY_TIMING_LIST[args.sb],timing_dict)
        if args.inject:
            inject_count += 1   
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('sb',type=int)
    parser.add_argument('--timestamp',type=str,help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49); if not given, timestamp is retrieved from sb00 file with os.path.getctime() or from time of rsync',default='')
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=raw_datasize)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vis_dir[:-1])
    parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--inject',action='store_true',default=False,help='Inject a burst into the gridded visibilities. Unless the --solo_inject flag is set, a noiseless injection will be integrated into the data.')
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=1)
    parser.add_argument('--briggs',action='store_true',help='If set use robust weighted gridding with \'briggs\' weighting')
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging',default=0)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--bmax',type=float,help='Maximum baseline length to include, default=None',default=np.inf)
    parser.add_argument('--wstack',action='store_true',help='If set use w-stacking algorithm with --Nlayers layers')
    parser.add_argument('--wstack_parallel',action='store_true',help='If set uses parallel processing for w-stacking')
    parser.add_argument('--Nlayers',type=int,help='Number of layers for w-stacking',default=18)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--flagSWAVE',action='store_true',help='Flag channels when SWAVE template RFI is detected, which manifests as a 2 Hz sin wave over ~5 minutes of data')
    parser.add_argument('--flagBPASS',action='store_true',help='Flag channels when BPASS template RFI is detected, which is simpl comparison to bandpass mean in visibilities')
    parser.add_argument('--flagFRCBAND',action='store_true',help='Flag channels in FRC miltiary allocation 1435-1525 MHz')
    parser.add_argument('--flagBPASSBURST',action='store_true',help='Flag channel when BPASS template RFI is detected in any timestep, i.e. should detect pulsed narrowband RFI')
    parser.add_argument('--flagcorrs',type=int,nargs='+',default=[],help='List of sb nodes [0,15] to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagants',type=int,nargs='+',default=[],help='List of antennas to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagchans',type=int,nargs='+',default=[],help='List of channels [0,(16*nchans_per_node - 1)] to flag')
    parser.add_argument('--flagbase',type=int,nargs='+',default=[],help='List of baselines [0,4655] to flag')
    parser.add_argument('--nbase',type=int,help='Expected number of baselines',default=4656)
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of processes used for multithreading; only used if --multiimage is set; default=16',default=16)
    #parser.add_argument('--multiimage',action='store_true',help='If set, uses multithreading for imaging')
    parser.add_argument('--pixperFWHM',type=float,help='Pixels per FWHM, default 3',default=pixperFWHM)
    #parser.add_argument('--multiimagepol',action='store_true',help='If set with --multiimage flag, runs separate threads for each polarization, otherwise ignored')
    parser.add_argument('--multisend',action='store_true',help='If set, uses multithreading to send data to the process server')
    parser.add_argument('--stagger_multisend',type=float,help='Specifies the time in seconds between sending each subband, default 0 sends all at once',default=0)
    parser.add_argument('--port',type=int,help='Port number for receiving data from subclient, default = 8080',default=8080)
    parser.add_argument('--multiport',nargs='+',default=list(8810 + np.arange(16)),help='List of port numbers to listen on, default using single port specified in --port',type=int)
    parser.add_argument('-T','--testh23',action='store_true')
    parser.add_argument('--inject_interval',type=int,help='Number of gulps between injections',default=90)
    args = parser.parse_args()
    main(args)



