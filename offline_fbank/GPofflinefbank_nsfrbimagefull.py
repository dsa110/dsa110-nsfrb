import argparse
from nsfrb.imaging import uv_to_pix,stack_images
import jax
from jax import numpy as jnp
from nsfrb import jax_funcs
import json
import copy
from nsfrb.planning import get_RA_cutoff
from nsfrb import searching as sl
import etcd3
import tracemalloc
from dsacalib import constants as ct
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait
import glob
import csv
from matplotlib import pyplot as plt
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


from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,pixperFWHM,chanbw,freq_axis, freq_axis_fullres,lambdaref,c,NSFRB_PSRDADA_KEY,NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,rttx_file,rtbench_file,nsamps,T,bad_antennas,flagged_antennas,Lon,Lat,Height,maxrawsamps,flagged_corrs,inject_dir,local_inject_dir,rtmemory_file,vis_dir,frame_dir,cand_dir,cwd,table_dir,bin_slow,bin_imgdiff,tsamp_slow,tsamp_imgdiff,ngulps_per_file
from nsfrb.config import tsamp as tsamp_ms
from nsfrb.config import NROWSUBIMG,NSUBIMG,SUBIMGPIX,SUBIMGORDER,baseband_tsamp
from nsfrb.imaging import inverse_revised_uniform_image,uv_to_pix, revised_robust_image,get_ra,briggs_weighting,uniform_grid,realtime_robust_image
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST,simple_flag_image
from nsfrb.TXclient import send_data,ipaddress
import time
from scipy.stats import norm,multivariate_normal
from nsfrb import pipeline
import os




from nsfrb.config import local_inject_dir,rtbench_file,rttx_file,GPofflinestagefile
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
from dsautils import cnf
from realtime import rtreader
from realtime import rtwriter
from psrdada import Reader
from nsfrb.config import NSFRB_PSRDADA_KEY,nsamps,NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,IMAGE_SIZE,DSAX_PSRDADA_KEY
"""
This script reads raw fast visibility data from a file on disk, applies fringe-stopping from a pre-made table,
applies calibration, and images. If specified, the resulting image is transmitted to the process server.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","hh16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m

#logger
logfile = "realtime_imager_log.txt"

# ETCD interface
from multiprocessing import Process, Queue
ETCD = ds.DsaStore()
ETCDKEY_CANDS = f'/mon/nsfrb/candidates'
ETCDKEY = f'/mon/nsfrb/fastvis'
ETCDKEY_INJECT = f'/mon/nsfrb/inject'
ETCDKEY_TIMING = f'/mon/nsfrb/timing'
ETCDKEY_TIMING_LIST = [f'/mon/nsfrbtiming/'+str(i+1) for i in range(len(corrs))]
ETCDKEY_CORRSTAGGER = f'/mon/nsfrbstagger'

#flagged antennas/
TXtask_list = []
def realtime_image_task(dat, tidx, U_wavs, V_wavs, i_indices_all, j_indices_all, i_conj_indices_all, j_conj_indices_all, bweights_all, gridsize,  pixel_resolution, nchans_per_node, fobs_j, jj, briggs=False, robust= 0.0, return_complex=False, inject_img=None, inject_flat=False, wstack=False, W_wavs=None,  Nlayers_w=18,pixperFWHM=pixperFWHM,wstack_parallel=False,PB_all=None):#,port=-1,ipaddress="",time_start_isot="", uv_diag=-1, Dec=-1, TXexecutor=None, stagger=0):

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
                                            clipuv=False,keeptime=True,wstack_parallel=wstack_parallel)/(1 if PB_all is None else PB_all[:,:,np.newaxis])
    return outimage,tidx

def printlog(txt,output_file,end='\n'):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    print(txt,file=fout,end=end,flush=True)
    if output_file != "":
        fout.close()
    return

def send_data_task(sbi,time_start_isot, uv_diag, Dec, dirty_img,verbose,port,timeout,failsafe,timage,ipaddress,protocol):
    """
    task to send data to the process server; this is only required for testing, the 
    real implementation will only send data for one corr node in the foreground process
    """
    ttx = time.time()
    try:
        msg=send_data(time_start_isot, uv_diag, Dec, dirty_img ,verbose=verbose,retries=5,keepalive_time=timeout,port=port,ipaddress=ipaddress,udpchunksize=args.udpchunksize,protocol=protocol,udpoffset=0)
    except Exception as exc:
        if failsafe:
            raise(exc)
        else:
            print(exc)
    txtime = time.time()-ttx
    timing_dict = etcd_get_dict_catch(ETCD,ETCDKEY_TIMING_LIST[sbi]) #ETCD.get_dict(ETCDKEY_TIMING_LIST[sbi])
    if timing_dict is None: timing_dict = dict()
    timing_dict["tx_time"] = txtime
    timing_dict["tot_time"] = time.time()-timage
    etcd_put_dict_catch(ETCD,ETCDKEY_TIMING_LIST[sbi],timing_dict) #ETCD.put_dict(ETCDKEY_TIMING_LIST[sbi],timing_dict)
    return txtime

from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
from nsfrb.config import lambdaref
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

from multiprocessing import Queue
QQUEUE = Queue()
def etcd_to_stagger(etcd_dict,sb,queue=QQUEUE):
    """
    This is a callback function that waits for previous corr node to send data
    """
    if ((sb>0 and (etcd_dict['status'][sb-1] and not etcd_dict['status'][sb])) or 
        (sb==0 and np.all(np.array(etcd_dict['status'])))):
        QQUEUE.put(etcd_dict['status'])
    return 

def etcd_put_dict_catch(ETCD,ekey,edict,output_file=""):
    try:
        ETCD.put_dict(ekey,edict)
    except etcd3.exceptions.ConnectionFailedError:
        printlog("Failed to put ETCD dict",output_file=output_file)
    return

def etcd_get_dict_catch(ETCD,ekey,edict=None,output_file=""):
    try:
        return ETCD.get_dict(ekey)
    except etcd3.exceptions.ConnectionFailedError:
        printlog("Failed to get ETCD dict",output_file=output_file)
        return edict



def corrstagger_send_task(time_start_isot, uv_diag, Dec, dirty_img, retries,multiport,ipaddress,udpchunksize,protocol,sb,timage,rttimeout,corrstagger_future,flagcorrs,rtlog_file="",rterr_file="",verbose=False,debug=False,failsafe=False):
    corrstaggerdict = etcd_get_dict_catch(ETCD,ETCDKEY_CORRSTAGGER,edict=None if corrstagger_future is None else corrstagger_future.result(),output_file=rterr_file) #ETCD.get_dict(ETCDKEY_CORRSTAGGER)
    if corrstaggerdict is None:
        corrstaggerdict = dict()
        corrstaggerdict = [False]*16
    printlog("INIT CORRSTATUS: " + str(corrstaggerdict['status']),output_file=rtlog_file)
    printlog(">>>>>"+str(corrstaggerdict['status'][sb-1]),output_file=rtlog_file)
    printlog("WAITING FOR QUEUE...",output_file=rtlog_file)
    if sb>0 or (sb==0 and not np.all(np.array(corrstaggerdict['status']))):
        try:
            corrstaggerdict['status'] = QQUEUE.get(timeout=0.75*max([0,rttimeout - (time.time()-timage)]))
        except:
            printlog("QUEUE TIMED OUT",output_file=rterr_file)
    printlog("PROCEEDING"+str(corrstaggerdict['status']),output_file=rtlog_file)

    if sb==0:
        corrstaggerdict['status'] = [False]*16
        for i in flagcorrs:
            corrstaggerdict['status'][i] = True
    printlog("SB "+str(sb)+" STARTING TX WITH CORR STATUS:"+str(corrstaggerdict['status']),output_file=rtlog_file)
    printlog(">>>>>TIMEOUT:"+str((rttimeout - (time.time()-timage))),output_file=rtlog_file)

    ttx = time.time()
    if verbose: printlog("[TIME LEFT]"+str(rttimeout - (time.time()-timage))+" sec",output_file=rtlog_file)
    if (rttimeout - (time.time()-timage)) < 0.1:
        if verbose: printlog("WITHHOLD TX, OUT OF TIME",output_file=rtlog_file)
        #if inject: inject_count += 1
        corrstaggerdict['status'][:sb+1] = [True]*(sb+1)
        #for i in range(sb+1):
        #    corrstaggerdict['status'][i] = True
        printlog("TIMEOUT, NEW CORRSTATUS: " + str(corrstaggerdict['status']),output_file=rtlog_file)
        etcd_put_dict_catch(ETCD,ETCDKEY_CORRSTAGGER,corrstaggerdict,output_file=rterr_file) #ETCD.put_dict(ETCDKEY_CORRSTAGGER,corrstaggerdict)
        return corrstaggerdict

    try:
        msg=send_data(time_start_isot, uv_diag, Dec, dirty_img ,verbose=verbose,retries=retries,keepalive_time=(rttimeout - (time.time()-timage)),port=multiport[int(sb%len(multiport))],ipaddress=ipaddress,udpchunksize=udpchunksize,protocol=protocol)
    except Exception as exc:
        if failsafe:
            raise(exc)
        else:
            printlog(exc,output_file=rtlog_file)
                
                
    txtime = time.time()-ttx
    if verbose: printlog("TXTIME:"+str(txtime) + " sec",output_file=rtlog_file)
    timing_dict = etcd_get_dict_catch(ETCD,ETCDKEY_TIMING_LIST[sb],output_file=rterr_file) #ETCD.get_dict(ETCDKEY_TIMING_LIST[args.sb])
    if timing_dict is None: timing_dict = dict()
    timing_dict["tx_time"] = txtime
    timing_dict["tot_time"] = time.time()-timage
    etcd_put_dict_catch(ETCD, ETCDKEY_TIMING_LIST[sb],timing_dict, output_file=rterr_file) #ETCD.put_dict(ETCDKEY_TIMING_LIST[args.sb],timing_dict)
    #if inject: inject_count += 1
    corrstaggerdict['status'][:sb+1] = [True]*(sb+1)
    #for i in range(sb+1):
    #    corrstaggerdict['status'][i] = True

    printlog("DONE, NEW CORRSTATUS: " + str(corrstaggerdict['status']),output_file=rtlog_file)
    etcd_put_dict_catch(ETCD, ETCDKEY_CORRSTAGGER,corrstaggerdict,output_file=rterr_file) #ETCD.put_dict(ETCDKEY_CORRSTAGGER,corrstaggerdict)

    return corrstaggerdict

from threading import Lock
jaxdev_inuse_0 = Lock()
jaxdev_inuse_1 = Lock()
def filterbank_image_task(sb,pixel_resolution,args,datadir,gulp,rtbench_file,rtlog_file,keep,U_wavs,V_wavs,W_wavs,bweights_all,i_indices_all,j_indices_all,i_conj_indices_all,j_conj_indices_all,cudaimage,usedev,t_indices_all,fcts,fct_dat_run_mean,bname, blen, UVW, antenna_order):
    try:
        dat,sb_,mjd_init,dec = pipeline.read_raw_vis(datadir+"nsfrb_sb{a:02d}_{b}.out".format(a=sb,b=args.fnum),get_header=False,nsamps=args.num_time_samples,nchan=args.nchans_per_node,gulp=gulp,headersize=16)
    except Exception as exc:
        printlog("Couldn't find "+datadir+"nsfrb_sb{a:02d}_{b}.out".format(a=sb,b=args.fnum)+":"+str(exc),output_file=rtlog_file)
        return np.zeros((args.gridsize,args.gridsize,args.num_time_samples)),sb,fct_dat_run_mean

    if args.debug:
        printlog("--->READ TIME: "+str(time.time()-tbuffer)+" sec",output_file=rtbench_file)
    timage = time.time()
    if args.debug: tbuffer= time.time()
    mjd = mjd_init + (gulp*T/1000/86400)
    if args.verbose: printlog(">>"+str(mjd)+"<<",output_file=rtlog_file)


    #manual flagging
    dat = dat[:,keep,:,:]
    if sb in list(flagged_corrs) + list(args.flagcorrs):
        dat[:] = np.nan
    fchans = np.array(args.flagchans,dtype=int)[np.logical_and(np.array(args.flagchans)>=sb*args.nchans_per_node,np.array(args.flagchans)<sb*args.nchans_per_node)]-(sb*args.nchans_per_node)
    dat[:,:,fchans,:] = np.nan

    if len(fcts)>0 and not (sb in list(flagged_corrs) + list(args.flagcorrs)):
        dat, bname_f, blen_f, UVW_f, antenna_order_f,fct_dat_run_mean,keep_f = flag_vis(dat, bname, blen, UVW, antenna_order, [], 0, [], flag_channel_templates = fcts, flagged_chans=[], flagged_baseline_idxs=[], returnidxs=True,dat_run_means=fct_dat_run_mean)
        if args.verbose: printlog("Bandpass flagging successful: "+str(fct_dat_run_mean),output_file=rtlog_file)


    #use MJD to get pointing
    time_start_isot = Time(mjd,format='mjd').isot


    dat[np.isnan(dat)]= 0
    #if args.verbose: printlog("DATA [POST-INJECTION]>"+str(dat)+"; "+str(np.sum(np.isnan(dat))),output_file=rtlog_file)

    if args.debug: printlog("--->INJECT TIME: "+str(time.time()-tbuffer)+" sec",output_file=rtbench_file)
    if args.debug: tbuffer = time.time()

    #imaging
    #if verbose: printlog("Start imaging",output_file=logfile)
    if args.wstack and args.verbose: printlog("W-stacking with "+str(args.Nlayers)+" layers",output_file=rtlog_file)
    dirty_img = np.zeros((args.gridsize,args.gridsize,dat.shape[0]))
    #j=args.sb

    task_list = []
    #for j in range(args.num_chans):
    tgrid = time.time()
    #if verbose: printlog("gridding in advance...",output_file=logfile)
    #make U,V,Ws in advance
    """
    if cudaimage:
        if usedev == 0: jaxdev_inuse_0.acquire()
        else: jaxdev_inuse_1.acquire()
        printlog("Acquired device "+str(usedev),output_file=rtbench_file)
    """
    for j in range(args.nchans_per_node):
        jj = (args.nchans_per_node*sb)+j
        #if verbose: printlog("submitting task:"+str(jj),output_file=logfile)
        print(jj)
        if cudaimage:
            #tmpVIS_lm = np.concatenate([(np.nanmean(dat[:,:,j,:],2)*bweights_all[:,jj]*args.gridsize).flatten(),np.conj(np.nanmean(dat[:,:,j,:],2)*bweights_all[:,jj]*args.gridsize).flatten()])
            #tmpVIS_lm = np.concatenate([(np.nanmean(dat[:,:,j,:],2)*bweights_all[:,jj]*args.gridsize)[:,:,np.newaxis],(np.conj(np.nanmean(dat[:,:,j,:],2)*bweights_all[:,jj]*args.gridsize))[:,:,np.newaxis]],2).flatten()

            tmpVIS_lm = (np.nanmean(dat[:,:,j,:],2)*bweights_all[:,jj]*args.gridsize).flatten()
            print(t_indices_all,i_indices_all[:,jj],j_indices_all[:,jj])
            dirty_img += jax_funcs.realtime_robust_image_jit_lowmem(jax.device_put(tmpVIS_lm,jax.devices()[0]),
                                                            jax.device_put(jnp.zeros((dat.shape[0],args.gridsize,args.gridsize),dtype=complex),jax.devices()[0]),
                                                            jax.device_put(t_indices_all,jax.devices()[0]),
                                                            jax.device_put(i_indices_all[:,jj],jax.devices()[0]),
                                                            jax.device_put(j_indices_all[:,jj],jax.devices()[0]),
                                                            jax.device_put(i_conj_indices_all[:,jj],jax.devices()[0]),
                                                            jax.device_put(j_conj_indices_all[:,jj],jax.devices()[0]),
                                                            0)[0]
            """
            dirty_img += np.array(jax_funcs.realtime_robust_image_jit(jax.device_put(jnp.array(np.nanmean(dat[:,:,j,:],2)*bweights_all[:,jj]*args.gridsize),jax.devices()[usedev]),
                                        jax.device_put(jnp.zeros((dat.shape[0],args.gridsize,args.gridsize),dtype=complex),jax.devices()[usedev]),
                                        jax.device_put(i_indices_all[:,jj],jax.devices()[usedev]),
                                        jax.device_put(j_indices_all[:,jj],jax.devices()[usedev]),
                                        jax.device_put(i_conj_indices_all[:,jj],jax.devices()[usedev]),
                                        jax.device_put(j_conj_indices_all[:,jj],jax.devices()[usedev]),0)[0])
            """
        else:
            dirty_img += realtime_robust_image(np.nanmean(dat[:,:,j,:],2),
                                                    U_wavs[:,jj],
                                                    V_wavs[:,jj],
                                                    args.gridsize,
                                                    args.robust,
                                                    None,
                                                    pixel_resolution,
                                                    args.pixperFWHM,
                                                    bweights_all[:,jj],
                                                    i_indices_all[:,jj],
                                                    j_indices_all[:,jj],
                                                    i_conj_indices_all[:,jj],
                                                    j_conj_indices_all[:,jj],
                                                    0)[0]
    """if cudaimage:
        if usedev == 0: jaxdev_inuse_0.release()
        else: jaxdev_inuse_1.release()
        printlog("Released device "+str(usedev),output_file=rtbench_file)
    """
    if args.debug: printlog("--->["+str(sb)+"]IMAGE TIME:" + str(time.time()-tbuffer)+" sec",output_file=rtbench_file)
    return dirty_img,sb,fct_dat_run_mean

#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
def main(args,GPcurrentnoise,GPlastframe):
    #corrstaggerdict = ETCD.get_dict(ETCDKEY_CORRSTAGGER)
    #if corrstaggerdict is None:
    corrstagger_future = None
    corrstaggerdict = dict()
    corrstaggerdict['status'] = [True]*16
    #corrstaggerdict['status'][args.sb] = False
    etcd_put_dict_catch(ETCD,ETCDKEY_CORRSTAGGER,corrstaggerdict,output_file="") #ETCD.put_dict(ETCDKEY_CORRSTAGGER,corrstaggerdict)
    #if args.corrstagger_multisend>0:
    #ETCD.add_watch(ETCDKEY_CORRSTAGGER, lambda etcd_dict : etcd_to_stagger(etcd_dict,args.sb))
    
    os.system("> " + rtbench_file)
    os.system("> " + rtmemory_file)
    if len(args.rtlog)>0:
        os.system("> "+ args.rtlog)
    if len(args.rterr)>0:
        os.system("> "+ args.rterr)
    #verbose = args.verbose

    if args.inject:
        inject_count = args.inject_interval - args.inject_delay


    #printlog("Using multi-threaded imaging with " + str(args.maxProcesses) + "threads",output_file=logfile)
    executor = ThreadPoolExecutor(args.maxProcesses)
    if args.multisend and len(args.multiport)>0:
        #printlog("Using multi-threaded TX client ",args.maxProcesses,"threads and " + str(len(args.multiport)) + " ports",output_file=logfile)
        TXexecutor = ThreadPoolExecutor(args.maxProcesses)
        global TXtask_list
    else: TXexecutor = None

    #initialize UVWs...note we MUST restart when declination is changed
    dirty_img = np.nan*np.ones((args.gridsize,args.gridsize,args.num_time_samples))
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None)
    datadir = vis_dir + args.GPdir + "/"
    sb,mjd_init,dec = pipeline.read_raw_vis(datadir+"nsfrb_sb00_{b}.out".format(b=args.fnum),get_header=True,nsamps=args.num_time_samples,nchan=args.nchans_per_node,gulp=0,headersize=16)
    fobs = (1e-3)*(np.reshape(freq_axis_fullres,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1))
    

    pt_dec = dec*np.pi/180.
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
        for jj in range(bweights_all.shape[1]):
            bweights_all[:,jj] = briggs_weighting(U_wavs[:,jj], V_wavs[:,jj], args.gridsize, robust=args.robust,pixel_resolution=pixel_resolution)


    #create axes for GPU-based imaging
    """
    t_indices_gpu = np.concatenate([np.repeat(np.arange(nsamps,dtype=int),U.shape[0]),
                                    np.repeat(np.arange(nsamps,dtype=int),U.shape[0])],dtype=int)
    i_indices_gpu = np.zeros((i_indices_all.shape[0]*2*nsamps,i_indices_all.shape[1]),dtype=int)
    j_indices_gpu = np.zeros((i_indices_all.shape[0]*2*nsamps,i_indices_all.shape[1]),dtype=int)
    for jj in range(i_indices_all.shape[1]):
        i_indices_gpu[:,jj] = np.tile(np.concatenate([i_indices_all[:,jj],i_conj_indices_all[:,jj]],0),nsamps)
        j_indices_gpu[:,jj] = np.tile(np.concatenate([j_indices_all[:,jj],j_conj_indices_all[:,jj]],0),nsamps)
    """

    t_indices_gpu = np.repeat(np.arange(nsamps,dtype=int),U.shape[0])
    i_indices_gpu = np.zeros((i_indices_all.shape[0]*nsamps,i_indices_all.shape[1]),dtype=int)
    j_indices_gpu = np.zeros((i_indices_all.shape[0]*nsamps,i_indices_all.shape[1]),dtype=int)
    i_conj_indices_gpu = np.zeros((i_indices_all.shape[0]*nsamps,i_indices_all.shape[1]),dtype=int)
    j_conj_indices_gpu = np.zeros((i_indices_all.shape[0]*nsamps,i_indices_all.shape[1]),dtype=int)
    for jj in range(i_indices_all.shape[1]):
        i_indices_gpu[:,jj] = np.tile(i_indices_all[:,jj],nsamps)
        j_indices_gpu[:,jj] = np.tile(j_indices_all[:,jj],nsamps)
        i_conj_indices_gpu[:,jj] = np.tile(i_conj_indices_all[:,jj],nsamps)
        j_conj_indices_gpu[:,jj] = np.tile(j_conj_indices_all[:,jj],nsamps)

    
    #read and reshape into np array (25 times x 4656 baselines x 8 chans x 2 pols, complex)
    tasklist = []


    #set the dec, sb, and mjd
    Dec=dec
    #Dec = args.dec
    #sb = args.sb
    """
    f = open(args.mjdfile,"r")
    mjd_init = float(f.read())
    f.close()
    """
    rtlog_file = args.rtlog
    rterr_file = args.rterr
    #if args.verbose: printlog("STARTUP PARAMS:" + str((sb,Dec,mjd_init)),output_file=rtlog_file)
    startuperr = False


    if args.verbose: printlog("Ready for data",output_file=rtlog_file)
    
    #memory logging
    if args.debug:
        tracemalloc.start()
        mallocloop = 0
        startmalloc = tracemalloc.take_snapshot()
        f = open(rtmemory_file,"w")
        print("INITIAL MEMORY ALLOCATION",file=f)
        startstats = startmalloc.statistics('lineno')
        for i in range(len(startstats)):
            print(startstats[i],file=f)
        print("-"*20,file=f)
        print("")
        f.close()
    if args.verbose: 
        printlog("Will send data to IP " + str(args.ipaddress),output_file=rtlog_file)

    inject_count=0
    slow=args.slow and not args.imgdiff
    imgdiff=args.imgdiff
    if not imgdiff:
        max_gulps= ((maxrawsamps//args.num_time_samples)//(bin_slow if slow else 1))
        printlog("There are "+str(max_gulps) + " of "+str(args.num_time_samples*(bin_slow if slow else 1)) +" in each file",output_file=rtlog_file)
    else:
        max_gulps=1
        printlog("Imgdiff mode ==> 1 gulp",output_file=rtlog_file)
    gulps = np.arange(max([args.gulp_offset,0]),min([args.gulp_offset+args.num_gulps,max_gulps]),dtype=int)
    #initialize last_frame 
    if args.initframes:
        printlog("Initializing previous frames...",output_file=rtlog_file)
        GPlastframe = np.zeros((args.gridsize,args.gridsize,args.num_time_samples,args.num_chans))
        #sl.init_last_frame(args.gridsize,args.gridsize,args.num_time_samples-sl.maxshift,args.num_chans)
        #sl.init_last_frame(args.gridsize,args.gridsize,args.nsamps-sl.maxshift_slow,args.nchans,slow=True)

    #initialize noise stats
    if args.initnoise or args.initnoisezero:
        printlog("Initializing noise statistics...",output_file=rtlog_file)
        GPcurrentnoise = (np.zeros((len(sl.widthtrials),len(sl.DM_trials))),0)

        #noise.init_noise(sl.DM_trials,sl.widthtrials,args.gridsize,args.gridsize,zero=args.initnoisezero)
        #GP_current_noise = noise.noise_update_all(None,args.gridsize,args.gridsize,sl.DM_trials,sl.widthtrials,readonly=True)
        #np.save(noise_dir + "running_vis_mean.npy",None)
        #np.save(noise_dir + "running_vis_mean_burst.npy",None)

    allnoise = []


    #flagging
    fcts = []
    if args.flagSWAVE:
        fcts.append(fct_SWAVE)
    if args.flagBPASS:
        fcts.append(fct_BPASS)
    if args.flagFRCBAND:
        fcts.append(fct_FRCBAND)
    if args.flagBPASSBURST:
        fcts.append(fct_BPASSBURST)
    fct_dat_run_mean = [None]*len(fcts)
    if len(fcts)>0 and args.verbose: printlog("Bandpass flagging enabled",output_file=rtlog_file)

    appendinit=False

    for gulp in gulps:


        if args.debug: tbuffer = tbuffer1=time.time()


        #read from file
        image_tesseract =np.zeros((args.gridsize,args.gridsize,(args.num_time_samples if not imgdiff else (maxrawsamps//args.num_time_samples)//bin_imgdiff),1 if imgdiff else 16))
        RA_axis,DEC_axis,tmp = uv_to_pix(mjd_init,image_tesseract.shape[0],Lat=Lat,Lon=Lon,uv_diag=uv_diag,DEC=dec,pixperFWHM=args.pixperFWHM)
        printlog("Will make image of size:"+str(image_tesseract.shape),output_file=rtlog_file)
        if not (slow or imgdiff):
            
            img_id_isot = Time(mjd_init + ((gulp)*T/1000/86400),format='mjd').isot
            makeimg = args.makeimage
            if not makeimg and len(glob.glob(datadir+"ofbimage_" + img_id_isot + ".npy"))>0:
                try:
                    im_ = np.load(datadir+"ofbimage_" + img_id_isot + ".npy")
                    assert(im_.shape==image_tesseract.shape)
                    printlog("Using image from disk:"+datadir+"ofbimage_" + img_id_isot + ".npy",output_file=rtlog_file)
                    image_tesseract = im_
                except Exception as exc:
                    printlog("Image data not found:"+str(exc),output_file=rtlog_file)
                    makeimg = True
            else:
                makeimg = True
            if makeimg:
                    
                tasklist = []
                usedev = 0
                cudaimage = (args.cudaimage or args.mixedimage)
                if cudaimage and args.lockdev >= 0: usedev = args.lockdev
                if cudaimage: printlog("Using device "+str(args.lockdev),output_file=rtlog_file)
                t1 = time.time()
                for sb in range(16):
                    if (args.mixedimage and (sb%2==1)) or (not args.cudaimage):
                        printlog("USING CPU",output_file=rtlog_file)
                        tasklist.append(executor.submit(filterbank_image_task,sb,pixel_resolution,args,
                                            datadir,gulp,rtbench_file,rtlog_file,keep,
                                            U_wavs,V_wavs,W_wavs,bweights_all,
                                            i_indices_all,j_indices_all,
                                            i_conj_indices_all,j_conj_indices_all,
                                            False,usedev,None,fcts,fct_dat_run_mean,bname, blen, UVW, antenna_order))
                    elif (args.mixedimage and (sb%2==0)) or args.cudaimage:
                        printlog("USING GPU",output_file=rtlog_file)
                        tres = filterbank_image_task(sb,pixel_resolution,args,
                                            datadir,gulp,rtbench_file,rtlog_file,keep,
                                            U_wavs,V_wavs,W_wavs,bweights_all,
                                            i_indices_gpu,j_indices_gpu,
                                            i_conj_indices_gpu,j_conj_indices_gpu,
                                            True,usedev,t_indices_gpu,fcts,fct_dat_run_mean,bname, blen, UVW, antenna_order)
                        try:
                            fct_dat_run_mean = tres[2]

                            image_tesseract[:,:,:,tres[1]] = tres[0]
                        except Exception as exc:
                            printlog("no data")
                    if args.cudaimage and (args.lockdev < 0): 
                        usedev = (usedev+1)%2
                        printlog("Iterate device,"+str(usedev),output_file=rtlog_file)
                if (args.mixedimage) or (not args.cudaimage):
                    wait(tasklist)
                    for t in tasklist:
                        tres = t.result()
                        try:
                            fct_dat_run_mean = tres[2]
                            image_tesseract[:,:,:,tres[1]] = tres[0]
                        except Exception as exc:
                            printlog("no data",output_file=rtlog_file)
                printlog("Total imaging time:" + str(time.time() - t1)+" sec",output_file=rtlog_file)
        elif slow:
            printlog("SLOW-->looking for previously formed images...",output_file=rtlog_file)
            subintsize = (args.num_time_samples//bin_slow)
            slow_RA_cutoffs = []
            for subint in range(bin_slow):
                slow_RA_cutoffs.append(get_RA_cutoff(Dec,usefit=True,offset_s=tsamp_ms*args.num_time_samples*subint/1000))
                img_id_isot = Time(mjd_init + ((gulp+subint)*T/1000/86400),format='mjd').isot
                printlog(datadir+"ofbimage_" + img_id_isot + ".npy",output_file=rtlog_file)
                try:
                    im_ = np.load(datadir+"ofbimage_" + img_id_isot + ".npy")
                    #im_med = np.nanmedian(im_.reshape((args.gridsize,args.gridsize,subintsize,args.num_time_samples//subintsize,16)),3).repeat(args.num_time_samples//subintsize,2)
                    im_med = np.nanmedian(im_,2,keepdims=True)
                    image_tesseract[:,:,subint*subintsize:(subint+1)*subintsize,:] = np.nanmean((im_ - im_med).reshape((args.gridsize,args.gridsize,subintsize,args.num_time_samples//subintsize,16)),3) 
                    #image_tesseract[:,:,subint*subintsize:(subint+1)*subintsize,:] = np.nanmean((np.load(datadir+"ofbimage_" + img_id_isot + ".npy") - np.nanmedian(np.load(datadir+"ofbimage_" + img_id_isot + ".npy") ,2,keepdims=True)).reshape((args.gridsize,args.gridsize,subintsize,args.num_time_samples//subintsize,16)),2)
                    #image_tesseract[:,:,subint*subintsize:(subint+1)*subintsize,:] -= np.nanmedian(image_tesseract[:,:,subint*subintsize:(subint+1)*subintsize,:],2,keepdims=True)
                except Exception as exc:
                    printlog("Image data not found:"+str(exc),output_file=rtlog_file)
                

            printlog("stacking slow images...",output_file=rtlog_file)
            stack,tmp,tmp,min_gridsize = stack_images([image_tesseract[:,:,i*subintsize:(i+1)*subintsize,:] for i in range(bin_slow)],slow_RA_cutoffs)
            RA_axis = RA_axis[:min_gridsize]
            slow_RA_cutoff = args.gridsize - min_gridsize
            image_tesseract = np.concatenate(stack,axis=2)
            printlog("done",output_file=rtlog_file)

        elif imgdiff:
            printlog("IMGDIFF-->looking for previously formed images...",output_file=rtlog_file)
            image_tesseract_tmp =np.zeros((args.gridsize,args.gridsize,maxrawsamps//args.num_time_samples,1))
            slow_RA_cutoffs = []
            for subint in range(ngulps_per_file//bin_imgdiff):
                #slow_RA_cutoffs.append(get_RA_cutoff(Dec,usefit=True,offset_s=bin_imgdiff*tsamp_ms*args.num_time_samples*subint/1000))
                #image_tesseract_tmp =np.zeros((args.gridsize,args.gridsize,bin_imgdiff))#args.num_time_samples*bin_imgdiff))
                for subsubint in range(bin_imgdiff):
                    slow_RA_cutoffs.append(get_RA_cutoff(Dec,usefit=True,offset_s=tsamp_ms*args.num_time_samples*(bin_imgdiff*subint + subsubint)/1000))
                    img_id_isot = Time(mjd_init + ((gulp+(subint*bin_imgdiff + subsubint))*T/1000/86400),format='mjd').isot
                    printlog(datadir+"ofbimage_" + img_id_isot + ".npy",output_file=rtlog_file)
                    try:
                        im_ = np.load(datadir+"ofbimage_" + img_id_isot + ".npy")
                        image_tesseract_tmp[:,:,(subint*bin_imgdiff) + subsubint,0] = np.nanmean(im_ - np.nanmedian(im_,2,keepdims=True),(2,3))
                    except Exception as exc:
                        printlog("Image data not found:"+str(exc),output_file=rtlog_file)
                #image_tesseract[:,:,subint,0] = np.nanmean(image_tesseract_tmp,2)

            np.save("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline_fbank/tmp.npy",image_tesseract_tmp)
            printlog("stacking imgdiff images...",output_file=rtlog_file)
            stack,tmp,tmp,min_gridsize = stack_images([image_tesseract_tmp[:,:,i:(i+1),:] for i in range(image_tesseract_tmp.shape[2])],slow_RA_cutoffs)
            RA_axis = RA_axis[:min_gridsize]
            slow_RA_cutoff = args.gridsize - min_gridsize
            image_tesseract = np.nanmean(np.concatenate(stack,axis=2).reshape((args.gridsize,min_gridsize,(maxrawsamps//args.num_time_samples)//bin_imgdiff,bin_imgdiff,1)),3)
            printlog("done",output_file=rtlog_file)


        #make a flagged copy
        if (not args.imgdiff):
            postflagcorrs = list(args.postflagcorrs) + list(simple_flag_image(image_tesseract))
        printlog("Flagging:"+str(postflagcorrs),output_file=rtlog_file)
        image_tesseract_flagged = copy.deepcopy(image_tesseract)*image_tesseract.shape[3]/(image_tesseract.shape[3] - (len(args.flagcorrs) + len(args.postflagcorrs)))
        image_tesseract_flagged[:,:,:,np.array(postflagcorrs,dtype=int)] = 0


        
        mjd = mjd_init + (gulp*T/1000/86400)
        img_id_isot = Time(mjd,format='mjd').isot
        printlog("Completed image for gulp " + str(gulp) + "from file "+str(args.fnum) + "--> "+img_id_isot + " | "+str(image_tesseract.shape),output_file=rtlog_file)

        RA_axis,DEC_axis,tmp = uv_to_pix(mjd_init,image_tesseract.shape[0],Lat=Lat,Lon=Lon,uv_diag=uv_diag,DEC=dec,pixperFWHM=args.pixperFWHM)
        RA_axis_2D,DEC_axis_2D,tmp = uv_to_pix(mjd_init,image_tesseract.shape[0],Lat=Lat,Lon=Lon,uv_diag=uv_diag,DEC=dec,pixperFWHM=args.pixperFWHM,two_dim=True)
        time_axis = np.linspace(0,T,image_tesseract.shape[2])
        TOAs,image_tesseract_searched,image_tesseract_binned,total_noise = sl.run_search_GPU(image_tesseract_flagged,SNRthresh=args.SNRthresh,
                                                                            RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,canddict=dict(),
                                                                            usefft=args.usefft,multithreading=args.multithreading,
                                                                            nrows=args.nrows,ncols=args.ncols,
                                                                            output_file=rtlog_file,threadDM=args.threadDM,samenoise=args.samenoise,
                                                                            cuda=args.cuda,
                                                                            space_filter=args.spacefilter,kernel_size=args.kernelsize,
                                                                            exportmaps=args.exportmaps,
                                                                            append_frame= appendinit and (not imgdiff),DMbatches=args.DMbatches,
                                                                            SNRbatches=args.SNRbatches,usejax=args.usejax,noiseth=args.noiseth,
                                                                            RA_cutoff=get_RA_cutoff(dec,T=tsamp_ms*nsamps,pixsize=np.abs(RA_axis[1]-RA_axis[0])),
                                                                            DM_trials=sl.DM_trials,widthtrials=sl.widthtrials,
                                                                            applySNthresh=False,slow=slow,imgdiff=imgdiff,attach=dict(),
                                                                            completeness=False,forfeit=False,lockdev=args.lockdev,ofb_lastframe=GPlastframe[:,:image_tesseract.shape[1],:,:],
                                                                            ofb_currentnoise=GPcurrentnoise,ofb_lastframeslow=GPlastframe[:,:image_tesseract.shape[1],:,:])
        appendinit=True

        if args.debug:
            printlog("--->SEARCH TIME: "+str(time.time()-tbuffer)+" sec",output_file=rtbench_file)


        #global current_noise
        if (GPcurrentnoise[0][0,0] == 0) or (np.abs(total_noise[0,0] - GPcurrentnoise[0][0,0])<3*GPcurrentnoise[0][0,0]):
            GPcurrentnoise = (total_noise,GPcurrentnoise[1] +1)
            printlog("updating noise" + str(GPcurrentnoise),output_file=rtlog_file)
        #GP_current_noise = (noise.noise_update_all(total_noise,args.gridsize,args.gridsize,sl.DM_trials,sl.widthtrials,writeonly=True),GP_current_noise[1] + 1)
        #sl.save_last_frame(image_tesseract,full=True)
        GPlastframe = copy.deepcopy(image_tesseract)
        if (args.overwrite) and ((not (slow or imgdiff)) or (slow and args.saveslow) or (imgdiff and args.imgdiff)):
            #save image
            f = open(datadir+"ofbimage_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (not slow and imgdiff) else "") + ".npy","wb")
            np.save(f,image_tesseract)
            f.close()

            f = open(datadir+"ofbimage_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_searched.npy","wb")
            np.save(f,image_tesseract_searched)
            f.close()

            f = open(datadir+"ofbimage_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_TOAs.npy","wb")
            np.save(f,TOAs)
            f.close()

        if np.nanmax(image_tesseract_searched)>args.SNRthresh:
            #save image
            f = open(cand_dir + "raw_cands/" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (not slow and imgdiff) else "") + ".npy","wb")
            np.save(f,image_tesseract_flagged)
            f.close()

            f = open(cand_dir + "raw_cands/" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_searched.npy","wb")
            np.save(f,image_tesseract_searched)
            f.close()

            f = open(cand_dir + "raw_cands/" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_TOAs.npy","wb")
            np.save(f,TOAs)
            f.close()

            f = open(cand_dir + "raw_cands/" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_input.npy","wb")
            np.save(f,image_tesseract)
            f.close()

            if not imgdiff:
                #save image
                f = open(cand_dir + "raw_cands/" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (not slow and imgdiff) else "") + "_last.npy","wb")
                np.save(f,GPlastframe)
                f.close()


            printlog("Found candidates --> "+str("candidates_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + ".csv"),output_file=rtlog_file)
            ETCD.put_dict(
                    ETCDKEY_CANDS,
                    {
                        "candfile":"candidates_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + ".csv",
                        "uv_diag":uv_diag,
                        "dec":dec,
                        "img_shape":image_tesseract.shape,
                        "img_search_shape":image_tesseract_searched.shape
                    }
                )

        allnoise.append(GPcurrentnoise[0][0,0])
    if args.overwritenoise:
        f = open(datadir+"ofbimage_" + args.fnum + "_noisestats"+("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "")+".json","w")
        if not (slow or imgdiff):
            json.dump({"noise":GPcurrentnoise[0][0,0],
                "ngulps":GPcurrentnoise[1],
                "tottime_s":GPcurrentnoise[1]*tsamp_ms*args.num_time_samples/1000,
                "GPdir":args.GPdir,
                "fnum":args.fnum},f)
        elif slow:
            json.dump({"noise":GPcurrentnoise[0][0,0],
                "ngulps":GPcurrentnoise[1],
                "tottime_s":GPcurrentnoise[1]*tsamp_slow*args.num_time_samples/1000,
                "GPdir":args.GPdir,
                "fnum":args.fnum},f)
        elif imgdiff:
            json.dump({"noise":GPcurrentnoise[0][0,0],
                "ngulps":GPcurrentnoise[1],
                "tottime_s":GPcurrentnoise[1]*tsamp_ms*maxrawsamps/1000,
                "GPdir":args.GPdir,
                "fnum":args.fnum},f)
        f.close()

        np.save(datadir+"ofbimage_" + args.fnum + "_allnoise"+("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "")+".npy",np.array(allnoise))
    executor.shutdown()
    return GPcurrentnoise,GPlastframe


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    #parser.add_argument('--sb',type=int,help="sb num",default=0)
    parser.add_argument('--GPdir',type=str,help='directory of Galactic Plane data',default='')
    parser.add_argument('--fnum',type=str,help='filenumber',default='')
    parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=90)
    parser.add_argument('--gulp_offset',type=int,help='Gulp offset to start from, default = 0', default=0)
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=raw_datasize)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--inject',action='store_true',default=False,help='Inject a burst into the gridded visibilities. Unless the --solo_inject flag is set, a noiseless injection will be integrated into the data.')
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=8)
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
    parser.add_argument('--postflagcorrs',type=int,nargs='+',default=[],help='List of sb nodes [0,15] to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagants',type=int,nargs='+',default=[],help='List of antennas to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagchans',type=int,nargs='+',default=[],help='List of channels [0,(16*nchans_per_node - 1)] to flag')
    parser.add_argument('--flagbase',type=int,nargs='+',default=[],help='List of baselines [0,4655] to flag')
    parser.add_argument('--nbase',type=int,help='Expected number of baselines',default=4656)
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of processes used for multithreading; only used if --multiimage is set; default=16',default=16)
    parser.add_argument('--multiimage',action='store_true',help='If set, uses multithreading for imaging')
    parser.add_argument('--pixperFWHM',type=float,help='Pixels per FWHM, default 3',default=pixperFWHM)
    #parser.add_argument('--multiimagepol',action='store_true',help='If set with --multiimage flag, runs separate threads for each polarization, otherwise ignored')
    parser.add_argument('--multisend',action='store_true',help='If set, uses multithreading to send data to the process server')
    parser.add_argument('--stagger_multisend',type=float,help='Specifies the time in seconds between sending each subband, default 0 sends all at once',default=0)
    #parser.add_argument('--corrstagger_multisend',type=float,help='Specifies the time in seconds to query etcd for corr status',default=0)
    parser.add_argument('--port',type=int,help='Port number for receiving data from subclient, default = 8080',default=8080)
    parser.add_argument('--multiport',nargs='+',default=list(8810 + np.arange(16)),help='List of port numbers to listen on, default using single port specified in --port',type=int)
    parser.add_argument('-T','--testh23',action='store_true')
    parser.add_argument('--inject_interval',type=int,help='Number of gulps between injections',default=90)
    parser.add_argument('--inject_delay',type=float,help='Number of gulps to delay injection',default=0)
    parser.add_argument('--rttimeout',type=float,help='time to wait for search task to complete before cancelling, default=3 seconds',default=3)
    #parser.add_argument('--primarybeam',action='store_true',help='Apply a primary beam correction')
    parser.add_argument('--failsafe',action='store_true',help='Shutdown if real-time limit is exceeded')
    #parser.add_argument('--dec',type=float,help='Pointing declination',default=71.6)
    parser.add_argument('--mjdfile',type=str,help='MJD file',default='/home/ubuntu/tmp/mjd.dat')
    parser.add_argument('--rtlog',type=str,help='Send output to logfile specified, defaults to stdout',default='')
    parser.add_argument('--rterr',type=str,help='Send errors to logfile specified, defaults to stdout',default='')
    parser.add_argument('--debug',action='store_true',help='memory debugging')
    parser.add_argument('--retries',type=int,help='retries',default=1)
    parser.add_argument('--TXmode',type=str,choices=['subimg','subint','base'],default='base',help='TX mode')
    parser.add_argument('--TXnints',type=int,help='Number of sub-integrations for TXmode subint',default=5)
    parser.add_argument('--ipaddress',type=str,help='IP address of process server to send data to',choices=[os.environ["NSFRBIP"],os.environ["NSFRBIP2"]],default=os.environ["NSFRBIP"])
    parser.add_argument('--protocol',choices=['tcp','udp'],default='tcp',help='protocol to use to send data to process server,default=tcp')
    parser.add_argument('--udpchunksize',type=int,help='Data chunksize in bytes,default=25886',default=25886)
    parser.add_argument('--udproundup',action='store_true',help='Round sub-integration size up')
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 10',default=3)
    parser.add_argument('--usefft',action='store_true', help='Implement PSF spatial matched filter as a 2D FFT')
    parser.add_argument('--multithreading',action='store_true',help='Enable multithreading in search')
    parser.add_argument('--nrows',type=int,help='Number of rows to break image into if multithreading, default = 4',default=4)
    parser.add_argument('--ncols',type=int,help='Number of columns to break image into if multithreading, default = 2',default=2)
    parser.add_argument('--threadDM',action='store_true',help='Break DM trials among multiple threads')
    parser.add_argument('--samenoise',action='store_true',help='Assume the noise in each pixel is the same')
    parser.add_argument('--cuda',action='store_true',help='Uses PyTorch to accelerate computation with GPUs. The cuda flag overrides the multithreading option')
    parser.add_argument('--spacefilter',action='store_true', help='Use PSF to spatial matched filter the input image')
    parser.add_argument('--kernelsize',type=int,help='Kernel size for PSF spatial matched filter; default=151',default=151)
    parser.add_argument('--exportmaps',action='store_true',help='Output noise maps for each DM and width trial to the noise directory')
    parser.add_argument('--DMbatches',type=int,help='Number of pixel batches to submit dedispersion to the GPUs with, default = 1',default=1)
    parser.add_argument('--SNRbatches',type=int,help='Number of pixel batches to submit boxcar filtering to the GPUs with, default = 1',default=1)
    parser.add_argument('--usejax',action='store_true',help='Use JAX Just-In-Time compilation for GPU acceleration')
    parser.add_argument('--noiseth',type=float,help='S/N threshold below which samples are included in noise calculation; default=3',default=3)
    parser.add_argument('--initframes',action='store_true',help='Initializes previous frames for dedispersion')
    parser.add_argument('--initnoise',action='store_true',help='Initializes noise statistics from fast vis data for S/N estimates')
    parser.add_argument('--initnoisezero',action='store_true',help='Initializes noise to 0')
    parser.add_argument('--slow',action='store_true',help='Activate slow search pipeline, which bins data by 5 samples and re-searches')
    parser.add_argument('--imgdiff',action='store_true',help='Activate image differencing search pipeline, which bins data by 25 samples and searches 5-minute chunk at DM=0')
    parser.add_argument('--cleanup',action='store_true',help='Remove individual corr images when done')
    parser.add_argument('--model_weights3D',type=str, help='Path to the model weights file for 3D classifying',default=cwd + "/simulations_and_classifications/enhanced3dcnn_weights_final_remote.pth")
    parser.add_argument('--usepastimages',action='store_true',help='Use past images')
    parser.add_argument('--lockdev',type=int,help='Locks all search tasks to a single GPU, 0 or 1',default=-1)
    parser.add_argument('--cudaimage',action='store_true',help='GPU imaging')
    parser.add_argument('--mixedimage',action='store_true',help='Mixed GPU/CPU imaging')
    parser.add_argument('--saveslow',action='store_true',help='Save slow images to disk (testing only)')
    parser.add_argument('--saveimgdiff',action='store_true',help='Save image diff images to disk (testing only)')
    parser.add_argument('--makeimage',action='store_true',help='Make new image even if one is saved to disk')
    parser.add_argument('--overwrite',action='store_true',help='Save image to disk')
    parser.add_argument('--overwritenoise',action='store_true',help='Save noise to disk')
    args = parser.parse_args()

    GPcurrentnoise = (np.zeros((len(sl.widthtrials),len(sl.DM_trials))),0)
    GPlastframe = np.zeros((args.gridsize,args.gridsize,(args.num_time_samples if not args.imgdiff else (maxrawsamps//args.num_time_samples)//bin_imgdiff),16))
    #GPlastframe = np.zeros((args.gridsize,args.gridsize,args.num_time_samples,16))

    if len(args.fnum)==0:
        allfs = np.sort(glob.glob(vis_dir + args.GPdir + "/nsfrb_sb00_*.out"))
        print("Searching all data from "+args.GPdir)
        fi=0
        for fs in allfs:
            args.fnum=os.path.basename(fs)[os.path.basename(fs).index("sb00_")+5:-4]
            if fi>0:
                args.initframes = False
                args.initnoise = False
                args.initnoisezero = False
            print(fi,"<<<"+args.fnum+">>>")
            print("-"*100)
            print("")


            GPcurrentnoise,GPlastframe = main(args,GPcurrentnoise,GPlastframe)
            fi+=1
    else:
        print("<<<"+args.fnum+">>>")
        main(args,GPcurrentnoise,GPlastframe)

    



