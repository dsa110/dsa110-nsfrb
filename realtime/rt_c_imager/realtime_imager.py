import argparse
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


from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,pixperFWHM,chanbw,freq_axis_fullres,lambdaref,c,NSFRB_PSRDADA_KEY,NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,rttx_file,rtbench_file,nsamps,T,bad_antennas,flagged_antennas,Lon,Lat,Height,maxrawsamps,flagged_corrs,inject_dir,local_inject_dir,rtmemory_file
from nsfrb.config import NROWSUBIMG,NSUBIMG,SUBIMGPIX,SUBIMGORDER
from nsfrb.imaging import inverse_revised_uniform_image,uv_to_pix, revised_robust_image,get_ra,briggs_weighting,uniform_grid,realtime_robust_image
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST
from nsfrb.TXclient import send_data,ipaddress
import time
from scipy.stats import norm,multivariate_normal
from nsfrb import pipeline
import os

from nsfrb.config import local_inject_dir,rtbench_file,rttx_file
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




#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
def main(args):
    #corrstaggerdict = ETCD.get_dict(ETCDKEY_CORRSTAGGER)
    #if corrstaggerdict is None:
    corrstagger_future = None
    corrstaggerdict = dict()
    corrstaggerdict['status'] = [True]*16
    #corrstaggerdict['status'][args.sb] = False
    etcd_put_dict_catch(ETCD,ETCDKEY_CORRSTAGGER,corrstaggerdict,output_file="") #ETCD.put_dict(ETCDKEY_CORRSTAGGER,corrstaggerdict)
    #if args.corrstagger_multisend>0:
    ETCD.add_watch(ETCDKEY_CORRSTAGGER, lambda etcd_dict : etcd_to_stagger(etcd_dict,args.sb))
    
    os.system("> " + rtbench_file)
    os.system("> " + rtmemory_file)
    if len(args.rtlog)>0:
        os.system("> "+ args.rtlog)
    if len(args.rterr)>0:
        os.system("> "+ args.rterr)
    #verbose = args.verbose
    rtlog_file = args.rtlog
    rterr_file = args.rterr

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
    try:
        assert(np.abs(args.dec - (pt_dec*180/np.pi))<0.1)
    except:
        printlog("ALERT: CUSTOMDEC DISAGREES WITH ETCD POINTING DEC, DEFAULTING TO CUSTOMDEC --> " + str(args.dec) + " | " + str(pt_dec*180/np.pi),output_file=rterr_file)
        #args.dec=pt_dec*180/np.pi
        pt_dec = args.dec*np.pi/180
    Dec = pt_dec*180/np.pi
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
        for jj in range(bweights_all.shape[1]):
            bweights_all[:,jj] = briggs_weighting(U_wavs[:,jj], V_wavs[:,jj], args.gridsize, robust=args.robust,pixel_resolution=pixel_resolution)
    if args.primarybeam:
        PB_all = np.zeros((args.nchans_per_node,args.gridsize,args.gridsize))
        ra_grid_2D,dec_grid_2D,elev = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,two_dim=True,DEC=pt_dec*180/np.pi)
        for jj in range(args.nchans_per_node):
            chanidx = args.nchans_per_node*args.sb + jj
            PB_all[jj,:,:] = multivariate_normal.pdf(np.concatenate([ra_grid_2D[:,:,np.newaxis],
                                                                dec_grid_2D[:,:,np.newaxis]],2),
                                                        mean=(ra_grid_2D[args.gridsize//2,args.gridsize//2],
                                                              dec_grid_2D[args.gridsize//2,args.gridsize//2]),
                                                        cov=ellipse_to_covariance(1.22*((ct.C_GHZ_M/fobs[chanidx])/4.65)*180/np.pi/2.3548,
                                                                                  1.22*((ct.C_GHZ_M/fobs[chanidx])/4.65)*180/np.pi/2.3548,0))
            PB_all[jj,:,:] /= np.nanmax(PB_all[jj,:,:])
    else:
        PB_all = 1
    
    #read and reshape into np array (25 times x 4656 baselines x 8 chans x 2 pols, complex)
    gulp_counter = 0
    tasklist = []


    #set the dec, sb, and mjd
    #Dec = args.dec
    sb = args.sb
    """
    f = open(args.mjdfile,"r")
    mjd_init = float(f.read())
    f.close()
    """
    #rtlog_file = args.rtlog
    #rterr_file = args.rterr
    #if args.verbose: printlog("STARTUP PARAMS:" + str((sb,Dec,mjd_init)),output_file=rtlog_file)
    startuperr = False


    #create reader
    if args.verbose: printlog("Initializing reader...",output_file=rtlog_file)
    reader_connected=False
    ii=0
    
    while not reader_connected:
        try:
            reader = Reader(NSFRB_PSRDADA_KEY)
            reader_connected=True
        except Exception as exc:
            if args.verbose and ii==0:
                printlog("Trying to connect to ring buffer...",output_file=rtlog_file)
                printlog(exc,output_file=rtlog_file)
            continue
        ii+=1
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

    mjd_init = -1


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
    while True:


        if args.debug: tbuffer = tbuffer1=time.time()
        #dat = None
        #try:
        if args.verbose and args.debug:
            tmpfile = sys.stdout if len(args.rtlog)==0 else open(rtlog_file,"a")
            tmpfile2 = open(rtbench_file,"a")
            dat = rtreader.rtread(key=NSFRB_PSRDADA_KEY,nchan=args.nchans_per_node,nbls=args.nbase,nsamps=args.num_time_samples,readheader=False,reader=reader,verbose=True,verbosefile=tmpfile,verbosefile2=tmpfile2)
            if tmpfile != sys.stdout: tmpfile.close()
            tmpfile2.close()
        elif args.verbose:
            tmpfile = sys.stdout if len(args.rtlog)==0 else open(rtlog_file,"a")
            dat = rtreader.rtread(key=NSFRB_PSRDADA_KEY,nchan=args.nchans_per_node,nbls=args.nbase,nsamps=args.num_time_samples,readheader=False,reader=reader,verbose=True,verbosefile=tmpfile,verbosefile2=tmpfile)
            if tmpfile != sys.stdout: tmpfile.close()
        else:
            dat = rtreader.rtread(key=NSFRB_PSRDADA_KEY,nchan=args.nchans_per_node,nbls=args.nbase,nsamps=args.num_time_samples,readheader=False,reader=reader,verbose=False)
        #if args.verbose: printlog("DATA [PRE-FLAGGING]>"+str(dat)+"; "+str(np.sum(np.isnan(dat))),output_file=rtlog_file)
        if mjd_init == -1:

            f = open(args.mjdfile,"r")
            mjd_init = float(f.read())
            f.close()
            if args.verbose: printlog("STARTUP PARAMS:" + str((sb,Dec,mjd_init)),output_file=rtlog_file)
            
        if args.debug:
            printlog("--->READ TIME: "+str(time.time()-tbuffer)+" sec",output_file=rtbench_file)
        timage = time.time()
        if args.debug: tbuffer= time.time()
        mjd = mjd_init + (gulp_counter*T/1000/86400)
        gulp_counter += 1
        if args.verbose: printlog(">>"+str(mjd)+"<<",output_file=rtlog_file)
        #if args.testh23:
        #    mjd = Time.now().mjd

        
        #manual flagging
        dat = dat[:,keep,:,:]
        if args.sb in list(flagged_corrs) + list(args.flagcorrs):
            dat[:] = np.nan
        fchans = np.array(args.flagchans,dtype=int)[np.logical_and(np.array(args.flagchans)>=args.sb*args.nchans_per_node,np.array(args.flagchans)<args.sb*args.nchans_per_node)]-(args.sb*args.nchans_per_node)
        dat[:,:,fchans,:]=np.nan


        #bandpass flagging
        
        if len(fcts)>0 and not (args.sb in list(flagged_corrs) + list(args.flagcorrs)):
            dat, bname_f, blen_f, UVW_f, antenna_order_f,fct_dat_run_mean,keep_f = flag_vis(dat, bname, blen, UVW, antenna_order, [], 0, [], flag_channel_templates = fcts, flagged_chans=[], flagged_baseline_idxs=[], returnidxs=True,dat_run_means=fct_dat_run_mean)
            if args.verbose: printlog("Bandpass flagging successful: "+str(fct_dat_run_mean),output_file=rtlog_file)



        #if args.verbose: printlog("DATA [POST-FLAGGING]>"+str(dat)+"; "+str(np.sum(np.isnan(dat))),output_file=rtlog_file)
        
        #np.save(img_dir + "2025-02-16T20:36:48.010_rtvis.npy",dat)

        #if verbose: printlog("Collected 25 samples, imaging...",output_file=logfile)
        
        #use MJD to get pointing
        time_start_isot = Time(mjd,format='mjd').isot
        

        #creating injection
        inject_flat = False
        inject_img = np.zeros((args.gridsize,args.gridsize,dat.shape[0]))
        inject_now=False
        if args.inject and (inject_count>=args.inject_interval):
            inject_count = 0
            if args.verbose: printlog("Injecting pulse",output_file=rtlog_file)

            #look for an injection in etcd
            injection_params = etcd_get_dict_catch(ETCD, ETCDKEY_INJECT, output_file=rterr_file) #ETCD.get_dict(ETCDKEY_INJECT)
            if injection_params is None:
                if args.verbose: printlog("Injection not ready, postponing",output_file=rtlog_file)
                inject_count = args.inject_interval
            else:
                #update dict
                if 'ISOT' not in injection_params.keys():
                    injection_params['ISOT'] = time_start_isot
                printlog(injection_params,output_file=rtlog_file)
                #acknowledge receipt
                if args.testh23:
                    for sbi in range(16):
                        injection_params["ack"][sbi] = True
                else:
                    injection_params["ack"][args.sb] = True

                #check if correct time
                if True:#time_start_isot == injection_params['ISOT']:
                    if args.testh23:
                        for sbi in range(16):
                            injection_params["injected"][sbi] = True
                    else:
                        injection_params["injected"][args.sb] = True
                etcd_put_dict_catch(ETCD,ETCDKEY_INJECT,injection_params,output_file=rterr_file) #ETCD.put_dict(ETCDKEY_INJECT,injection_params)

                if True:#time_start_isot == injection_params['ISOT']:
                    #if verbose: printlog("Injection" + injection_params['ID'] + "found",output_file=logfile)
                    fname = "injection_" + str(injection_params['ID']) + "_sb" +str("0" if args.sb<10 else "")+ str(args.sb) + ".npy"
                    fname = injection_params['fname'] + str(args.sb) + ".npy"
                    
                    if args.verbose: printlog(fname,output_file=rtlog_file)
                    #read
                    try:
                        inject_img = np.load(local_inject_dir + fname)
                        assert(inject_img.shape==(args.gridsize,args.gridsize,dat.shape[0]))
                        inject_now=True
                    except Exception as exc:
                        inject_flat = False
                        inject_img = np.zeros((args.gridsize,args.gridsize,dat.shape[0]))
                        if args.verbose: printlog(str(args.sb)+" inject failed",output_file=rtlog_file)
                    #clear data if we only want the injection
                    if injection_params['inject_only']: dat[:,:,:,:] = 0
                    inject_flat = injection_params['inject_flat']
                    if args.verbose: printlog("Done injecting",output_file=rtlog_file)
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

        for j in range(args.nchans_per_node):
            jj = (args.nchans_per_node*args.sb)+j
            #if verbose: printlog("submitting task:"+str(jj),output_file=logfile)
            if args.multiimage:
                for tidx in range(5):
    

                    task_list.append(executor.submit(realtime_robust_image,
                                                    np.nanmean(dat[tidx*5:(tidx+1)*5,:,j,:],2),
                                                    U_wavs[:,jj],
                                                    V_wavs[:,jj],
                                                    args.gridsize,
                                                    args.robust,
                                                    None if (not inject_now) else inject_img[:,:,tidx*5:(tidx+1)*5]/dat.shape[-1]/args.nchans_per_node,
                                                    pixel_resolution,
                                                    args.pixperFWHM,
                                                    None if not args.briggs else bweights_all[:,jj],
                                                    i_indices_all[:,jj],
                                                    j_indices_all[:,jj],
                                                    i_conj_indices_all[:,jj],
                                                    j_conj_indices_all[:,jj],
                                                    tidx))
            
            else:
                dirty_img += realtime_robust_image(np.nanmean(dat[:,:,j,:],2),
                                                    U_wavs[:,jj],
                                                    V_wavs[:,jj],
                                                    args.gridsize,
                                                    args.robust,
                                                    None if (not inject_now) else inject_img/dat.shape[-1]/args.nchans_per_node,
                                                    pixel_resolution,
                                                    args.pixperFWHM,
                                                    None if not args.briggs else bweights_all[:,jj],
                                                    i_indices_all[:,jj],
                                                    j_indices_all[:,jj],
                                                    i_conj_indices_all[:,jj],
                                                    j_conj_indices_all[:,jj],
                                                    0)[0]
            """
                    task_list.append(executor.submit(realtime_image_task,dat[tidx*5:(tidx+1)*5,:,j,:],
                    #task_list.append(realtime_image_task(dat[tidx*5:(tidx+1)*5,:,j,:],
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
                                                    fobs[jj],
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
                                                    args.wstack_parallel,
                                                    None if not args.primarybeam else PB_all[j,:,:]))
            else:
                dirty_img += realtime_image_task(dat[:,:,j,:],
                                                    0,
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
                                                    fobs[jj],
                                                    jj,
                                                    args.briggs,
                                                    args.robust,
                                                    False,
                                                    inject_img/dat.shape[-1]/args.nchans_per_node,
                                                    False,
                                                    (args.wstack or args.wstack_parallel),
                                                    W_wavs,
                                                    #k_indices_all,
                                                    #k_conj_indices_all,
                                                    args.Nlayers,
                                                    args.pixperFWHM,
                                                    args.wstack_parallel,
                                                    None if not args.primarybeam else PB_all[j,:,:])[0]
        """
        if args.multiimage:
            
            wait(task_list)
            for t in task_list:
                m=t.result()
                dirty_img[:,:,m[1]*5:(m[1]+1)*5] += m[0] #t.result()
                                          
        #if args.verbose: printlog("DATA [POST-IMAGING]>"+str(dirty_img)+"; "+str(np.sum(np.isnan(dirty_img))),output_file=rtlog_file)
        

        if args.debug: printlog("--->IMAGE TIME:" + str(time.time()-tbuffer)+" sec",output_file=rtbench_file)
        if args.debug: tbuffer = time.time()
        #if verbose: printlog(str("Imaging complete:" + str(time.time()-timage) + "s"),output_file=logfile)
        rtime=time.time()-timage
        if args.testh23:
            for sbi in range(len(corrs)):
                timing_dict = etcd_get_dict_catch(ETCD,ETCDKEY_TIMING_LIST[sbi],output_file=rterr_file) #ETCD.get_dict(ETCDKEY_TIMING_LIST[sbi])
                if timing_dict is None: timing_dict = dict()
                timing_dict["corr_num"] = sbi
                timing_dict["ISOT"] = time_start_isot
                timing_dict["image_time"] = rtime
                etcd_put_dict_catch(ETCD,ETCDKEY_TIMING_LIST[sbi],timing_dict,output_file=rterr_file) #ETCD.put_dict(ETCDKEY_TIMING_LIST[sbi],timing_dict)
                #timing_dict[sbi]["tx_time"] = -1
        else:
            timing_dict = etcd_get_dict_catch(ETCD,ETCDKEY_TIMING_LIST[args.sb],output_file=rterr_file)#ETCD.get_dict(ETCDKEY_TIMING_LIST[args.sb])
            if timing_dict is None: timing_dict = dict()
            timing_dict["corr_num"] = args.sb
            timing_dict["ISOT"] = time_start_isot
            timing_dict["image_time"] = rtime
            etcd_put_dict_catch(ETCD,ETCDKEY_TIMING_LIST[args.sb],timing_dict,output_file=rterr_file) #ETCD.put_dict(ETCDKEY_TIMING_LIST[args.sb],timing_dict)
            #timing_dict[args.sb]["tx_time"] = -1
        #ETCD.put_dict(ETCDKEY_TIMING,timing_dict)

        rtwriter.rtwrite(dirty_img,key=DSAX_PSRDADA_KEY,addheader=False,header=dict(),dtype=np.float64)

        """
        if args.search:
            if args.multisend:
                corrstagger_future = executor.submit(corrstagger_send_task,
                                            time_start_isot, uv_diag, Dec, dirty_img, args.retries,
                                            args.multiport,args.ipaddress,args.udpchunksize,args.protocol,args.sb,time.time(),
                                            args.rttimeout,corrstagger_future,args.flagcorrs,
                                            rtlog_file,rterr_file,args.verbose,args.debug,args.failsafe)
            else:
                corrstaggerdict = corrstagger_send_task(time_start_isot, uv_diag, Dec, dirty_img, args.retries,
                                            args.multiport,args.ipaddress,args.udpchunksize,args.protocol,args.sb,timage,
                                            args.rttimeout,corrstagger_future,args.flagcorrs,
                                            rtlog_file,rterr_file,args.verbose,args.debug,args.failsafe)
        """
        if args.inject:
            inject_count += 1

        """











        if args.debug: printlog("--->ETCD TIME: " + str(time.time()-tbuffer)+" sec",output_file=rtbench_file)
        if args.debug: tbuffer = time.time()
        if args.failsafe and rtime>args.rttimeout:
            
            
            executor.shutdown()
            if args.verbose: printlog("Realtime exceeded, shutting down imager",output_file=rtlog_file)
            try:
                reader.disconnect()
            except Exception as e:
                pass
            return 


        #send to proc server
        if args.save:
            np.save(img_dir + time_start_isot + "_rtimage.npy",dirty_img)
        if args.search:
            if args.testh23:
                tasklist = []
                for sbi in range(len(corrs)):
                    if args.verbose: printlog("[TIME LEFT]"+str(args.rttimeout - (time.time()-timage))+" sec",output_file=rtlog_file)
                    tasklist.append(executor.submit(send_data_task,sbi,time_start_isot, uv_diag, Dec, dirty_img,args.verbose,args.multiport[int(sbi%len(args.multiport))],10,args.failsafe,timage,args.ipaddress,args.protocol))
                wait(tasklist)
                txtime = tasklist[-1].result()
            else:
                if args.stagger_multisend>0:
                    printlog("STAGGERING SB"+str(args.sb)+" BY "+str(args.sb*args.stagger_multisend)+" sec",output_file=rtlog_file)
                    time.sleep(args.sb*args.stagger_multisend)
                    printlog("DONE",output_file=rtlog_file)
                elif args.corrstagger_multisend>0:
                    printlog("MONITORING CORR DICT EVERY "+str(args.corrstagger_multisend)+" sec",output_file=rtlog_file)
                    corrstaggerdict = etcd_get_dict_catch(ETCD,ETCDKEY_CORRSTAGGER,edict=corrstaggerdict,output_file=rterr_file) #ETCD.get_dict(ETCDKEY_CORRSTAGGER)
                    printlog("INIT CORRSTATUS: " + str(corrstaggerdict['status']),output_file=rtlog_file)
                    printlog(">>>>>"+str(corrstaggerdict['status'][args.sb-1]),output_file=rtlog_file)
                    printlog("WAITING FOR QUEUE...",output_file=rtlog_file)
                    if args.sb>0 or (args.sb==0 and not np.all(np.array(corrstaggerdict['status']))):
                        try:
                            corrstaggerdict['status'] = QQUEUE.get(timeout=0.75*max([0,args.rttimeout - (time.time()-timage)]))
                        except:
                            printlog("QUEUE TIMED OUT",output_file=rterr_file)
                    printlog("PROCEEDING"+str(corrstaggerdict['status']),output_file=rtlog_file)
                    
                    if args.sb==0: 
                        corrstaggerdict['status'] = [False]*16
                        for i in args.flagcorrs:
                            corrstaggerdict['status'][i] = True
                    printlog("SB "+str(args.sb)+" STARTING TX WITH CORR STATUS:"+str(corrstaggerdict['status']),output_file=rtlog_file)
                    printlog(">>>>>TIMEOUT:"+str((args.rttimeout - (time.time()-timage))),output_file=rtlog_file)

                ttx = time.time()
                if args.verbose: printlog("[TIME LEFT]"+str(args.rttimeout - (time.time()-timage))+" sec",output_file=rtlog_file)
                if (args.rttimeout - (time.time()-timage)) < 0.1:
                    if args.verbose: printlog("WITHHOLD TX, OUT OF TIME",output_file=rtlog_file)
                    if args.inject: inject_count += 1
                    if args.corrstagger_multisend>0:
                        for i in range(args.sb+1):
                            corrstaggerdict['status'][i] = True
                        #corrstaggerdict['status'] = [True]*16 #just for testing
                        printlog("TIMEOUT, NEW CORRSTATUS: " + str(corrstaggerdict['status']),output_file=rtlog_file)
                        etcd_put_dict_catch(ETCD,ETCDKEY_CORRSTAGGER,corrstaggerdict,output_file=rterr_file) #ETCD.put_dict(ETCDKEY_CORRSTAGGER,corrstaggerdict)
                    continue
                try:
                    if args.TXmode=='subimg':
                        msg_or_udpoffset=0
                        for sidx in range(len(SUBIMGORDER)):
                            print(">>>",sidx)
                            msg_or_udpoffset=send_data(time_start_isot, uv_diag, Dec, dirty_img[SUBIMGPIX*SUBIMGORDER[sidx][0]:SUBIMGPIX*(SUBIMGORDER[sidx][0]+1),
                                                                                   SUBIMGPIX*SUBIMGORDER[sidx][1]:SUBIMGPIX*(SUBIMGORDER[sidx][1]+1),:] ,
                                                                                verbose=args.verbose,retries=args.retries,keepalive_time=(args.rttimeout - (time.time()-timage)),port=args.multiport[int(args.sb%len(args.multiport))],ipaddress=args.ipaddress,udpchunksize=args.udpchunksize,protocol=args.protocol,udpoffset=(0 if args.protocol=='tcp' else msg_or_udpoffset))
                    elif args.TXmode=='subint' and args.TXnints>1:
                        stime=(args.rttimeout - (time.time()-timage))/args.TXnints
                        stasks=[]
                        for sidx in range(args.TXnints):
                            print(">>>",sidx,(-16*(sidx%2))+args.multiport[int(args.sb%len(args.multiport))])
                            if sidx<args.TXnints-1:
                                subintsize = int(dirty_img.shape[2]//args.TXnints)
                                minidx = subintsize*sidx
                                maxidx = minidx + subintsize
                            else:
                                subintsize = dirty_img.shape[2] - int(dirty_img.shape[2]//args.TXnints)*sidx
                                maxidx = dirty_img.shape[2]
                                minidx = maxidx - subintsize
                            print(">>>",sidx,(-16*(sidx%2))+args.multiport[int(args.sb%len(args.multiport))],(minidx,maxidx))
                            msg=send_data(time_start_isot, uv_diag, Dec, dirty_img[:,:,minidx:maxidx],None,args.sb,'',128,args.verbose,args.retries,(args.rttimeout - (time.time()-timage)),args.multiport[int(args.sb%len(args.multiport))],args.ipaddress,args.protocol,args.udpchunksize,0)

                    else:
                        msg=send_data(time_start_isot, uv_diag, Dec, dirty_img ,verbose=args.verbose,retries=args.retries,keepalive_time=(args.rttimeout - (time.time()-timage)),port=args.multiport[int(args.sb%len(args.multiport))],ipaddress=args.ipaddress,udpchunksize=args.udpchunksize,protocol=args.protocol)
                except Exception as exc:
                    if args.failsafe:
                        raise(exc)
                    else:
                        printlog(exc,output_file=rtlog_file)
                txtime = time.time()-ttx
                if args.verbose: printlog("TXTIME:"+str(txtime) + " sec",output_file=rtlog_file)
                timing_dict = etcd_get_dict_catch(ETCD,ETCDKEY_TIMING_LIST[args.sb],output_file=rterr_file) #ETCD.get_dict(ETCDKEY_TIMING_LIST[args.sb])
                if timing_dict is None: timing_dict = dict()
                timing_dict["tx_time"] = txtime
                timing_dict["tot_time"] = time.time()-timage
                etcd_put_dict_catch(ETCD, ETCDKEY_TIMING_LIST[args.sb],timing_dict, output_file=rterr_file) #ETCD.put_dict(ETCDKEY_TIMING_LIST[args.sb],timing_dict)
                if args.corrstagger_multisend>0:
                    for i in range(args.sb+1):
                        corrstaggerdict['status'][i] = True

                    #corrstaggerdict['status'] = [True]*16 #just for testing
                    #corrstaggerdict['status'][args.sb-1] = False
                    printlog("DONE, NEW CORRSTATUS: " + str(corrstaggerdict['status']),output_file=rtlog_file)
                    etcd_put_dict_catch(ETCD, ETCDKEY_CORRSTAGGER,corrstaggerdict,output_file=rterr_file) #ETCD.put_dict(ETCDKEY_CORRSTAGGER,corrstaggerdict)
            if args.failsafe and time.time()-timage>args.rttimeout:
                executor.shutdown()
                if args.verbose: printlog("Realtime exceeded, shutting down imager",output_file=rtlog_file)
                try:
                    reader.disconnect()
                except Exception as e:
                    pass
                return
        if args.inject:
            inject_count += 1
        if args.debug:
            printlog("--->TX TIME: " + str(time.time()-tbuffer) + " sec",output_file=rtlog_file)
            printlog("--->TX TIME: " + str(time.time()-tbuffer) + " sec",output_file=rtbench_file)
            printlog("TOTAL TIME: " + str(time.time()-tbuffer1) + " sec",output_file=rtbench_file)
            printlog("-"*20,output_file=rtbench_file)
        #del dat
        #del dirty_img
        #del inject_img

        if args.debug:
            endstats = tracemalloc.take_snapshot().compare_to(startmalloc,'lineno')
            f = open(rtmemory_file,"a")
            print("LOOP " + str(mallocloop) + "MEMORY ALLOCATION",file=f)
            for i in range(len(endstats)):
                print(endstats[i],file=f)
            print("-"*20,file=f)
            print("",file=f)
            f.close()
            mallocloop += 1
        #break
        """
    executor.shutdown()
    try:
        reader.disconnect()
    except Exception as e:
        pass
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('--sb',type=int,help="sb num",default=0)
    parser.add_argument('--timestamp',type=str,help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49); if not given, timestamp is retrieved from sb00 file with os.path.getctime() or from time of rsync',default='')
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=raw_datasize)
    #parser.add_argument('--path',type=str,help='Path to raw data files',default=vis_dir[:-1])
    #parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
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
    parser.add_argument('--primarybeam',action='store_true',help='Apply a primary beam correction')
    parser.add_argument('--failsafe',action='store_true',help='Shutdown if real-time limit is exceeded')
    parser.add_argument('--dec',type=float,help='Pointing declination',default=71.6)
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
    args = parser.parse_args()
    main(args)



