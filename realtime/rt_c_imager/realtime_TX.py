import argparse
import struct
import copy
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
    dirty_img = np.nan*np.ones((args.gridsize,args.gridsize,args.num_time_samples),dtype=np.float64)
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None)
    try:
        assert(np.abs(args.dec - (pt_dec*180/np.pi))<0.1)
    except:
        printlog("ALERT: CUSTOMDEC DISAGREES WITH ETCD POINTING DEC, DEFAULTING TO CUSTOMDEC --> " + str(args.dec) + " | " + str(pt_dec*180/np.pi),output_file=rterr_file)
        pt_dec = args.dec*np.pi/180 #args.dec=pt_dec*180/np.pi
    Dec = pt_dec*180/np.pi
    fobs = (1e-3)*(np.reshape(freq_axis_fullres,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1))
    

    #pt_dec = Dec*np.pi/180.
    #if verbose: printlog("Pointing dec (deg):" + str(pt_dec*180/np.pi),output_file=logfile)
    bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)

    #write antennas to .bin
    ant1 = []
    ant2 = []
    for i in range(len(bname)):
        ant1.append(list(bname)[i][:list(bname)[i].index("-")])
        ant2.append(list(bname)[i][list(bname)[i].index("-")+1:])



    print("Final UVW Shape:"+str(UVW.shape))
    UVW = UVW.astype(np.float64)
    blen = np.sqrt(UVW[0,:,0]**2 + UVW[0,:,1]**2).astype(np.float64)
    with open(args.outdir + "U.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(struct.pack("<d",UVW[0,i,0]))
    with open(args.outdir + "V.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(struct.pack("<d",UVW[0,i,1]))
    with open(args.outdir + "W.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(struct.pack("<d",UVW[0,i,2]))
    with open(args.outdir + "BLEN.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(struct.pack("<d",blen[i]))
    with open(args.outdir + "ANT1.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(np.uint8(ant1[i]).tobytes())
    with open(args.outdir + "ANT2.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(np.uint8(ant2[i]).tobytes())


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
            reader = Reader(DSAX_PSRDADA_KEY)#NSFRB_PSRDADA_KEY)
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
    inject_count=0
    while True:


        if args.debug: tbuffer = tbuffer1=time.time()
        #dat = None
        #try:
        if args.verbose and args.debug:
            tmpfile = sys.stdout if len(args.rtlog)==0 else open(rtlog_file,"a")
            tmpfile2 = open(rtbench_file,"a")
            dirty_img = rtreader.rtread_imaging(key=DSAX_PSRDADA_KEY,gridsize=args.gridsize,nsamps=args.num_time_samples,reader=reader,verbose=True,verbosefile=tmpfile,verbosefile2=tmpfile2) #rtreader.rtread(key=NSFRB_PSRDADA_KEY,nchan=args.nchans_per_node,nbls=args.nbase,nsamps=args.num_time_samples,readheader=False,reader=reader,verbose=True,verbosefile=tmpfile,verbosefile2=tmpfile2)
            if tmpfile != sys.stdout: tmpfile.close()
            tmpfile2.close()
        elif args.verbose:
            tmpfile = sys.stdout if len(args.rtlog)==0 else open(rtlog_file,"a")
            dirty_img = rtreader.rtread_imaging(key=DSAX_PSRDADA_KEY,gridsize=args.gridsize,nsamps=args.num_time_samples,reader=reader,verbose=True,verbosefile=tmpfile,verbosefile2=tmpfile) #rtreader.rtread(key=NSFRB_PSRDADA_KEY,nchan=args.nchans_per_node,nbls=args.nbase,nsamps=args.num_time_samples,readheader=False,reader=reader,verbose=True,verbosefile=tmpfile,verbosefile2=tmpfile)
            if tmpfile != sys.stdout: tmpfile.close()
        else:
            dirty_img = rtreader.rtread_imaging(key=DSAX_PSRDADA_KEY,gridsize=args.gridsize,nsamps=args.num_time_samples,reader=reader,verbose=True) #rtreader.rtread(key=NSFRB_PSRDADA_KEY,nchan=args.nchans_per_node,nbls=args.nbase,nsamps=args.num_time_samples,readheader=False,reader=reader,verbose=False)
        dirty_img = copy.deepcopy(dirty_img)
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
        time_start_isot = Time(mjd,format='mjd').isot


        #creating injection
        inject_flat = False
        inject_img = np.zeros((args.gridsize,args.gridsize,dirty_img.shape[-1]))
        inject_now=False
        if args.inject and (inject_count>=args.inject_interval):
            inject_count = 0
            #if verbose: printlog("Injecting pulse",output_file=logfile)

            #look for an injection in etcd
            injection_params = etcd_get_dict_catch(ETCD, ETCDKEY_INJECT, output_file=rterr_file) #ETCD.get_dict(ETCDKEY_INJECT)
            if injection_params is None:
                #if verbose: printlog("Injection not ready, postponing",output_file=logfile)
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
                if time_start_isot == injection_params['ISOT']:
                    if args.testh23:
                        for sbi in range(16):
                            injection_params["injected"][sbi] = True
                    else:
                        injection_params["injected"][args.sb] = True
                etcd_put_dict_catch(ETCD,ETCDKEY_INJECT,injection_params,output_file=rterr_file) #ETCD.put_dict(ETCDKEY_INJECT,injection_params)

                if time_start_isot == injection_params['ISOT']:
                    #if verbose: printlog("Injection" + injection_params['ID'] + "found",output_file=logfile)
                    fname = "injection_" + str(injection_params['ID']) + "_sb" +str("0" if args.sb<10 else "")+ str(args.sb) + ".npy"
                    fname = injection_params['fname'] + str(args.sb) + ".npy"

                    if args.verbose: printlog(fname,output_file=rtlog_file)
                    #read
                    try:
                        inject_img = np.load(local_inject_dir + fname)
                        assert(inject_img.shape==(args.gridsize,args.gridsize,dirty_img.shape[-1]))
                        inject_now=True
                    except Exception as exc:
                        inject_flat = False
                        inject_img = np.zeros((args.gridsize,args.gridsize,dirty_img.shape[-1]))
                        if args.verbose: printlog(str(args.sb)+" inject failed",output_file=rtlog_file)
                    #clear data if we only want the injection
                    if injection_params['inject_only']: dirty_img[:,:,:] = 0
                    inject_flat = injection_params['inject_flat']
                    if args.verbose: printlog("Done injecting",output_file=rtlog_file)
            dirty_img += inject_img
        np.save("TESTIMAGE.npy",dirty_img)

        if args.debug: tbuffer = time.time()

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
        inject_count += 1

        if args.debug:
            printlog("--->TX TIME: "+str(time.time()-tbuffer)+" sec",output_file=rtbench_file)

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
    parser.add_argument("--outdir",type=str,help='output directory',default="./")
    args = parser.parse_args()
    main(args)



