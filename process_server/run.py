import numpy as np
import select
import os
import jax
import socket
import time
from matplotlib import pyplot as plt
import random
import copy
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import truncnorm
from scipy.signal import peak_widths
from scipy.stats import norm
from event import names
#from gen_dmtrials_copy import gen_dm
import argparse
from astropy.time import Time
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
#cwd = os.environ['NSFRBDIR']

import sys
#sys.path.append(cwd + "/") #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
import csv
import copy

from nsfrb.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset
from nsfrb.noise import init_noise,noise_update_all,get_noise_dict
from simulations_and_classifications import generate_PSF_images as scPSF
#from nsfrb.simulating import make_PSF_cube
fsize=45
fsize2=35
plt.rcParams.update({
                    'font.size': fsize,
                    'font.family': 'sans-serif',
                    'axes.labelsize': fsize,
                    'axes.titlesize': fsize,
                    'xtick.labelsize': fsize,
                    'ytick.labelsize': fsize,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 1,
                    'lines.markersize': 5,
                    'legend.fontsize': fsize2,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})


"""
This file runs the process server which receives data from the RX server and buffers it until data from all 16 channels 
is received; then it starts the search pipeline
"""
#from nsfrb import searching as sl
from nsfrb import pipeline
from nsfrb import plotting as pl
from nsfrb import config
from nsfrb import jax_funcs
"""s
Directory for output data
"""
"""
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = cwd + "/src/.pipestatus.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = cwd + "/scripts/script_flags/searchlog_flags.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_flags/searchlog_flags.txt"
output_file = cwd + "-logfiles/run_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
processfile = cwd + "-logfiles/process_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt"
flagfile = cwd + "/process_server/process_flags.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_flags.txt"
cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/"#cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
psf_dir = cwd + "-PSF/"
error_file = cwd + "-logfiles/error_log.txt"
"""
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,Lon,Lat

"""
NSFRB modules
"""
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack 
from nsfrb.imaging import uv_to_pix

"""
Dask manager
"""
"""
from dask.distributed import Client,Queue,fire_and_forget

QSETUP = False
if 'DASKPORT' in os.environ.keys():
    try:
        QCLIENT = Client("tcp://127.0.0.1:"+os.environ['DASKPORT'],timeout=1,heartbeat_interval=1000)#get_client()
        QWORKERS = ['proc_server_WRKR']
        QSETUP = True
        QQUEUE = Queue("cand_cutter_queue")
    except TimeoutError as exc:
        printlog("Scheduler not started, cannot send to queue",output_file=processfile)
    except OSError as exc:
        printlog("Scheduler not started, cannot send to queue",output_file=processfile)
"""
import dsautils.dsa_store as ds
ETCD = ds.DsaStore()
ETCDKEY = f'/mon/nsfrb/candidates'

from nsfrb import searching as sl
"""if 'DASKPORT' in os.environ.keys():
    QCLIENT = Client("tcp://127.0.0.1:"+os.environ['DASKPORT'])
    QWORKERS = ['proc_server_WRKR']#-0','cand_cutter_WRKR-1']
    QQUEUE = Queue("cand_cutter_queue")"""
"""
HTTP variables
"""
success = "HTTP/1.1 200 OK\nContent-Type: text/plain\nContent-Length: 40\n\nFile Received, Process Server Status: "
HEADER_DELIM = '0d0a0d0a'
##defines function to set flags for process server
pflagdict = dict()
pflagdict['parse_error'] = 1
pflagdict['datasize_error'] = 2
pflagdict['shape_error'] = 4
pflagdict['invalid'] = 8
pflagdict['all'] = 15
def set_pflag_loc(flag=None,on=True,reset=False):
    if (not (flag in pflagdict.keys())): return None
    return pflagdict[flag]	




"""
Create a structure for full image
"""
class fullimg:
    def __init__(self,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape=(32,32,25,16),dtype=np.float16):
        self.image_tesseract = np.zeros(shape,dtype=dtype)
        self.corrstatus = np.zeros(16,dtype=bool)
        self.img_id_isot = img_id_isot
        self.img_id_mjd = img_id_mjd
        self.img_uv_diag =img_uv_diag
        self.shape = shape
        self.img_dec = img_dec
        #get ra and dec axes
        self.RA_axis,self.DEC_axis,tmp = uv_to_pix(self.img_id_mjd,self.shape[0],Lat=Lat,Lon=Lon,uv_diag=img_uv_diag,DEC=img_dec)
        printlog("Created RA and DEC axes of size" + str(self.RA_axis.shape) + "," + str(self.DEC_axis.shape),output_file=processfile)
        printlog(self.RA_axis,output_file=processfile)
        printlog(self.DEC_axis,output_file=processfile)
    def add_corr_img(self,data,corr_node,testmode=False):
        self.image_tesseract[:,:,:,corr_node] = data
        #if testmode:
        self.corrstatus[corr_node] = 1
        return
	    
    def is_full(self):
        return np.all(self.corrstatus==1)

def find_id(img_id_isot,fullimg_array):
    printlog(fullimg_array,output_file=processfile)
    for i in range(len(fullimg_array)):
        printlog(fullimg_array[i] is None,output_file=processfile)
        if fullimg_array[i] is not None:
            printlog(fullimg_array[i].corrstatus,output_file=processfile)
            printlog(fullimg_array[i].img_id_isot,output_file=processfile)
    if len(fullimg_array) == 0: return -1,-1

    #also want to see if any spaces are open
    openidx = -1

    for i in range(len(fullimg_array)):
        
        if fullimg_array[i] is None and openidx == -1:
            openidx = i
        elif fullimg_array[i] is None: 
            continue
        elif fullimg_array[i].img_id_isot == img_id_isot: 
            printlog(img_id_isot + " " + str(fullimg_array[i].img_id_isot), output_file=processfile)
            return i,-1
    return -1,openidx

"""
Dictionary that maps corr nodes to ip addresses
"""
corraddrs = {'10.41.0.91' : 0, #sb00/corr03
            '10.41.0.117' : 1, #sb01/corr04
            '10.41.0.79' : 2, #sb02/corr05
            '10.41.0.127' : 3, #sb03/corr06
            '10.41.0.116' : 4, #sb04/corr07
            '10.41.0.99' : 5, #sb05/corr08
            '10.41.0.122' : 6, #sb06/corr10
            '10.41.0.121' : 7, #sb07/corr11
            '10.41.0.61' : 8, #sb08/corr12
            '10.41.0.115' : 9, #sb09/corr14
            '10.41.0.113' : 10, #sb10/corr15
            '10.41.0.83' : 11, #sb11/corr16
            '10.41.0.92' : 12, #sb12/corr18
            '10.41.0.103' : 13, #sb13/corr19
            '10.41.0.82' : 14, #sb14/corr21
            '10.41.0.71' : 15, #sb15/corr22
            '10.41.0.5' : 0, #182' : 0, #h23, placeholder
            '10.42.0.115' : 0,#'10.41.0.94' : 0 #corr20
            '10.42.0.232' : 0,
            '10.41.0.254' : 0, #h24
            '10.42.0.228' : 0
}

dtypelookup = {1 : np.int8,
               2 : np.float16,
               3 : np.int16,
               4 : np.float32,
               5 : np.int32,
               8 : np.float64,
               9 : np.int64,
               16: np.float128,
}

"""
b"PUT /_h23_IMG2023-10-03T21:56:46.215.npy HTTP/1.1\r\nAccept-Encoding: identity\r\nHost: 10.41.0.94:8080\r\nUser-Agent: curl/7.78.0\r\nAccept: */*\r\nReferer: rbose\r\nContent-Length: 1440128\r\nExpect: 100-continue\r\n\r\n\x93NUMPY\x01\x00v\x00{'descr': '<f8', 'fortran_order': False, "
b"'shape': (300, 300, 2), }
"""
def parse_packet(fullMsg,maxbytes,headersize,datasize,port,corr_address,testh23=False):
    #break into header and data
    HTTPheaderMsg = bytes.fromhex(fullMsg[:fullMsg.index(HEADER_DELIM)])
    NPheaderMsgHex = fullMsg[fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+2:fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2)]
    NPheaderMsg = bytes.fromhex(NPheaderMsgHex)
    #dataMsg = fullMsg[fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2):fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(maxbytes*2)]

    #decode headers
    printlog(HTTPheaderMsg,output_file=processfile)	
    HTTPheaderMsgStr = HTTPheaderMsg.decode('utf-8')
    printlog(HTTPheaderMsgStr,output_file=processfile)
    printlog(NPheaderMsg,output_file=processfile)
    NPheaderMsgStr = NPheaderMsg.decode('utf-8')
    printlog(NPheaderMsgStr,output_file=processfile)
    
    #get metadata
    #img_id_isot = HTTPheaderMsgStr[HTTPheaderMsgStr.index('IMG')+3:HTTPheaderMsgStr.index('.npy')]
    img_id_isot = HTTPheaderMsgStr[HTTPheaderMsgStr.index('IMG')+3:HTTPheaderMsgStr.index('_UV')]
    img_uv_diag = np.frombuffer(bytes.fromhex(HTTPheaderMsgStr[HTTPheaderMsgStr.index('UV')+2:HTTPheaderMsgStr.index('_DE')]))[0]
    img_dec = np.frombuffer(bytes.fromhex(HTTPheaderMsgStr[HTTPheaderMsgStr.index('DE')+2:HTTPheaderMsgStr.index('.npy')]))[0]
    img_id_mjd = Time(img_id_isot,format='isot').mjd
    #corr_address = address#HTTPheaderMsgStr[HTTPheaderMsgStr.index('Host')+6:HTTPheaderMsgStr.index(':'+str(port))]
    corr_node = corraddrs[corr_address]
    content_length = int(HTTPheaderMsgStr[HTTPheaderMsgStr.index('Content-Length')+16:HTTPheaderMsgStr.index('Expect')-2])
    shape = pipeline.get_shape_from_raw(bytes(NPheaderMsgHex,'utf-8'),headersize)#tuple(NPheaderMsgStr[NPheaderMsgStr.index('shape')+8:NPheaderMsgStr.index(')')+1])
    printlog("address:"+str(corr_address),output_file=processfile)
    printlog("corr:"+str(corr_node),output_file=processfile)
    printlog("img_id:" +str(img_id_isot),output_file=processfile)
    printlog("shape:" + str(shape),output_file=processfile)
    printlog("UVdiag:" + str(img_uv_diag),output_file=processfile)

    #use content length to get just data portion
    #data = fullMsg[fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2):fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(content_length*2)]
    #printlog(fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(content_length*2)-fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2),output_file=processfile)
    #printlog(str(data[:128]),output_file=processfile)
   
    printlog("totaldatasize: " + str(len(fullMsg)),output_file=processfile)
    printlog("without HTTP header: "  + str(len(fullMsg[fullMsg.index(HEADER_DELIM):])),output_file=processfile)
    printlog("without NP header: " + str(len(fullMsg[fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2):])),output_file=processfile) 
    data = fullMsg[fullMsg.index(NPheaderMsgHex) + len(NPheaderMsgHex):fullMsg.index(NPheaderMsgHex) + len(NPheaderMsgHex) + (2*content_length)]


    printlog("datahex: " + str(len(data)),output_file=processfile)
    imgbytes = bytes.fromhex(data)
    printlog("databytes: " + str(len(imgbytes)),output_file=processfile)
    img_data = np.frombuffer(imgbytes,dtype=dtypelookup[datasize]).reshape(shape)
    
    #***only keep this part while we test with h23***
    if testh23:
        corraddrs[corr_address] += 1
        if corraddrs[corr_address] > 15:
            corraddrs[corr_address] = 0

    return corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,img_data

fullimg_dict = dict()
def future_callback(future,SNRthresh,timestepisot,RA_axis,DEC_axis,etcd_enabled):
    """
    This function prints the result once a thread finishes processing an image
    """
    #if QSETUP and not (future.result()[1] is None):
    #    QQUEUE.put(future.result()[1])
    if etcd_enabled and not (future.result()[1] is None):
        printlog("adding " + future.result()[1] + "to etcd queue",output_file=processfile)
        ETCD.put_dict(
                    ETCDKEY,
                    {
                        "candfile":future.result()[1],
                        "uv_diag":fullimg_dict[timestepisot].img_uv_diag,
                        "dec":fullimg_dict[timestepisot].img_dec,
                        "img_shape":fullimg_dict[timestepisot].image_tesseract.shape,
                        "img_search_shape":fullimg_dict[timestepisot].image_tesseract_searched.shape
                    }
                )
    printlog(future.result()[0],output_file=processfile)
    pl.binary_plot(future.result()[0],SNRthresh,timestepisot,RA_axis,DEC_axis)
    printlog("****Thread Completed****",output_file=processfile)
    printlog(future.result(),output_file=processfile)
    printlog("************************",output_file=processfile)

    #delete from array
    del fullimg_dict[timestepisot]
    return

def main(args):
    #redirect stderr
    sys.stderr = open(error_file,"w")
    

    #if "DASKPORT" in os.environ.keys():
    #    printlog("Using Dask Scheduler on Port " + str(os.environ['DASKPORT']) + " for cand_cutter queue",output_file=processfile)
    if args.etcd:
        printlog("Etcd enabled, will push candidates to " + ETCDKEY,output_file=processfile)

    #update default values and lookup tables
    sl.SNRthresh = args.SNRthresh
    if args.gridsize != config.gridsize or args.nchans != config.nchans or args.nsamps != config.nsamps:

        config.nsamps = args.nsamps
        sl.nsamps = args.nsamps
        config.T = config.nsamps*config.tsamp
        sl.T = config.nsamps*config.tsamp
        sl.time_axis = np.linspace(0,config.T,config.nsamps)
        sl.widthtrials = sl.widthtrials[sl.widthtrials<args.nsamps]
        sl.nwidths = len(sl.widthtrials)
        sl.full_boxcar_filter = sl.gen_boxcar_filter(sl.widthtrials,args.nsamps)

        config.nchans = args.nchans
        sl.nchans = args.nchans
        config.chanbw = (config.fmax-config.fmin)/config.nchans #MHz
        sl.chanbw = (config.fmax-config.fmin)/config.nchans 
        sl.freq_axis = np.linspace(config.fmin,config.fmax,config.nchans)

        config.gridsize = args.gridsize
        sl.gridsize = args.gridsize
        sl.RA_axis = np.linspace(config.RA_point-(config.pixsize*config.gridsize/2),config.RA_point+(config.pixsize*config.gridsize/2),config.gridsize)
        sl.DEC_axis = np.linspace(config.DEC_point-(config.pixsize*config.gridsize/2),config.DEC_point+(config.pixsize*config.gridsize/2),config.gridsize)


        sl.DM_trials = np.array(sl.gen_dm(sl.minDM,sl.maxDM,config.DM_tol,config.fc*1e-3,config.nchans,config.tsamp,config.chanbw,args.nsamps))#[0:1]
        printlog(sl.DM_trials,output_file=processfile)
        sl.nDMtrials = len(sl.DM_trials)

        sl.full_boxcar_filter = sl.gen_boxcar_filter(sl.widthtrials,config.nsamps)

        sl.corr_shifts_all_append,sl.tdelays_frac_append,sl.corr_shifts_all_no_append,sl.tdelays_frac_no_append = sl.gen_dm_shifts(sl.DM_trials,sl.freq_axis,config.tsamp,config.nsamps) 

        sl.default_PSF,sl.default_PSF_params = scPSF.manage_PSF(sl.PSF_dict,args.kernelsize,sl.DEC_axis[len(sl.DEC_axis)//2],sl.default_PSF_params,sl.default_PSF,args.nsamps)
        #sl.default_PSF = scPSF.generate_PSF_images(psf_dir,np.nanmean(sl.DEC_axis),args.kernelsize//2,True,args.nsamps) #make_PSF_cube(gridsize=args.kernelsize,nsamps=args.nsamps,nchans=args.nchans)
        #sl.default_PSF_params = (args.kernelsize,"{d:.2f}".format(d=np.nanmean(sl.DEC_axis)))

        sl.current_noise = noise_update_all(None,config.gridsize,config.gridsize,sl.DM_trials,sl.widthtrials,readonly=True) #get_noise_dict(config.gridsize,config.gridsize)
        sl.tDM_max = (4.15)*np.max(sl.DM_trials)*((1/np.min(sl.freq_axis)/1e-3)**2 - (1/np.max(sl.freq_axis)/1e-3)**2) #ms
        sl.maxshift = int(np.ceil(sl.tDM_max/config.tsamp))

        printlog("UPDATED MAXSHIFT:" + str(sl.maxshift),output_file=processfile)
    if args.nocutoff:
        sl.default_cutoff = 0
    else:
        printlog(sl.DEC_axis,output_file=processfile)
        sl.default_cutoff = sl.get_RA_cutoff(np.nanmean(sl.DEC_axis),pixsize=sl.DEC_axis[1]-sl.DEC_axis[0])
        printlog("Initialized pixel cutoff to " + str(sl.default_cutoff) + " pixels",output_file=processfile)

    #write DM and width trials to file for cand cutter
    np.save(cand_dir + "DMtrials.npy",np.array(sl.DM_trials))
    np.save(cand_dir + "widthtrials.npy",np.array(sl.widthtrials))
    np.save(cand_dir + "SNRthresh.npy",sl.SNRthresh)
    np.save(cand_dir + "DMcorr_shifts.npy",sl.corr_shifts_all_no_append)
    np.save(cand_dir + "DMdelays_frac.npy",sl.tdelays_frac_no_append)

    #initialize last_frame 
    if args.initframes:
        printlog("Initializing previous frames...",output_file=processfile)
        sl.init_last_frame(args.gridsize,args.gridsize,args.nsamps-sl.maxshift,args.nchans)

    #initialize noise stats
    if args.initnoise or args.initnoisezero:
        printlog("Initializing noise statistics...",output_file=processfile)
        init_noise(sl.DM_trials,sl.widthtrials,config.gridsize,config.gridsize,zero=args.initnoisezero)
        sl.current_noise = noise_update_all(None,config.gridsize,config.gridsize,sl.DM_trials,sl.widthtrials,readonly=True)
        np.save(noise_dir + "running_vis_mean.npy",None)
        np.save(noise_dir + "running_vis_mean_burst.npy",None)

    #initialize jax functions
    if args.usejax:
        #if args.initframes: nsamps = args.nsamps*2
        #else: nsamps = args.nsamps
        #printlog("Initializing DM trial shifts...",output_file=processfile)
        #jax_funcs.init_dm_arrays(sl.DM_trials,sl.freq_axis,nsamps=nsamps,tsamp=sl.tsamp,gridsize_RA=args.gridsize//args.DMbatches,gridsize_DEC=args.gridsize//args.DMbatches)
        #printlog("TDELAYS:" + str(config.tdelays_frac),output_file=processfile)
        #printlog("CORR_LOW:" + str(config.corr_shifts_all_low),output_file=processfile)
        #printlog("CORR_HI:" + str(config.corr_shifts_all_hi),output_file=processfile)
        printlog("Initializing JIT functions...",output_file=processfile)
        if args.appendframe:
            tDM_max = (4.15)*np.max(sl.DM_trials)*((1/sl.fmin/1e-3)**2 - (1/sl.fmax/1e-3)**2) #ms
            maxshift = int(np.ceil(tDM_max/sl.tsamp))
            corr_shifts_all = sl.corr_shifts_all_append
            tdelays_frac = sl.tdelays_frac_append
        else: 
            maxshift = 0
            corr_shifts_all = sl.corr_shifts_all_no_append
            tdelays_frac = sl.tdelays_frac_no_append

        if args.DMbatches > 1:
            #subgridsize_DEC = subgridsize_RA = args.gridsize//args.DMbatches
            subgridsize_DEC = args.gridsize//args.DMbatches
            subgridsize_RA = (args.gridsize-sl.default_cutoff)//args.DMbatches
            #subband_size = args.nchans//(args.DMbatches)#*args.DMbatches)
            for i in range(args.DMbatches):
                for j in range(args.DMbatches):
                    jax_funcs.matched_filter_fft_jit(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize,args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]))


                    jax_funcs.dedisp_snr_fft_jit_0(jax.device_put(np.array(np.random.normal(size=(args.gridsize//args.DMbatches,args.gridsize//args.DMbatches,maxshift + args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),
                                               jax.device_put(corr_shifts_all,jax.devices()[0]),
                                               jax.device_put(tdelays_frac,jax.devices()[0]),
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth,i=i,j=j)
                    jax_funcs.dedisp_snr_fft_jit_0(jax.device_put(np.array(np.random.normal(size=(args.gridsize//args.DMbatches,args.gridsize//args.DMbatches,maxshift + args.nsamps,args.nchans)),dtype=np.float32),
                                               jax.devices()[1]),jax.device_put(corr_shifts_all,jax.devices()[1]),
                                               jax.device_put(tdelays_frac,jax.devices()[1]),
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=0.1,i=i,j=j)

        else:
            jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,maxshift+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),jax.device_put(corr_shifts_all,jax.devices()[0]),
                                               jax.device_put(tdelays_frac,jax.devices()[0]),
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,maxshift+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),jax.device_put(corr_shifts_all,jax.devices()[1]),
                                               jax.device_put(tdelays_frac,jax.devices()[1]),
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)



    printlog("USEFFT = " + str(args.usefft),output_file=processfile)
    #total expected number of bytes for each sub-band image
    if args.datasize%2 != 0:
        maxbytes = args.gridsize*args.gridsize*args.nsamps*(args.datasize-1) + args.headersize #really just payload size
        maxbyteshex = (args.gridsize*args.gridsize*args.nsamps*(args.datasize-1) + args.headersize + 4)*2 + 404 #http header is 404
    else:
        maxbytes = args.gridsize*args.gridsize*args.nsamps*args.datasize + args.headersize #really just payload size
        maxbyteshex = (args.gridsize*args.gridsize*args.nsamps*args.datasize + args.headersize + 4)*2 + 404 #http header is 404
    printlog("MAXBYTES: " + str(maxbytes),output_file=processfile)
    printlog("SHAPE: "  + str((args.gridsize,args.gridsize,args.nsamps,args.nchans)),output_file=processfile)
   
    #array to store image ids temporarily
    #fullimg_array = np.ndarray(shape=(args.maxProcesses),dtype=fullimg)
    #fullimg_dict = dict()

    #create socket
    printlog("creating socket...",output_file=processfile,end='')
    servSockD = socket.socket(socket.AF_INET, socket.SOCK_STREAM,0)
    printlog("Done!",output_file=processfile)    

    #bind to port number
    port = args.port
    printlog("binding to port " + str(port) + "...",output_file=processfile,end='')
    servSockD.bind(('', port))
    printlog("Done!",output_file=processfile)

    #listen for conections
    printlog("listening for connections...",output_file=processfile,end='')
    servSockD.listen(args.maxconnect)
    printlog("Made connection",output_file=processfile)
    
    #initialize a pool of processes for concurent execution
    #maxProcesses = 5
    #if "DASKPORT" in os.environ.keys() and QSETUP:
    #    executor = QCLIENT
    #else:
    executor = ThreadPoolExecutor(args.maxProcesses)
    #executor = Client(processes=False)#"10.41.0.254:8844")

    task_list = []

    while True: # want to keep accepting connections
        printlog("accepting connection...",output_file=processfile,end='')
        clientSocket,address = servSockD.accept()
        clientSocket.setblocking(0)
        corr_address, tmp = clientSocket.getpeername()
        printlog("client: " + str(corr_address) + "...",output_file=processfile,end='')
        recstatus = 1
        fullMsg = ""
        printlog("Done!",output_file=processfile)
        printlog("Receiving data...",output_file=processfile)
        
        #set timeout and expected number of bytes to read
        clientSocket.settimeout(args.timeout) 
        totalbytes = 0
        pflag = 0

        """
        #get the address size from the first chunk  of data
        try:
            (strData, ancdata, msg_flags, address) = clientSocket.recvmsg(255)
            recstatus = len(strData)
            maxbytesaddr = len(strData[:16].decode('utf-8')[:strData[:16].decode('utf-8').index('E')])
            printlog("ADDRESSS SIZE: " + str(maxbytesaddr),output_file=processfile)
            fullMsg +=strData.hex()
            totalbytes += recstatus
        except Exception as ex:
            if type(ex) == socket.timeout:
                printlog("Timed out on first read, invalid start bytes: " + str(x),output_file=processfile)
                printlog("Setting invalid start flag...",output_file=processfile,end='')
                if pipeline.set_pflag("parse_error") == None:
                    printlog("Error setting flags, abort",output_file=processfile)
                    break
                printlog("Done, continue",output_file=processfile)
                continue
            else:
                raise
        """
        #while (recstatus> 0) and (totalbytes < maxbytes):#+maxbytesaddr):
        t_timeout = time.time()
        t_startread = time.time()
        totalbyteshex =0
        while (totalbyteshex < maxbyteshex) and time.time()-t_startread<60:# and time.time()-t_timeout<args.timeout:
            try:
                #check if data is ready to read first
                t_ready = time.time()
                while not select.select([clientSocket],[],[],args.timeout) and time.time()-t_ready<args.timeout:
                    continue
                if not select.select([clientSocket],[],[],args.timeout):
                    raise socket.timeout
                printlog("Data ready",output_file=processfile)
                
                (strData, ancdata, msg_flags, address) = clientSocket.recvmsg(args.chunksize)#255)
                #printlog(strData,output_file=processfile)
                recstatus = len(strData)
                if recstatus > 0: 
                    t_timeout = time.time()
                if recstatus == 0 and time.time()-t_timeout>args.timeout:
                    raise socket.timeout

                """
                if recstatus+totalbytes > maxbytes:
                    printlog("Read " + str(recstatus) + " bytes, truncating to " + str(maxbytes-totalbytes) + ", total " + str(totalbytes+maxbytes-totalbytes),output_file=processfile)
                    #printlog("Read " + str(len(strData.hex())) + " bytes, truncating to " + str((maxbytes-totalbytes)*2) + ", total " + str(fullMsg+(strData[:maxbytes-totalbytes].hex())),output_file=processfile)
                    strData = strData[:maxbytes-totalbytes]
                    recstatus = len(strData)
                else:
                """
                printlog("Read "+ str(recstatus) + " bytes, total "+ str(totalbytes+recstatus),output_file=processfile)
                #printlog("Read "+ str(len(strData.hex())) + " bytes, total "+ str(len(fullMsg+strData.hex())),output_file=processfile)
                printlog("Message flags:" + str(msg_flags),output_file=processfile)
                printlog("AncData:" + str(ancdata),output_file=processfile)
                #if recstatus < args.chunksize:
                #    printlog("--->" + str(strData),output_file=processfile)

                #printlog(strData.hex(),output_file=processfile,end='')
                fullMsg += strData.hex()
                totalbytes += recstatus
                totalbyteshex += len(strData.hex())
                #don't know how long the header is, so don't start counting until hit NP data
                if "93" in fullMsg:
                    printlog("Found start byte at index " + str(fullMsg.index("93")),output_file=processfile)
                    totalbytes = (len(fullMsg) - fullMsg.index("93"))//2
                #if totalbytes >= maxbytes: printlog(strData,output_file=processfile)        
            except Exception as ex:
                if type(ex) == socket.timeout:
                    printlog("Timed out after reading " + str(totalbytes) + " bytes; proceeding...",output_file=processfile)
                    break
                else:
                    raise
        printlog("Done! Total bytes read:" + str(totalbytes),output_file=processfile)
        #printlog(bytes.fromhex(fullMsg[-7:-1]).decode('utf-8'),output_file=processfile)
        #printlog(bytes.fromhex(fullMsg[:1024]).decode('utf-8'),output_file=processfile)
        totalbytessend = 0
        #successmsg = bytes(success + '0\n','utf-8')
        #printlog("Sending response...",output_file=processfile,end='')
        #while (totalbytessend < len(successmsg)):
        #    totalbytessend += clientSocket.send(successmsg)        
        #printlog("Done! Total bytes sent:" + str(totalbytessend) + "/" + str(len(successmsg)),output_file=processfile)
        
        #check if data is the size we expect
        try:
            #assert(totalbytes>=maxbytes)
            assert(totalbyteshex>=maxbyteshex)
        except AssertionError as exc:
            printlog("Invalid data size, " + str(totalbytes) + " received when expected at least " + str(maxbytes) + ": " + str(exc),output_file=processfile)
            
            printlog("Invalid data size, " + str(totalbyteshex) + " received when expected at least " + str(maxbyteshex) + ": " + str(exc),output_file=processfile)
            printlog("Setting truncated data size flag...",output_file=processfile,end='')
            pflag = set_pflag_loc("datasize_error")
            if pflag == None:
                printlog("Error setting flags, abort",output_file=processfile)
                break
            printlog("Done, continue",output_file=processfile)
            #continue
        if pflag != 0:
            successmsg = bytes(success + str(pflag) + '\n','utf-8')
            printlog("Sending response...",output_file=processfile,end='')
            while (totalbytessend < len(successmsg)):
                totalbytessend += clientSocket.send(successmsg)
            printlog("Done! Total bytes sent:" + str(totalbytessend) + "/" + str(len(successmsg)),output_file=processfile)        
            continue


        #try to parse to get address
        try:
            corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData = parse_packet(fullMsg=fullMsg,maxbytes=maxbytes,headersize=args.headersize,datasize=args.datasize,port=args.port,corr_address=corr_address,testh23=args.testh23)
            #if set_pflag_loc("all",on=False) == None:
            #    printlog("Error setting flags, abort",processfile=processfile)
            #    break
        except Exception as exc:
            if type(exc) == UnicodeDecodeError: 
                printlog("Error parsing data: " + str(exc),output_file=processfile)
                printlog("Setting parse error flag...",output_file=processfile,end='')
                pflag = set_pflag_loc("parse_error")
                if pflag == None: 
                    printlog("Error setting flags, abort",processfile=processfile)
                    break
                printlog("Done, continue",output_file=processfile)
                #continue
            if type(exc) == ValueError:
                printlog("Invalid data size: " + str(exc),output_file=processfile)
                printlog("Setting datasize flag...",output_file=processfile,end='')
                pflag = set_pflag_loc("datasize_error")
                if pflag == None:
                    printlog("Error setting flags, abort",processfile=processfile)
                    break
                printlog("Done, continue",output_file=processfile)
                #continue
            else:
                clientSocket.close()
                raise exc 
        
        successmsg = bytes(success + str(pflag) + '\n','utf-8')
        printlog("Sending response...",output_file=processfile,end='')
        while (totalbytessend < len(successmsg)):
            totalbytessend += clientSocket.send(successmsg)
        printlog("Done! Total bytes sent:" + str(totalbytessend) + "/" + str(len(successmsg)),output_file=processfile)
        if pflag != 0:
            continue
        printlog("Data: " + str(arrData),output_file=processfile)

        #if object is in dict
        if img_id_isot not in fullimg_dict.keys():
            fullimg_dict[img_id_isot] = fullimg(img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape=tuple(np.concatenate([shape,[args.nchans]])))

        """
        #if object corresponding to the image is in list
        idx,openidx = find_id(img_id_isot,fullimg_array)
        printlog("FIND_ID: " + str(idx) + ", " + str(openidx),output_file=processfile)#if it's not in the list, but there's an open spot, add it
        if idx == -1 and openidx != -1:
            #need to create new object
            fullimg_array[openidx] = fullimg(img_id_isot,img_id_mjd,shape=tuple(np.concatenate([shape,[args.nchans]])))
            idx = openidx
        elif idx == -1 and openidx == -1: # shouldn't reach this case often, but if we don't have space for a new object, busy wait
            while openidx == -1: 
                printlog("Process server image array full, waiting for opening...",output_file=processfile,end='')
                idx,openidx = find_id(img_id_isot,fullimg_array)
        #otherwise, just add to the image at idx
        """ 	
        #add image and update flags
        fullimg_dict[img_id_isot].add_corr_img(arrData,corr_node,args.testh23) #fullimg_array[idx].add_corr_img(arrData,corr_node,args.testh23)
        #if the image is complete, start the search
        printlog("corrstatus:",output_file=processfile,end='')
        printlog(fullimg_dict[img_id_isot].corrstatus,output_file=processfile)
        if fullimg_dict[img_id_isot].is_full(): #fullimg_array[idx].is_full():
            #submit a search task to the process pool
            printlog("Submitting new task for image " + str(img_id_isot),output_file=processfile)
            RA_axis_idx = copy.deepcopy(fullimg_dict[img_id_isot].RA_axis) #copy.deepcopy(fullimg_array[idx].RA_axis)
            DEC_axis_idx= copy.deepcopy(fullimg_dict[img_id_isot].DEC_axis) #copy.deepcopy(fullimg_array[idx].DEC_axis)

            #update noise from file if offline
            if args.offline:
                sl.last_frame = sl.get_last_frame()
            
            """
            if "DASKPORT" in os.environ.keys() and QSETUP:
                task_list.append(executor.submit(sl.search_task,fullimg_dict[img_id_isot],args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,args.PyTorchDedispersion,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.appendframe,args.DMbatches,args.SNRbatches,args.usejax,QSETUP,workers=QWORKERS))
                fire_and_forget(task_list[-1])
            
            else:   
            """
            #initialize noise stats
            if args.fprtest or args.fnrtest:
                printlog("FPR Test, Re-Initializing noise statistics...",output_file=processfile)
                init_noise(sl.DM_trials,sl.widthtrials,config.gridsize,config.gridsize)
                sl.current_noise = noise_update_all(None,config.gridsize,config.gridsize,sl.DM_trials,sl.widthtrials,readonly=True)



            task_list.append(executor.submit(sl.search_task,fullimg_dict[img_id_isot],args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,args.PyTorchDedispersion,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.fprtest,args.fnrtest,args.appendframe,args.DMbatches,args.SNRbatches,args.usejax,args.noiseth,args.nocutoff,args.realtime))
            
            #printlog(future.result(),output_file=processfile)
            task_list[-1].add_done_callback(lambda future: future_callback(future,args.SNRthresh,img_id_isot,RA_axis_idx,DEC_axis_idx,args.etcd))
            #after finishes execution, remove from list by setting element to None
            #fullimg_array[idx] = None
    

        

        #sys.stdout.flush()
    executor.shutdown()
    clientSocket.close()



if __name__=="__main__":
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 10',default=10)
    parser.add_argument('--port',type=int,help='Port number for receiving data from subclient, default = 8080',default=8080)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default=301',default=301)
    parser.add_argument('--nsamps',type=int,help='Expected number of time samples (integrations) for each sub-band image, default=25',default=25)
    parser.add_argument('--nchans',type=int,help='Expected number of sub-band images for each full image, default=16',default=16)
    parser.add_argument('--datasize',type=int,help='Expected size of each element in sub-band image in bytes,default=8',default=8,choices=list(dtypelookup.keys()))
    parser.add_argument('--chunksize',type=int,help='Number of bytes to read from client at a time, default=18874368 (for data size ~18 MB)',default=18874368)
    parser.add_argument('--subimgpix',type=int,help='Length of image cutouts in pixels, default=11',default=11)
    parser.add_argument('-T','--testh23',action='store_true')
    parser.add_argument('--maxconnect',type=int,help='Maximum number of connections accepted by the server, default=16',default=16)
    parser.add_argument('--timeout',type=float,help='Max time in seconds to wait for more data to be ready to receive, default = 1',default=1)

    #arguments for classifier from classifier.py
    #parser.add_argument('--npy_file', type=str, required=True, help='Path to the NumPy file containing the images')
    parser.add_argument('--model_weights', type=str, help='Path to the model weights file',default=cwd + "/simulations_and_classifications/model_weights.pth")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of images that can be searched at once, default = 5, maximum is 40',default=5)
    parser.add_argument('--headersize',type=int,help='Number of bytes representing the header; note this varies depending on the data shape, default = 128',default=128)
    parser.add_argument('--spacefilter',action='store_true', help='Use PSF to spatial matched filter the input image')
    parser.add_argument('--kernelsize',type=int,help='Kernel size for PSF spatial matched filter; default=151',default=151)
    parser.add_argument('--usefft',action='store_true', help='Implement PSF spatial matched filter as a 2D FFT')
    parser.add_argument('--cluster',action='store_true',help='Enable clustering with HDBSCAN')
    parser.add_argument('--multithreading',action='store_true',help='Enable multithreading in search')
    parser.add_argument('--nrows',type=int,help='Number of rows to break image into if multithreading, default = 4',default=4)
    parser.add_argument('--ncols',type=int,help='Number of columns to break image into if multithreading, default = 2',default=2)
    parser.add_argument('--threadDM',action='store_true',help='Break DM trials among multiple threads')
    parser.add_argument('--samenoise',action='store_true',help='Assume the noise in each pixel is the same')
    parser.add_argument('--cuda',action='store_true',help='Uses PyTorch to accelerate computation with GPUs. The cuda flag overrides the multithreading option')
    parser.add_argument('--toslack',action='store_true',help='Sends Candidate Summary Plots to Slack')
    parser.add_argument('--PyTorchDedispersion',action='store_true',help='Uses GPU-accelerated dedispersion code from https://github.com/nkosogor/PyTorchDedispersion')
    parser.add_argument('--exportmaps',action='store_true',help='Output noise maps for each DM and width trial to the noise directory')
    parser.add_argument('--initframes',action='store_true',help='Initializes previous frames for dedispersion')
    parser.add_argument('--initnoise',action='store_true',help='Initializes noise statistics from fast vis data for S/N estimates')
    parser.add_argument('--initnoisezero',action='store_true',help='Initializes noise to 0')
    parser.add_argument('--savesearch',action='store_true',help='Saves the searched image as a numpy array')
    parser.add_argument('--fprtest',action='store_true',help='Saves only searched data and writes peak SNR to file')
    parser.add_argument('--fnrtest',action='store_true',help='Saves only searched data and writes peak SNR to file')
    parser.add_argument('--appendframe',action='store_true',help='Use the previous image to fill in dedispersion search')
    parser.add_argument('--DMbatches',type=int,help='Number of pixel batches to submit dedispersion to the GPUs with, default = 1',default=1)
    parser.add_argument('--SNRbatches',type=int,help='Number of pixel batches to submit boxcar filtering to the GPUs with, default = 1',default=1)
    parser.add_argument('--usejax',action='store_true',help='Use JAX Just-In-Time compilation for GPU acceleration')
    parser.add_argument('--offline',action='store_true',help='Run system offline, relaxes realtime requirement and can update noise from injections')
    parser.add_argument('--etcd',action='store_true',help='Enable etcd reading/writing of candidates')
    parser.add_argument('--noiseth',type=float,help='S/N threshold below which samples are included in noise calculation; default=3',default=3)#Quantile threshold below which samples are included in noise calculation; default=0.1',default=0.1)
    parser.add_argument('--nocutoff',action='store_true',help='If set, ignores offset between successive time batches (3.25 seconds)')
    parser.add_argument('--realtime',action='store_true',help='Running in realtime system, puts image data in PSRDADA buffer')
    args = parser.parse_args()


    
    main(args)
