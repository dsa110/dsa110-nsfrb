import numpy as np
from queue import Empty as Exception_Empty
import gc
import tracemalloc
import glob
import json
from multiprocessing import Manager,Queue
from nsfrb.planning import get_RA_cutoff
from threading import Lock, Timer
from dask.distributed import Lock as Lock_DASK
from dask.distributed import Client,get_client
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
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait

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
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,Lon,Lat,tsamp_slow,bin_slow,pixperFWHM,output_file,bin_imgdiff,sslogfile,table_dir,procserver_memory_file

"""
NSFRB modules
"""
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack 
from nsfrb.imaging import uv_to_pix,stack_images

"""
Dask manager
"""
import dsautils.dsa_store as ds
ETCD = ds.DsaStore()
ETCDKEY = f'/mon/nsfrb/candidates'
ETCDKEY_SEARCHTIMING = f'/mon/nsfrbsearchtiming'
ETCDKEY_PACKET = f'/mon/nsfrbpackets'

from nsfrb import searching as sl
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
from nsfrb.config import NROWSUBIMG,NSUBIMG,SUBIMGPIX,SUBIMGORDER
"""
NROWSUBIMG = 7
NSUBIMG = NROWSUBIMG*NROWSUBIMG
SUBIMGPIX=43
SUBIMGORDER = []
for i in range(NROWSUBIMG):
    for j in range(NROWSUBIMG):
        SUBIMGORDER.append((i,j))
"""
#QQUEUE_SRCH= Queue()
#QQUEUE_CALLBACK=Queue()
QQUEUE_UDP = Queue()
timeout_isots = [] #keep track of which ones have timed out so we can ignore any additional data
fullimg_dict = dict()
slow_fullimg_dict =dict()
imgdiff_fullimg_dict=dict()
"""
def timeout_handler(img_id_isot,queue=QQUEUE_UDP,fullimg_dict=fullimg_dict):        
    try:
        printlog("FROM TIMEOUT:"+str(fullimg_dict.keys()) + str(img_id_isot),output_file=processfile)
        fullimg = fullimg_dict[img_id_isot]
        printlog("Image "+fullimg.img_id_isot+" timed out, searching now",output_file=processfile)
        #timeout_isots.append(fullimg.img_id_isot)
        fullimg.nanfill()
        queue.put_nowait((-1,
                fullimg.img_id_isot,
                fullimg.img_id_mjd,
                fullimg.img_uv_diag,
                fullimg.img_dec,
                fullimg.shape[:-1],
                None,
                -99))
        printlog("["+fullimg.img_id_isot+"]EXITING TIMEOUT HANDLER|"+str(queue.qsize()),output_file=processfile)
    except Exception as exc:
        printlog("[EXCEPTION FROM TIMEOUT_HANDLER]:"+str(exc),output_file=processfile)
    return
"""

class fullimg:
    def __init__(self,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape=(32,32,25,16),dtype=np.float16,slow=False,bin_slow=bin_slow,imgdiff=False,bin_imgdiff=bin_imgdiff,TXsubimg=False,TXsubint=False,TXnints=1):
        printlog("INIT FULLIMG WITH SHAPE:"+str(shape),output_file=processfile)
        if not (slow or imgdiff):
            try:
                self.timer = Timer(config.tsamp*shape[2]/1000, lambda: self.timeout_handler())
                self.timer.start()
                self.timerlock=Lock()
                printlog("ACTIVATED TIMER",output_file=processfile)
            except Exception as exc:
                printlog("FAILED TO ACTIVATE TIMER:"+str(exc),output_file=processfile)
        self.queuetime = time.time()
        self.running = False
        self.image_tesseract = np.zeros(shape,dtype=dtype)
        self.corrstatus = np.zeros(16,dtype=int)
        self.thash=hex(random.getrandbits(32))
        """
        if TXsubimg:
            self.corrstatus -= (NSUBIMG-1)
            self.shape = tuple([shape[0]*NROWSUBIMG,shape[1]*NCOLSUBIMG]+list(shape[2:]))
        elif TXsubint and TXnints>1:
            self.corrstatus -= (TXnints-1)
            self.shape = tuple(list(shape[:2])+[shape[2]*TXnints])
        else:
        """
        self.shape = shape
        self.img_id_isot = img_id_isot
        self.img_id_mjd = img_id_mjd
        self.img_uv_diag =img_uv_diag
        #self.shape = shape
        self.img_dec = img_dec
        #get ra and dec axes
        self.RA_axis,self.DEC_axis,tmp = uv_to_pix(self.img_id_mjd,self.shape[0],Lat=Lat,Lon=Lon,uv_diag=img_uv_diag,DEC=img_dec,pixperFWHM=pixperFWHM)
        self.slow = slow
        self.imgdiff=imgdiff
        self.slow_counter=0
        self.bin_slow=  bin_slow
        self.bin_interval_slow = int(self.shape[2])//self.bin_slow
        self.imgdiffgulps = self.shape[2]
        self.bin_imgdiff = bin_imgdiff
        if slow:
            printlog("bin slow:" + str(self.bin_slow) + "; bin interval:" + str(self.bin_interval_slow),output_file=processfile)
            """
            self.RA_cutoff = sl.get_RA_cutoff(img_dec,pixsize=np.abs(self.RA_axis[1]-self.RA_axis[0]))
            self.slow_RA_cutoff = (self.bin_slow-1)*self.RA_cutoff
            self.shape = (self.shape[0],self.shape[1]-self.slow_RA_cutoff,self.shape[2],self.shape[3])
            self.image_tesseract = np.zeros((shape[0],shape[1]-self.slow_RA_cutoff,shape[2],shape[3]),dtype=dtype)
            self.RA_axis = self.RA_axis[self.slow_RA_cutoff:]
            """
            #make list of possible mjds
            self.img_id_mjd_list = []
            self.slow_RA_cutoffs = []
            for i in range(self.bin_slow):
                self.img_id_mjd_list.append(self.img_id_mjd + (config.tsamp*self.shape[2]*i/1000/86400))
                self.slow_RA_cutoffs.append(get_RA_cutoff(self.img_dec,usefit=True,offset_s=config.tsamp*self.shape[2]*i/1000))
            self.img_id_mjd_list = np.array(self.img_id_mjd_list,dtype=np.float64)
            self.slowstatus = np.zeros(len(self.img_id_mjd_list),dtype=int)
        elif imgdiff:
            printlog("starting object for image difference mode with " + str(self.imgdiffgulps) + " gulps at a time",output_file=processfile)
            #make list of possible mjds
            self.img_id_mjd_list = []
            self.slow_RA_cutoffs = []
            for i in range(self.imgdiffgulps):
                for j in range(self.bin_imgdiff):
                    self.img_id_mjd_list.append(self.img_id_mjd + (config.tsamp*config.nsamps*(self.bin_imgdiff*i + j)/1000/86400))
                self.slow_RA_cutoffs.append(get_RA_cutoff(self.img_dec,usefit=True,offset_s=j*config.tsamp*config.nsamps*i/1000))
            self.img_id_mjd_list = np.array(self.img_id_mjd_list,dtype=np.float64)
            self.imgdiffstatus = np.zeros(len(self.img_id_mjd_list),dtype=int)
        printlog("Created RA and DEC axes of size" + str(self.RA_axis.shape) + "," + str(self.DEC_axis.shape),output_file=processfile)
        printlog(self.RA_axis,output_file=processfile)
        printlog(self.DEC_axis,output_file=processfile)
    
    def timeout_handler(self):
        self.timerlock.acquire()
        if not self.running:
            printlog("Image "+self.img_id_isot+" timed out, searching now",output_file=processfile)
            timeout_isots.append(self.img_id_isot)
            self.nanfill()
            QQUEUE_UDP.put_nowait(("mport",-1,self.img_id_isot,self.img_id_mjd,self.img_uv_diag,self.img_dec,self.shape[:-1],None,-99))
            printlog("["+self.img_id_isot+"]EXITING TIMEOUT HANDLER",output_file=processfile)
        else:
            printlog("Already running, ignore timeout",output_file=processfile)
        self.timerlock.release()
        return
    
    def add_corr_img(self,data,corr_node,testmode=False,TXsubimg=False,TXsubint=False,TXnints=1):
        """
        if TXsubimg:
            sidx = NSUBIMG-1+self.corrstatus[corr_node]
            printlog("SUBIMG ROW SLICES:" + str((SUBIMGPIX*SUBIMGORDER[sidx][0],SUBIMGPIX*(SUBIMGORDER[sidx][0]+1))),output_file=processfile)
            printlog("SUBIMG COL SLICES:" + str((SUBIMGPIX*SUBIMGORDER[sidx][1],SUBIMGPIX*(SUBIMGORDER[sidx][1]+1))),output_file=processfile)
            self.image_tesseract[SUBIMGPIX*SUBIMGORDER[sidx][0]:SUBIMGPIX*(SUBIMGORDER[sidx][0]+1),
                                 SUBIMGPIX*SUBIMGORDER[sidx][1]:SUBIMGPIX*(SUBIMGORDER[sidx][1]+1),:,corr_node] = data
        elif TXsubint and TXnints>1:
            sidx = TXnints-1+self.corrstatus[corr_node]
            printlog("SUBINT SLICES:"+str((sidx*(self.shape[2]//TXnints),(sidx+1)*(self.shape[2]//TXnints))),output_file=processfile)
            self.image_tesseract[:,:,int(sidx*(self.shape[2]//TXnints)):int((sidx+1)*(self.shape[2]//TXnints)),corr_node] = data
        else:
        """
        self.image_tesseract[:,:,:,corr_node] = data
        #if testmode:
        self.corrstatus[corr_node] += 1
        printlog("CORR NODE STATUS:"+str(self.corrstatus[corr_node]),output_file=processfile)
        return
    def slow_mjd_in_img(self,mjd):
        printlog("CHECKING IF " + str(self.img_id_mjd_list) + " CONTAINS " + str(mjd),output_file=processfile)

        img_idx = np.argmin(np.abs(self.img_id_mjd_list - mjd))
        foundmjd = np.abs(self.img_id_mjd_list[img_idx]-mjd)*86400*1000 <= (config.tsamp*self.shape[2]/2)
        printlog("IT DOES" + str("" if foundmjd else " NOT"),output_file=processfile)
        return (img_idx if foundmjd else -1),foundmjd
    def imgdiff_mjd_in_img(self,mjd):
        printlog("CHECKING IF " + str(self.img_id_mjd_list) + " CONTAINS " + str(mjd),output_file=processfile)

        img_idx = np.argmin(np.abs(self.img_id_mjd_list - mjd))
        foundmjd = np.abs(self.img_id_mjd_list[img_idx]-mjd)*86400*1000 <= (config.tsamp*config.nsamps/2)
        printlog("IT DOES" + str("" if foundmjd else " NOT"),output_file=processfile)
        return (img_idx if foundmjd else -1),foundmjd

    def slow_append_img(self,data,img_idx):
        self.image_tesseract[:,:,img_idx*self.bin_interval_slow:(img_idx+1)*self.bin_interval_slow,:] = np.nanmean(data.reshape((self.shape[0],self.shape[1],self.bin_interval_slow,self.bin_slow,self.shape[3])),3)
        """
        self.image_tesseract[:,:,img_idx*self.bin_interval_slow:(img_idx+1)*self.bin_interval_slow,:] = np.nanmean(data[:,self.shape[0]-self.shape[1]-self.RA_cutoff*(self.bin_slow-1-img_idx):self.shape[0]-self.RA_cutoff*(self.bin_slow-1-img_idx),:,:].reshape((self.shape[0],self.shape[1],self.bin_interval_slow,self.bin_slow,self.shape[3])),3)
        """
        printlog("FROM SLOW APPEND_1 " + str(img_idx) + ":" + str(self.image_tesseract[:,:,img_idx*self.bin_interval_slow:(img_idx+1)*self.bin_interval_slow,:]),output_file=processfile)
        self.image_tesseract[:,:,img_idx*self.bin_interval_slow:(img_idx+1)*self.bin_interval_slow,:] -= np.nanmedian(self.image_tesseract[:,:,img_idx*self.bin_interval_slow:(img_idx+1)*self.bin_interval_slow,:],axis=2,keepdims=True)
        printlog("FROM SLOW APPEND_2 " + str(img_idx)  + ":" + str(self.image_tesseract[:,:,img_idx*self.bin_interval_slow:(img_idx+1)*self.bin_interval_slow,:]),output_file=processfile)
        self.slowstatus[img_idx] = 1
        printlog("MJD LIST:" + str(self.img_id_mjd_list),output_file=processfile)
        printlog("SLOW STATUS:" + str(self.slowstatus),output_file=processfile)
        self.slow_counter += 1
        #align if full
        if self.slow_is_full():
            stack,tmp,tmp,min_gridsize = stack_images([self.image_tesseract[:,:,i*self.bin_interval_slow:(i+1)*self.bin_interval_slow,:] for i in range(self.bin_slow)],self.slow_RA_cutoffs)
            self.RA_axis = self.RA_axis[:min_gridsize]
            self.slow_RA_cutoff = self.shape[1] - min_gridsize
            self.image_tesseract = np.concatenate(stack,axis=2)
            self.shape = self.image_tesseract.shape
            printlog("Stacked slow images:" + str(self.image_tesseract.shape),output_file=processfile)
        return
    def imgdiff_append_img(self,data,img_idx):
        self.image_tesseract[:,:,int(img_idx//self.bin_imgdiff),0] += np.nanmean(data,(2,3)) 
        printlog("FROM IMGDIFF APPEND_1 " + str(img_idx) + ":" + str(self.image_tesseract[:,:,img_idx:(img_idx+1),:]),output_file=processfile)
        self.image_tesseract[:,:,int(img_idx//self.bin_imgdiff),0] -= np.nanmedian(np.nanmean(data,3),axis=2)
        printlog("FROM IMGDIFF APPEND_2 " + str(img_idx)  + ":" + str(self.image_tesseract[:,:,img_idx:(img_idx+1),:]),output_file=processfile)
        self.imgdiffstatus[img_idx] = 1
        printlog("MJD LIST:" + str(self.img_id_mjd_list),output_file=processfile)
        printlog("IMGDIFF STATUS:" + str(self.imgdiffstatus),output_file=processfile)
        self.slow_counter += 1
        #self.imgdiffstatus[:] = 1
        #align if full
        if self.imgdiff_is_full():
            printlog("STACKING IMAGES",output_file=processfile)
            stack,tmp,tmp,min_gridsize = stack_images([self.image_tesseract[:,:,i:(i+1),:] for i in range(self.imgdiffgulps)],self.slow_RA_cutoffs)
            self.RA_axis = self.RA_axis[:min_gridsize]
            self.imgdiff_RA_cutoff = self.shape[1] - min_gridsize
            self.image_tesseract = np.concatenate(stack,axis=2)
            self.shape = self.image_tesseract.shape
            printlog("Stacked imgdiff images:" + str(self.image_tesseract.shape),output_file=processfile)
        return

    def slow_is_full(self):
        return np.all(self.slowstatus==1)#(self.slow_counter)*self.bin_interval_slow >= self.shape[2]
    def imgdiff_is_full(self):
        return np.all(self.imgdiffstatus==1)
    def is_full(self):
        return np.all(self.corrstatus==1)
    def is_timeout(self):
        printlog("CHECKING TIMEOUT:"+str(time.time()-self.queuetime),output_file=processfile)
        if self.slow:
            printlog("&SLOW",output_file=processfile)
            return (time.time()-self.queuetime) >= 2*(config.tsamp_slow*self.shape[2]/1000)
        elif self.imgdiff:
            printlog("&IMGDIFF",output_file=processfile)
            return (time.time()-self.queuetime) >= 2*(config.tsamp_imgdiff*self.shape[2]/1000)
        return (time.time()-self.queuetime) >= (config.tsamp*self.shape[2]/1000)
    def nanfill(self):
        """
        if data has timed out, fill missing data with nan and mark full
        """
        printlog("FILLING NAN"+str(self.image_tesseract.shape)+str(self.corrstatus)+str(type(self.corrstatus)),output_file=processfile)
        if self.slow:
            printlog("&SLOW",output_file=processfile)
            self.image_tesseract[:,:,np.array(self.slowstatus)<1,:] = np.nan
            self.slowstatus=np.ones_like(self.slowstatus)
        elif self.imgdiff:
            printlog("&IMGDIFF",output_file=processfile)
            self.image_tesseract[:,:,np.array(self.imgdiffstatus)<1,:]=np.nan
            self.imgdiffstatus=np.ones_like(self.imgdiffstatus)
        else:
            self.image_tesseract[:,:,:,np.array(self.corrstatus)<1]=np.nan
            self.corrstatus = np.ones_like(self.corrstatus)
        printlog("DONE FILLING NAN",output_file=processfile)

        return

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
import json
f = open(table_dir + "corripaddrs.json","r")
corraddrs = json.load(f)
f.close()
print(corraddrs)

testport_corrs = {8080:0,
                  8810:0,
                  8826:0,
                  8842:0,
                  8858:0,
                  8876:0,
                  8811:1,
                  8812:2,
                  8813:3,
                  8814:4,
                  8815:5,
                  8816:6,
                  8817:7,
                  8818:8,
                  8819:9,
                  8820:10,
                  8821:11,
                  8822:12,
                  8823:13,
                  8824:14,
                  8825:15}

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
    if testh23:
        corr_node = testport_corrs[port]
        if port == 8080:
            testport_corrs[port] +=1
            if testport_corrs[port] > 15:
                testport_corrs[port] = 0
    else:
        corr_node = corraddrs[corr_address]

    content_length = int(HTTPheaderMsgStr[HTTPheaderMsgStr.index('Content-Length')+16:HTTPheaderMsgStr.index('Expect')-2])
    shape = pipeline.get_shape_from_raw(bytes(NPheaderMsgHex,'utf-8'),headersize)[:3]#tuple(NPheaderMsgStr[NPheaderMsgStr.index('shape')+8:NPheaderMsgStr.index(')')+1])
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
    """
    if testh23:
        corraddrs[corr_address] += 1
        if corraddrs[corr_address] > 15:
            corraddrs[corr_address] = 0
    """
    return corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,img_data

def future_callback(future,SNRthresh,etcd_enabled,dask_enabled):
    """
    This function prints the result once a thread finishes processing an image
    """
    #if QSETUP and not (future.result()[1] is None):
    #    QQUEUE.put(future.result()[1])
    if dask_enabled:
        fresult = future
    else:
        fresult = future.result()
    slow = fresult[1]
    imgdiff = fresult[2]
    timestepisot = fresult[5]
    thash = fresult[6]
    printlog("|CALLBACK| "+str("[SLOW]" if slow else "")+str("[IMGDIFF]" if imgdiff else "")+" deleting "+timestepisot,output_file=processfile)
    if etcd_enabled and not (fresult[0] is None):
        printlog("adding " + fresult[0] + "to etcd queue",output_file=processfile)
        if slow:
            ETCD.put_dict(
                    ETCDKEY,
                    {
                        "candfile":fresult[0],
                        "uv_diag":slow_fullimg_dict[timestepisot].img_uv_diag,
                        "dec":slow_fullimg_dict[timestepisot].img_dec,
                        "img_shape":slow_fullimg_dict[timestepisot].image_tesseract.shape,
                        "img_search_shape":slow_fullimg_dict[timestepisot].image_tesseract_searched.shape
                    }
                )
        elif imgdiff:
            ETCD.put_dict(
                    ETCDKEY,
                    {
                        "candfile":fresult[0],
                        "uv_diag":imgdiff_fullimg_dict[timestepisot].img_uv_diag,
                        "dec":imgdiff_fullimg_dict[timestepisot].img_dec,
                        "img_shape":imgdiff_fullimg_dict[timestepisot].image_tesseract.shape,
                        "img_search_shape":imgdiff_fullimg_dict[timestepisot].image_tesseract_searched.shape
                    }
                )
        else:
            ETCD.put_dict(
                    ETCDKEY,
                    {
                        "candfile":fresult[0],
                        "uv_diag":fullimg_dict[timestepisot].img_uv_diag,
                        "dec":fullimg_dict[timestepisot].img_dec,
                        "img_shape":fullimg_dict[timestepisot].image_tesseract.shape,
                        "img_search_shape":fullimg_dict[timestepisot].image_tesseract_searched.shape
                    }
                )
    #timing_dict = ETCD.get_dict(ETCDKEY_SEARCHTIMING)
    #if timing_dict is None: timing_dict = dict()
    timing_dict = dict()
    timing_dict["search_time"]=fresult[-2]
    timing_dict["search_tx_time"]=fresult[-1]
    #timing_dict["search_completed"]=True
    ETCD.put_dict(ETCDKEY_SEARCHTIMING,timing_dict)

    #printlog(future.result()[0],output_file=processfile)
    #pl.binary_plot(fullimg_dict[timestepisot].image_tesseract_searched,SNRthresh,timestepisot,RA_axis,DEC_axis)
    printlog("****Thread Completed****",output_file=processfile)
    printlog(fresult,output_file=processfile)
    printlog("************************",output_file=processfile)

    #delete from array
    if slow:
        printlog(("***>[SLOW]",len(slow_fullimg_dict.keys()),timestepisot,timestepisot in slow_fullimg_dict.keys()))
        slow_fullimg_dict[timestepisot] = None
        del slow_fullimg_dict[timestepisot]
        printlog(("***>[SLOW]",len(slow_fullimg_dict.keys()),timestepisot,timestepisot in slow_fullimg_dict.keys()))
    elif imgdiff:
        printlog(("***>[IMGDIFF]",len(imgdiff_fullimg_dict.keys()),timestepisot,timestepisot in imgdiff_fullimg_dict.keys()))
        imgdiff_fullimg_dict[timestepisot] = None
        del imgdiff_fullimg_dict[timestepisot]
        printlog(("***>[IMGDIFF]",len(imgdiff_fullimg_dict.keys()),timestepisot,timestepisot in imgdiff_fullimg_dict.keys()))
    else:
        printlog(("***>[BASE]",len(fullimg_dict.keys()),timestepisot,timestepisot in fullimg_dict.keys()))
        fullimg_dict[timestepisot] = None
        del fullimg_dict[timestepisot]
        printlog(("***>[BASE]",len(fullimg_dict.keys()),timestepisot,timestepisot in fullimg_dict.keys()))
    f = open(sslogfile,"a")
    f.write("[stop] [" + thash +"] " + str(time.time()))
    f.close()
    printlog("***GARBAGE COLLECTION>>"+str(gc.collect()),output_file=processfile)
    return

def future_callback_attach(future,SNRthresh,timestepisot,RA_axis,timestepisot_slow,RA_axis_slow,timestepisot_imgdiff,RA_axis_imgdiff,DEC_axis,etcd_enabled,dask_enabled,thash):
    """
    This function prints the result once a thread finishes processing an image
    """
    #if QSETUP and not (future.result()[1] is None):
    #    QQUEUE.put(future.result()[1])
    if dask_enabled:
        fresult = future
    else:
        fresult = future.result()
    
    for c in range(len(fresult[0])):
        slow = fresult[1][c]
        imgdiff = fresult[2][c]
        if etcd_enabled:
            printlog("adding " + fresult[0][c] + "to etcd queue",output_file=processfile)
            if slow:
                ETCD.put_dict(
                    ETCDKEY,
                    {
                        "candfile":fresult[0][c],
                        "uv_diag":slow_fullimg_dict[timestepisot_slow].img_uv_diag,
                        "dec":slow_fullimg_dict[timestepisot_slow].img_dec,
                        "img_shape":slow_fullimg_dict[timestepisot_slow].image_tesseract.shape,
                        "img_search_shape":slow_fullimg_dict[timestepisot_slow].image_tesseract_searched.shape
                    }
                )
            elif imgdiff:
                ETCD.put_dict(
                    ETCDKEY,
                    {
                        "candfile":fresult[0][c],
                        "uv_diag":imgdiff_fullimg_dict[timestepisot_imgdiff].img_uv_diag,
                        "dec":imgdiff_fullimg_dict[timestepisot_imgdiff].img_dec,
                        "img_shape":imgdiff_fullimg_dict[timestepisot_imgdiff].image_tesseract.shape,
                        "img_search_shape":imgdiff_fullimg_dict[timestepisot_imgdiff].image_tesseract_searched.shape
                    }
                )
            else:
                ETCD.put_dict(
                    ETCDKEY,
                    {
                        "candfile":fresult[0][c],
                        "uv_diag":fullimg_dict[timestepisot].img_uv_diag,
                        "dec":fullimg_dict[timestepisot].img_dec,
                        "img_shape":fullimg_dict[timestepisot].image_tesseract.shape,
                        "img_search_shape":fullimg_dict[timestepisot].image_tesseract_searched.shape
                    }
                )
    #timing_dict = ETCD.get_dict(ETCDKEY_SEARCHTIMING)
    #if timing_dict is None: timing_dict = dict()
    timing_dict = dict()
    timing_dict["search_time"]=fresult[-2]
    timing_dict["search_tx_time"]=fresult[-1]
    #timing_dict["search_completed"]=True
    ETCD.put_dict(ETCDKEY_SEARCHTIMING,timing_dict)

    #printlog(future.result()[0],output_file=processfile)
    #pl.binary_plot(fullimg_dict[timestepisot].image_tesseract_searched,SNRthresh,timestepisot,RA_axis,DEC_axis)
    printlog("****Thread Completed****",output_file=processfile)
    printlog(fresult,output_file=processfile)
    printlog("************************",output_file=processfile)

    #delete from array
    if timestepisot_slow is not None:
        del slow_fullimg_dict[timestepisot_slow]
    if timestepisot_imgdiff is not None:
        del imgdiff_fullimg_dict[timestepisot_imgdiff]
    del fullimg_dict[timestepisot]
    f = open(sslogfile,"a")
    f.write("[stop] [" + thash +"] " + str(time.time()))
    f.close()
    printlog("***GARBAGE COLLECTION>>"+str(gc.collect()),output_file=processfile)
    return

def udpreadtask(servSockD,ii,port,maxbytes,maxbyteshex,timeout,chunksize,headersize,datasize,testh23,offline,protocol,udpchunksize,substage,TXsubimg,TXsubint,TXnints,portmapping,dask_enabled,sublock,queue=QQUEUE_UDP):
    try:
        if dask_enabled:
            executor = get_client()
            sublock = Lock_DASK("sublock",client=executor)
        socksuffix = "SOCKET " + str(ii) + " >"
        printlog(socksuffix + "Initialized reader " + str(ii) + " on port "+str(port),output_file=processfile)
        while True:
            printlog(socksuffix + "QUEUE SIZE: "+str(QQUEUE_UDP.qsize())+"|"+str(QQUEUE_UDP.full()),output_file=processfile)

            t1 = time.time()
            readsockets=[]
            while len(readsockets)==0:
                readsockets,writesockets,errsockets = select.select([servSockD],[],[],timeout)

            printlog(socksuffix+"start reading>>[TIME]"+str(time.time()),output_file=processfile)
            printlog(socksuffix+"TIME TO WAIT FOR DATA:"+str(time.time()-t1)+"s",output_file=processfile)
            t1 = time.time()
            corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port = readcorrdata(servSockD,ii,servSockD.getsockname()[1],maxbytes,
                                                                                                maxbyteshex,timeout,chunksize,headersize,datasize,testh23,
                                                                                                offline,protocol,udpchunksize)
            corr_node=int(corr_node)
            printlog(socksuffix+"TIME TO READ DATA:"+str(time.time()-t1)+"s",output_file=processfile)
            if TXsubimg or TXsubint:
                sublock.acquire()
                t1 = time.time()
                if img_id_isot not in substage[corr_node].keys():
                    if TXsubimg:
                        substage[corr_node][img_id_isot] = dict()
                        substage[corr_node][img_id_isot]["data"]=np.zeros_like(arrData).repeat(NROWSUBIMG,axis=0).repeat(NCOLSUBIMG,axis=1)
                        substage[corr_node][img_id_isot]["count"]=0
                    elif TXsubint:
                        substage[corr_node][img_id_isot] = dict()
                        substage[corr_node][img_id_isot]["data"]=None #np.zeros_like(arrData).repeat(TXnints,axis=2)
                        substage[corr_node][img_id_isot]["count"]=0
                printlog(socksuffix+"Is "+img_id_isot+"in subdict? "+str(img_id_isot in substage[corr_node].keys()),output_file=processfile)
                printlog(socksuffix+"Adding to subdict:"+str((img_id_isot,corr_node,substage[corr_node][img_id_isot]["count"])),output_file=processfile)

                if TXsubimg:
                    sidx=substage[corr_node][img_id_isot]["count"]
                    substage[corr_node][img_id_isot]["data"][SUBIMGPIX*SUBIMGORDER[sidx][0]:SUBIMGPIX*(SUBIMGORDER[sidx][0]+1),
                                                                    SUBIMGPIX*SUBIMGORDER[sidx][1]:SUBIMGPIX*(SUBIMGORDER[sidx][1]+1),:] = arrData
                elif TXsubint:
                    #sidx = substage[corr_node][img_id_isot]["count"]
                    sidx = portmapping[port]
                    if substage[corr_node][img_id_isot]["data"] is None:
                        substage[corr_node][img_id_isot]["data"] = copy.deepcopy(arrData)
                    else:
                        substage[corr_node][img_id_isot]["data"] = np.concatenate([substage[corr_node][img_id_isot]["data"],arrData],axis=2)
                    #substage[corr_node][img_id_isot]["data"][:,:,int(sidx*(substage[corr_node][img_id_isot]["data"].shape[2]//TXnints)):int((sidx+1)*(substage[corr_node][img_id_isot]["data"].shape[2]//TXnints))] = arrData
                substage[corr_node][img_id_isot]["count"] += 1
                printlog(socksuffix+"CHECKPOINT1",output_file=processfile)
                printlog(socksuffix+"TIME TO ORGANIZE SUBINTS:"+str(time.time()-t1)+"s,"+str(substage[corr_node][img_id_isot]["count"]),output_file=processfile)
            

                if (TXsubimg and substage[corr_node][img_id_isot]["count"]>=NSUBIMG) or (TXsubint and substage[corr_node][img_id_isot]["count"]>=TXnints):
                    printlog(socksuffix+"GOT IN THE CORRECT CASE",output_file=processfile)
                    arrData=substage[corr_node][img_id_isot]["data"]
                    shape=arrData.shape
                    del substage[corr_node][img_id_isot]
                    printlog(socksuffix+"Gathered all subdata for corr"+str(corr_node),output_file=processfile)
                    sublock.release()
                else:
                    sublock.release()
                    continue
            t1 = time.time()
            printlog(socksuffix+"Adding to queue...",output_file=processfile)
            #try:
            queue.put_nowait(("mport",corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port))
            #except Exception as exc:
            #    printlog(socksuffix+"QUEUE PUT FAILED FOR "+img_id_isot+">>"+str(exc),output_file=processfile)
            printlog(socksuffix+"Done",output_file=processfile)
            printlog(socksuffix+"TIME TO ADD TO QUEUE:"+str(time.time()-t1)+"s",output_file=processfile)
            printlog("***GARBAGE COLLECTION>>"+str(gc.collect()),output_file=processfile)
    except Exception as exc:
        printlog("|EXCEPTION|"+socksuffix+str(exc),output_file=processfile)

    return

ECODE_BREAK = -1
ECODE_CONT = -2
ECODE_SUCCESS = 0
def readcorrdata(servSockD,ii,port,maxbytes,maxbyteshex,timeout,chunksize,headersize,datasize,testh23,offline,protocol,udpchunksize):
    socksuffix = "SOCKET " + str(ii) + " >>"
    if protocol=='udp':
        lastbyte=-1
        recdatabytes=bytes(0)
        nchunks = int(maxbytes//udpchunksize)
        nrec = int((maxbytes//nchunks) + headersize) #header includes byte number[8], host[16], isot[23],hdrarray[24]
        initmaxbytes=False
        while len(recdatabytes)<maxbytes-headersize:
            if len(recdatabytes)==0:
                servSockD.settimeout(None)
                printlog(socksuffix + "start UDP read...",output_file=processfile,end='')
            else:
                servSockD.settimeout(timeout/nchunks)
            try:
                data,addr = servSockD.recvfrom(nrec)
                bytenum=int.from_bytes(data[:8],byteorder='big')
                assert(bytenum>lastbyte)
                host=data[8:8+16].decode()
                img_id_isot=data[8+16:8+16+23].decode()
                img_uv_diag,img_dec,corr_node,shape_0,shape_1,shape_2,shape_3 = tuple(np.frombuffer(data[8+16+23:headersize]))
                printlog(socksuffix + str((bytenum,host,img_id_isot,img_uv_diag,img_dec,corr_node)),output_file=processfile)

                if bytenum-lastbyte>1:
                    printlog(socksuffix + "filling " + str(int((bytenum-lastbyte-1)*(nrec-headersize))) + " dropped bytes with 0",output_file=processfile)
                    recdatabytes += bytes(int((bytenum-lastbyte-1)*(nrec-headersize)))
                recdatabytes += data[headersize:]
                lastbyte=bytenum
            except Exception as exc:
                if 'time' in str(exc):
                    printlog(socksuffix + "Timeout?: "+str(exc),output_file=processfile)
                    if len(recdatabytes)==0:
                        printlog(socksuffix + "Timed out on first read, data unavailable",output_file=processfile)
                        return ECODE_BREAK
                elif 'Assertion' in str(exc):
                    printlog(socksuffix + "Bad data order:"+str(exc),output_file=processfile)

                if len(recdatabytes)<maxbytes-headersize:
                    recdatabytes += bytes(int((nrec-headersize)))
                lastbyte+=1
            shape = np.array([shape_0,shape_1,shape_2,shape_3],dtype=int)
            shape = tuple(shape[shape!=0])
            if not initmaxbytes:
                maxbytes=1
                for ss in shape:
                    maxbytes *= ss
                maxbytes = maxbytes*datasize + headersize
                maxbyteshex=(maxbytes + 4)*2 + 404
                nchunks = int(maxbytes//udpchunksize)
                nrec = int((maxbytes//nchunks) + headersize)
                initmaxbytes=True
                printlog(socksuffix+"Expecting "+str(maxbytes)+"bytes",output_file=processfile)
        arrData = np.frombuffer(recdatabytes,dtype=np.float64).reshape(shape)
        port = int(host[host.index(":")+1:])
        img_id_mjd=Time(img_id_isot,format='isot').mjd
        printlog(socksuffix +"Done, read "+str(len(recdatabytes))+"/"+str(maxbytes-headersize)+" into array shaped "+str(shape),output_file=processfile)
        return corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port
    printlog(socksuffix + "accepting connection...",output_file=processfile,end='')
    clientSocket,address = servSockD.accept()
    clientSocket.setblocking(0)
    
    corr_address, tmp = clientSocket.getpeername()
    printlog(socksuffix + "client: " + str(corr_address) + "...",output_file=processfile,end='')
    recstatus = 1
    fullMsg = ""
    printlog(socksuffix + "Done!",output_file=processfile)
    printlog(socksuffix + "Receiving data...",output_file=processfile)

    #set timeout and expected number of bytes to read
    clientSocket.settimeout(timeout)
    totalbytes = 0
    pflag = 0

    #while (recstatus> 0) and (totalbytes < maxbytes):#+maxbytesaddr):
    t_timeout = time.time()
    t_startread = time.time()
    totalbyteshex =0

    while (totalbyteshex < maxbyteshex) and time.time()-t_startread<60:# and time.time()-t_timeout<args.timeout:
        try:
            #check if data is ready to read first
            t_ready = time.time()
            while not select.select([clientSocket],[],[],timeout) and time.time()-t_ready<timeout:
                continue
            if not select.select([clientSocket],[],[],timeout):
                raise socket.timeout
            printlog(socksuffix+ "Data ready",output_file=processfile)

            (strData, ancdata, msg_flags, address) = clientSocket.recvmsg(chunksize)#255)
            recstatus = len(strData)
            if recstatus > 0:
                t_timeout = time.time()
            if recstatus == 0 and time.time()-t_timeout>timeout:
                raise socket.timeout

            printlog(socksuffix+"Read "+ str(recstatus) + " bytes, total "+ str(totalbytes+recstatus),output_file=processfile)
            printlog(socksuffix+"Message flags:" + str(msg_flags),output_file=processfile)
            printlog(socksuffix+"AncData:" + str(ancdata),output_file=processfile)
            fullMsg += strData.hex()
            totalbytes += recstatus
            totalbyteshex += len(strData.hex())
            #don't know how long the header is, so don't start counting until hit NP data
            if "93" in fullMsg:
                printlog(socksuffix+"Found start byte at index " + str(fullMsg.index("93")),output_file=processfile)
                totalbytes = (len(fullMsg) - fullMsg.index("93"))//2
        except Exception as ex:
            if type(ex) == socket.timeout:
                printlog(socksuffix+"Timed out after reading " + str(totalbytes) + " bytes; proceeding...",output_file=processfile)
                break
            else:
                raise
    printlog(socksuffix+"Done! Total bytes read:" + str(totalbytes),output_file=processfile)
    totalbytessend = 0

    #check if data is the size we expect
    try:
        assert(totalbyteshex>=maxbyteshex)
    except AssertionError as exc:
        printlog(socksuffix+"Invalid data size, " + str(totalbytes) + " received when expected at least " + str(maxbytes) + ": " + str(exc),output_file=processfile)

        printlog(socksuffix+"Invalid data size, " + str(totalbyteshex) + " received when expected at least " + str(maxbyteshex) + ": " + str(exc),output_file=processfile)
        printlog(socksuffix+"Setting truncated data size flag...",output_file=processfile,end='')
        pflag = set_pflag_loc("datasize_error")
        if pflag == None:
            printlog(socksuffix+"Error setting flags, abort",output_file=processfile)
            return ECODE_BREAK # break
        printlog(socksuffix+"Done, continue",output_file=processfile)
    if pflag != 0:
        successmsg = bytes(success + str(pflag) + '\n','utf-8')
        printlog(socksuffix+"Sending response...",output_file=processfile,end='')
        while (totalbytessend < len(successmsg)):
            totalbytessend += clientSocket.send(successmsg)
        printlog(socksuffix+"Done! Total bytes sent:" + str(totalbytessend) + "/" + str(len(successmsg)),output_file=processfile)
        return ECODE_CONT#continue

    #try to parse to get address
    try:
        corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData = parse_packet(fullMsg=fullMsg,maxbytes=maxbytes,headersize=headersize,datasize=datasize,port=port,corr_address=corr_address,testh23=testh23)
        printlog("PARSE SUCCESS:"+str((corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape)),output_file=processfile)
    except Exception as exc:
        if type(exc) == UnicodeDecodeError:
            printlog(socksuffix+"Error parsing data: " + str(exc),output_file=processfile)
            printlog(socksuffix+"Setting parse error flag...",output_file=processfile,end='')
            pflag = set_pflag_loc("parse_error")
            if pflag == None:
                printlog(socksuffix+"Error setting flags, abort",processfile=processfile)
                return ECODE_BREAK #break
            printlog(socksuffix+"Done, continue",output_file=processfile)
        if type(exc) == ValueError:
            printlog(socksuffix+"Invalid data size: " + str(exc),output_file=processfile)
            printlog(socksuffix+"Setting datasize flag...",output_file=processfile,end='')
            pflag = set_pflag_loc("datasize_error")
            if pflag == None:
                printlog(socksuffix+"Error setting flags, abort",processfile=processfile)
                return ECODE_BREAK #break
            printlog(socksuffix+"Done, continue",output_file=processfile)
        else:
            clientSocket.close()
            raise exc

    successmsg = bytes(success + str(pflag) + '\n','utf-8')
    printlog(socksuffix+"Sending response...",output_file=processfile,end='')
    while (totalbytessend < len(successmsg)):
        totalbytessend += clientSocket.send(successmsg)
    printlog(socksuffix+"Done! Total bytes sent:" + str(totalbytessend) + "/" + str(len(successmsg)),output_file=processfile)
    if pflag != 0:
        return ECODE_CONT #continue
    printlog(socksuffix+"Data: " + str(arrData),output_file=processfile)

    #reopen

    return corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port

multiport_accepting = dict()
def multiport_task(corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,ii,testh23,offline,SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,PyTorchDedispersion,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,appendframe,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,nchans,executor,slow,imgdiff,etcd_enabled,dask_enabled,
                                    attachmode,completeness,slowlock,searchlock,forfeit,rtastrocal,testsinglenode,TXsubimg,TXsubint,TXnints):
    """
    This task sets up the given socket to accept connections, reads
    data when a client connects, and submits a search task
    """
    
    if dask_enabled:
        executor = get_client()
        slowlock = Lock_DASK("dasklock",client=executor)
        searchlock = Lock_DASK("srchlock",client=executor)
    socksuffix = "MPORT " + str(ii) + " ["+img_id_isot+"] >>"
    #if object is in dict
    printlog(socksuffix + "STARTING MULTIPORT TASK>>[TIME]"+str(time.time()),output_file=processfile)
    printlog(socksuffix + "TRYING TO ACQUIRE MUTEX...",output_file=processfile)
    slowlock.acquire()
    printlog(socksuffix + "ACQUIRED",output_file=processfile)
    if corr_node != -1:
        if img_id_isot not in fullimg_dict.keys():
            fullimg_dict[img_id_isot] = fullimg(img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape=tuple(np.concatenate([shape,[nchans]])),TXsubimg=TXsubimg,TXsubint=TXsubint,TXnints=TXnints)
        
        printlog(socksuffix+"IMAGE SHAPE: " + str(img_id_isot) + ", " + str(fullimg_dict[img_id_isot].image_tesseract.shape),output_file=processfile)
        #add image and update flags
        if testsinglenode:
            for corr_node_ in range(16):
                fullimg_dict[img_id_isot].add_corr_img(arrData,corr_node_,testh23,TXsubimg,TXsubint,TXnints) #fullimg_array[idx].add_corr_img(arrData,corr_node,args.testh23)
            time.sleep(1)
        else:
            printlog("RIGHT HERE "+str(corr_node)+str(arrData.shape)+str(img_id_isot in fullimg_dict.keys()),output_file=processfile)
            fullimg_dict[img_id_isot].add_corr_img(arrData,corr_node,testh23,TXsubimg,TXsubint,TXnints) #fullimg_array[idx].add_corr_img(arrData,corr_node,args.testh23)
            printlog("NOW HERE",output_file=processfile)
        #if the image is complete, start the search
        printlog(socksuffix+"corrstatus:",output_file=processfile,end='')
        printlog(fullimg_dict[img_id_isot].corrstatus,output_file=processfile)
    else:
        timeout_isots.append(img_id_isot)
    fullimg_dict[img_id_isot].timerlock.acquire()

    #check if timed out
    #if fullimg_dict[img_id_isot].is_timeout():
    #    printlog(socksuffix+"searching incomplete data for "+img_id_isot,output_file=processfile)
    #    fullimg_dict[img_id_isot].nanfill()
    printlog(socksuffix + "TRYING TO RELEASE MUTEX...",output_file=processfile)
    slowlock.release()
    printlog(socksuffix + "RELEASED",output_file=processfile)

    printlog(socksuffix + "MULTIPORT MIDWAY>>[TIME]"+str(time.time()),output_file=processfile)


    if fullimg_dict[img_id_isot].is_full(): #fullimg_array[idx].is_full():
        f = open(sslogfile,"a")
        thash = fullimg_dict[img_id_isot].thash #hex(random.getrandbits(32))
        f.write("[start] [" + thash + "] " + str(time.time()))
        f.close()

        #save data for flux and astrometric cal
        if rtastrocal:
            reftime = Time(int(np.floor(img_id_mjd)),format='mjd')
            if len(glob.glob(table_dir + "/rt_speccal_timestamps_"+reftime.isot+".json"))>0:
                printlog(table_dir + "/rt_speccal_timestamps_"+reftime.isot+".json",output_file=processfile)
                f = open(table_dir + "/rt_speccal_timestamps_"+reftime.isot+".json","r")
                speccaldict = json.load(f)
                f.close()
                for k in speccaldict.keys():
                    printlog(k,output_file=processfile)
                    if 0< (img_id_mjd-speccaldict[k])<(config.T/1000/86400):
                        printlog(vis_dir + "/"+str(k).replace(" ","")+"/image_"+img_id_isot+".npy",output_file=processfile)
                        np.save(vis_dir + "/"+str(k).replace(" ","")+"/image_"+img_id_isot+".npy",fullimg_dict[img_id_isot].image_tesseract)
            if len(glob.glob(table_dir + "/rt_astrocal_timestamps_"+reftime.isot+".json"))>0:
                printlog(table_dir + "/rt_astrocal_timestamps_"+reftime.isot+".json",output_file=processfile)
                f = open(table_dir + "/rt_astrocal_timestamps_"+reftime.isot+".json","r")
                astrocaldict = json.load(f)
                f.close()
                for k in astrocaldict.keys():
                    printlog(k,output_file=processfile)
                    if 0<(img_id_mjd-astrocaldict[k])<(15*config.T/1000/86400):
                        printlog(vis_dir + "/"+str(k).replace(" ","")+"/image_"+img_id_isot+".npy",output_file=processfile)
                        np.save(vis_dir + "/"+str(k).replace(" ","")+"/image_"+img_id_isot+".npy",fullimg_dict[img_id_isot].image_tesseract)


        
        #submit a search task to the process pool
        printlog(socksuffix+"Submitting new task for image " + str(img_id_isot),output_file=processfile)
        RA_axis_idx = copy.deepcopy(fullimg_dict[img_id_isot].RA_axis) #copy.deepcopy(fullimg_array[idx].RA_axis)
        DEC_axis_idx= copy.deepcopy(fullimg_dict[img_id_isot].DEC_axis) #copy.deepcopy(fullimg_array[idx].DEC_axis)

        #update noise from file if offline
        if offline:
            sl.last_frame = sl.get_last_frame()

        #initialize noise stats
        if fprtest or fnrtest:
            printlog(socksuffix+"FPR Test, Re-Initializing noise statistics...",output_file=processfile)
            init_noise(sl.DM_trials,sl.widthtrials,config.gridsize,config.gridsize)
            sl.current_noise = noise_update_all(None,config.gridsize,config.gridsize,sl.DM_trials,sl.widthtrials,readonly=True)

        printlog(">>>>"+str(img_id_mjd)+"<<<<",output_file=processfile)
        printlog(socksuffix + "MULTIPORT ALSO>>[TIME]"+str(time.time()),output_file=processfile)
        #make slow fullimag object
        if slow:
            printlog(socksuffix + "TRYING TO ACQUIRE MUTEX...",output_file=processfile)
            slowlock.acquire()
            printlog(socksuffix + "ACQUIRED",output_file=processfile)
            slowdone = False
            try:
                if len(slow_fullimg_dict.keys())>0:
                    k_idxs_ = (img_id_mjd-Time(list(slow_fullimg_dict.keys()),format='isot').mjd)>=0
                    k_idx = np.argmin((img_id_mjd-Time(list(slow_fullimg_dict.keys()),format='isot').mjd)[k_idxs_])
                    printlog(socksuffix + str(k_idxs_),output_file=processfile)
                    printlog(socksuffix + str(k_idx),output_file=processfile)
                    printlog(socksuffix+str(Time(list(slow_fullimg_dict.keys()),format='isot').mjd[k_idxs_]),output_file=processfile)
                    k = list(slow_fullimg_dict.keys())[k_idx]
                    #k = (np.array(slow_fullimg_dict.keys())[k_idxs_])[np.argmin(img_id_mjd - np.array(Time(list(slow_fullimg_dict.keys()),format='isot').mjd[k_idxs_]))]
                    #k=np.array(slow_fullimg_dict.keys())[k_idxs_][np.argmin(img_id_mjd-Time(np.array(slow_fullimg_dict.keys())[k_idxs_],format='isot').mjd)]
                    printlog(socksuffix+"CLOSEST K:"+str(k),output_file=processfile)
                    #for k in [k]:#slow_fullimg_dict.keys():
                    img_idx,foundmjd = slow_fullimg_dict[k].slow_mjd_in_img(img_id_mjd)
                    if (not slow_fullimg_dict[k].slow_is_full()) and foundmjd:
                        printlog(socksuffix+"SLOW MJD:" + str(img_id_mjd),output_file=processfile)
                        printlog(socksuffix+str((img_id_mjd - Time(str(k),format='isot').mjd)*86400*1000)  + "/" + str(config.nsamps*config.tsamp_slow) + " slow append:" + str(slow_fullimg_dict[k].slow_counter),output_file=processfile)
                        printlog(socksuffix+"IS THE ISOT STILL THERE?:"+str(img_id_isot in fullimg_dict.keys()),output_file=processfile)
                        if corr_node==-1 and (img_id_isot not in fullimg_dict.keys()):
                            printlog(socksuffix+"ABORT TIMEOUT SEARCH",output_file=processfile)
                            slowlock.release()
                            fullimg_dict[img_id_isot].timerlock.release()
                            return ECODE_BREAK
                        slow_fullimg_dict[k].slow_append_img(fullimg_dict[img_id_isot].image_tesseract,img_idx)

                        printlog(socksuffix+"SUCCESSFUL SLOW APPEND",output_file=processfile)
                        slowdone = True
                        #break
            except Exception as exc:
                printlog(socksuffix+"|EXCEPTION| SLOW:"+str(exc),output_file=processfile)
                printlog(exc,output_file=processfile)
            printlog(socksuffix + "SLOW PREAPPEND>>[TIME]"+str(time.time()),output_file=processfile)
            if not slowdone:
                printlog(socksuffix+"FIRST SLOW MJD:" + str(img_id_mjd),output_file=processfile)
                slow_fullimg_dict[img_id_isot] = fullimg(img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape=tuple(np.concatenate([shape,[nchans]])),slow=True)
                slow_fullimg_dict[img_id_isot].slow_append_img(fullimg_dict[img_id_isot].image_tesseract,0)
                k = img_id_isot
            slowsearch_now = (slowdone and slow_fullimg_dict[k].slow_is_full())
            if slowsearch_now:
                slow_fullimg_dict[k].thash = fullimg_dict[img_id_isot].thash
            printlog(socksuffix + "SLOW POSTAPPEND>>[TIME]"+str(time.time()),output_file=processfile)
            printlog(socksuffix + "TRYING TO RELEASE MUTEX...",output_file=processfile)
            slowlock.release()
            printlog(socksuffix + "RELEASED",output_file=processfile)

        else:
            slowsearch_now = False
        #make imgdiff fullimag object
        if imgdiff:
            printlog(socksuffix + "TRYING TO ACQUIRE MUTEX...",output_file=processfile)
            slowlock.acquire()
            printlog(socksuffix + "ACQUIRED",output_file=processfile)
            imgdiffdone = False
            try:
                if len(imgdiff_fullimg_dict.keys())>0:
                    kd_idxs_ = (img_id_mjd-Time(list(imgdiff_fullimg_dict.keys()),format='isot').mjd)>=0
                    kd_idx = np.argmin((img_id_mjd-Time(list(imgdiff_fullimg_dict.keys()),format='isot').mjd)[kd_idxs_])
                    kd = list(imgdiff_fullimg_dict.keys())[kd_idx]
                    #kd = (np.array(imgdiff_fullimg_dict.keys())[kd_idxs_])[np.argmin(img_id_mjd - np.array(Time(list(imgdiff_fullimg_dict.keys()),format='isot').mjd[kd_idxs_]))]
                    #kd=np.array(imgdiff_fullimg_dict.keys())[kd_idxs_][np.argmin(img_id_mjd-Time(np.array(imgdiff_fullimg_dict.keys())[kd_idxs_],format='isot').mjd)]
                    printlog(socksuffix+"CLOSEST KD:"+str(kd),output_file=processfile)
                    #for kd in [kd]:#for kd in imgdiff_fullimg_dict.keys():
                    img_idx,foundmjd = imgdiff_fullimg_dict[kd].imgdiff_mjd_in_img(img_id_mjd)
                    if (not imgdiff_fullimg_dict[kd].imgdiff_is_full()) and foundmjd:
                        printlog(socksuffix+"IMGDIFF MJD:" + str(img_id_mjd),output_file=processfile)
                        printlog(socksuffix+str((img_id_mjd - Time(str(kd),format='isot').mjd)*86400*1000)  + "/" + str(config.nsamps*config.tsamp_slow) + " imgdiff append:" + str(imgdiff_fullimg_dict[kd].slow_counter),output_file=processfile)
                    
                        printlog(socksuffix+"IS THE ISOT STILL THERE?:"+str(img_id_isot in fullimg_dict.keys()),output_file=processfile)
                        if corr_node==-1 and (img_id_isot not in fullimg_dict.keys()):
                            printlog(socksuffix+"ABORT TIMEOUT SEARCH",output_file=processfile)
                            slowlock.release()
                            fullimg_dict[img_id_isot].timerlock.release()
                            return ECODE_BREAK
                        imgdiff_fullimg_dict[kd].imgdiff_append_img(fullimg_dict[img_id_isot].image_tesseract,img_idx)

                        imgdiffdone=True
                        #break
            except Exception as exc:
                printlog(socksuffix+"|EXCEPTION| IMGDIFF:"+str(exc),output_file=processfile)
                printlog(exc,output_file=processfile)
            printlog(socksuffix + "IMGDIFF PREAPPEND>>[TIME]"+str(time.time()),output_file=processfile)
            if not imgdiffdone:
                printlog(socksuffix+"FIRST IMGDIFF MJD:" + str(img_id_mjd),output_file=processfile)
                imgdiff_fullimg_dict[img_id_isot] = fullimg(img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape=tuple(np.concatenate([shape[:2],[args.imgdiffgulps,1]])),slow=False,imgdiff=True)
                imgdiff_fullimg_dict[img_id_isot].imgdiff_append_img(fullimg_dict[img_id_isot].image_tesseract,0)
                kd = img_id_isot
            imgdiffsearch_now = (imgdiffdone and imgdiff_fullimg_dict[kd].imgdiff_is_full())
            if imgdiffsearch_now:
                imgdiff_fullimg_dict[kd].thash = fullimg_dict[img_id_isot].thash
            printlog(socksuffix + "IMGDIFF POSTAPPEND>>[TIME]"+str(time.time()),output_file=processfile)
            printlog(socksuffix + "TRYING TO RELEASE MUTEX...",output_file=processfile)
            slowlock.release()
            printlog(socksuffix + "RELEASED",output_file=processfile)
        else:
            imgdiffsearch_now = False

        if corr_node==-1 and (img_id_isot not in fullimg_dict.keys()):
            printlog(socksuffix+"ABORT TIMEOUT SEARCH",output_file=processfile)
            fullimg_dict[img_id_isot].timerlock.release()
            return ECODE_BREAK


        #submit task
        printlog(socksuffix + "PRESUBMIT TASK>>[TIME]"+str(time.time()),output_file=processfile)
        if attachmode:
            fullimg_dict[img_id_isot].running = True
            slow_fullimg_dict[k].running=True
            imgdiff_fullimg_dict[kd].running=True
            printlog(socksuffix + "SUBMITTING ATTACH-MODE TASK",output_file=processfile)
            if dask_enabled:
                stask = executor.submit(sl.search_task,searchlock,fullimg_dict[img_id_isot],SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,appendframe,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,False,False,(slow_fullimg_dict[k] if slowsearch_now else None),(imgdiff_fullimg_dict[kd] if imgdiffsearch_now else None),True,completeness,resources={'GPU': 1,'MEMORY':10e6})
            else:
                stask = executor.submit(sl.search_task,searchlock,fullimg_dict[img_id_isot],SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,appendframe,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,False,False,(slow_fullimg_dict[k] if slowsearch_now else None),(imgdiff_fullimg_dict[kd] if imgdiffsearch_now else None),True,completeness)
            stask.add_done_callback(lambda future: future_callback_attach(future,SNRthresh,img_id_isot,RA_axis_idx,
                                                                                            str(k) if slowsearch_now else None,
                                                                                            RA_axis_idx[slow_fullimg_dict[k].slow_RA_cutoff:] if slowsearch_now else None,
                                                                                            str(kd) if imgdiffsearch_now else None,
                                                                                            RA_axis_idx[imgdiff_fullimg_dict[kd].imgdiff_RA_cutoff:] if imgdiffsearch_now else None,
                                                                                            DEC_axis_idx,
                                                                                            etcd_enabled,dask_enabled,thash))
            fullimg_dict[img_id_isot].timerlock.release()
            return [stask]

        ret_tasks = []
        if (not forfeit) or (forfeit and (not slowsearch_now) and (not imgdiffsearch_now)):
            printlog("ABOUT TO SUBMIT SEARCH TASK...",output_file=processfile)
            fullimg_dict[img_id_isot].running = True
            
            if dask_enabled:
                stask = executor.submit(sl.search_task,searchlock,fullimg_dict[img_id_isot],SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,appendframe,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,False,False,None,None,False,completeness,forfeit,resources={'GPU': 1,'MEMORY':10e6})
            else:
                """
                stask = executor.submit(sl.search_task,searchlock,fullimg_dict[img_id_isot],SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,appendframe,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,False,False,None,None,False,completeness,forfeit))
                """
                printlog(socksuffix+"Adding to SEARCH QUEUE:"+str(("srch",img_id_isot,thash,False,False)),output_file=processfile)
                QQUEUE_UDP.put(("srch",img_id_isot,thash,False,False))
            #stask.add_done_callback(lambda future: future_callback(future,SNRthresh,img_id_isot,RA_axis_idx,DEC_axis_idx,etcd_enabled,dask_enabled,thash))
            #ret_tasks.append(stask)
            printlog("DID IT SUBMIT?",output_file=processfile)
        elif forfeit and (slowsearch_now or imgdiffsearch_now):
            fullimg_dict[img_id_isot].timerlock.release()
            fullimg_dict[img_id_isot] = None
            del fullimg_dict[img_id_isot]
            printlog(socksuffix+"BASE SEARCH FORFEIT "+img_id_isot,output_file=processfile)

        #task_list.append(stask)
        if slowsearch_now and ((not forfeit) or (forfeit and (not imgdiffsearch_now))):
            slow_fullimg_dict[k].running=True
            printlog(socksuffix+"SLOWSEARCH NOW:" + str(slow_fullimg_dict[k]),output_file=processfile)
            printlog(socksuffix+str(slow_fullimg_dict[k].image_tesseract),output_file=output_file)
            #np.save("tmp_slow_search_input.npy",slow_fullimg_dict[k].image_tesseract)
            if dask_enabled:
                sstask = executor.submit(sl.search_task,searchlock,slow_fullimg_dict[k],SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,appendframe,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,True,False,None,None,False,completeness,forfeit,resources={'GPU': 1,'MEMORY':10e6})
            else:
                """
                sstask = executor.submit(sl.search_task,searchlock,slow_fullimg_dict[k],SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,appendframe,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,True,False,None,None,False,completeness,forfeit))
                """    
                printlog(socksuffix+"Adding to SEARCH QUEUE [SLOW]:"+str(("srch",str(k),thash,True,False)),output_file=processfile)
                QQUEUE_UDP.put(("srch",str(k),thash,True,False))
            #sstask.add_done_callback(lambda future: future_callback(future,SNRthresh,str(k),RA_axis_idx[slow_fullimg_dict[k].slow_RA_cutoff:],DEC_axis_idx,etcd_enabled,dask_enabled,thash))
            #ret_tasks.append(sstask)
        elif forfeit and slowsearch_now:
            slow_fullimg_dict[k] = None
            del slow_fullimg_dict[k]
            printlog(socksuffix + "SLOW SEARCH FORFEIT "+str(k),output_file=processfile)

        if imgdiffsearch_now:
            imgdiff_fullimg_dict[kd].running=True
            printlog(socksuffix+"IMGDIFFSEARCH NOW:" + str(imgdiff_fullimg_dict[kd]),output_file=processfile)
            printlog(socksuffix+str(imgdiff_fullimg_dict[kd].image_tesseract),output_file=output_file)
            #np.save("tmp_slow_search_input.npy",slow_fullimg_dict[k].image_tesseract)
            if dask_enabled:
                ssstask = executor.submit(sl.search_task,searchlock,imgdiff_fullimg_dict[kd],SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,False,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,False,True,None,None,False,completeness,forfeit,resources={'GPU': 1,'MEMORY':10e6})
            else:
                """
                ssstask = executor.submit(sl.search_task,searchlock,imgdiff_fullimg_dict[kd],SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,
                                    multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,
                                    spacefilter,kernelsize,exportmaps,savesearch,fprtest,fnrtest,False,DMbatches,
                                    SNRbatches,usejax,noiseth,nocutoff,realtime,False,True,None,None,False,completeness,forfeit))
                """
                printlog(socksuffix+"Adding to SEARCH QUEUE [IMGDIFF]:"+str(("srch",str(kd),thash,False,True)),output_file=processfile)
                QQUEUE_UDP.put(("srch",str(kd),thash,False,True))
            #ssstask.add_done_callback(lambda future: future_callback(future,SNRthresh,str(kd),RA_axis_idx[imgdiff_fullimg_dict[kd].imgdiff_RA_cutoff:],DEC_axis_idx,etcd_enabled,dask_enabled,thash))
            #ret_tasks.append(ssstask)
        printlog("***GARBAGE COLLECTION>>"+str(gc.collect()),output_file=processfile)
        fullimg_dict[img_id_isot].timerlock.release()
        printlog(socksuffix+"MULTIPATH FINISHED>>[TIME]"+str(time.time()),output_file=processfile )
        return ret_tasks
        printlog(socksuffix+" "+str(ret_tasks) + " tasks",output_file=processfile )
        #if slowsearch_now and not imgdiffsearch_now: return [stask,sstask]
        #elif slowsearch_now and imgdiffsearch_now: return [stask,sstask,ssstask]
        #elif not slowsearch_now and imgdiffsearch_now: return [stask,ssstask]
        #return [stask]
    fullimg_dict[img_id_isot].timerlock.release()
    printlog(socksuffix+"MULTIPATH FINISHED>>[TIME]"+str(time.time()),output_file=processfile )
    return ECODE_SUCCESS

def main(args):
    #redirect stderr
    sys.stderr = open(error_file,"w")
    
    #if "DASKPORT" in os.environ.keys():
    #    printlog("Using Dask Scheduler on Port " + str(os.environ['DASKPORT']) + " for cand_cutter queue",output_file=processfile)
    if args.etcd:
        printlog("Etcd enabled, will push candidates to " + ETCDKEY,output_file=processfile)

    #update default values and lookup tables
    sl.SNRthresh = args.SNRthresh
    if args.gridsize != config.gridsize or args.nchans != config.nchans or args.nsamps != config.nsamps or args.imgdiffgulps != config.ngulps_per_file:

        config.nsamps = args.nsamps
        sl.nsamps = args.nsamps
        config.T = config.nsamps*config.tsamp
        sl.T = config.nsamps*config.tsamp
        sl.time_axis = np.linspace(0,config.T,config.nsamps)
        sl.widthtrials = sl.widthtrials[sl.widthtrials<args.nsamps]
        sl.nwidths = len(sl.widthtrials)
        sl.full_boxcar_filter = sl.gen_boxcar_filter(sl.widthtrials,args.nsamps)
        sl.full_boxcar_filter_imgdiff = sl.gen_boxcar_filter(sl.widthtrials,args.imgdiffgulps)

        config.nchans = args.nchans
        sl.nchans = args.nchans
        config.freq_axis_fullres = 1000*((1.53-np.arange(8192)*0.25/8192)[1024:1024+int(config.nchans*config.NUM_CHANNELS/2)]) #MHz
        config.freq_axis = np.reshape(config.freq_axis_fullres,(config.nchans,int(config.NUM_CHANNELS/2))).mean(axis=1) #MHz
        sl.freq_axis = config.freq_axis
        config.chanbw = np.abs(config.freq_axis[0]-config.freq_axis[1]) #MHz
        config.fmax = np.max(config.freq_axis) #MHz
        config.fmin = np.min(config.freq_axis) #MHz
        config.fc =  (config.fmin+config.fmax)/2 #MHz
        config.lambdamin = (config.c/(config.fmax*1e6)) #m
        config.lambdamax = (config.c/(config.fmin*1e6)) #m
        config.lambdac = (config.c/(config.fc*1e6)) #m
        config.lambdaref = (config.c/(config.freq_axis_fullres[0]*1e6))

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

        sl.tDM_max = (4.15)*np.max(sl.DM_trials)*((1/np.min(sl.freq_axis)/1e-3)**2 - (1/np.max(sl.freq_axis)/1e-3)**2) #ms
        sl.maxshift = int(np.ceil(sl.tDM_max/config.tsamp))

        sl.DM_trials_slow = np.array(sl.gen_dm(sl.minDM*5,sl.maxDM*5,config.DM_tol,config.fc*1e-3,config.nchans,config.tsamp_slow,config.chanbw,args.nsamps))#np.array(sl.gen_dm(sl.minDM*5,sl.maxDM,config.DM_tol_slow,config.fc*1e-3,config.nchans,config.tsamp_slow,config.chanbw,args.nsamps))#[0:1]
        sl.nDMtrials_slow = len(sl.DM_trials_slow)

        sl.corr_shifts_all_append_slow,sl.tdelays_frac_append_slow,sl.corr_shifts_all_no_append_slow,sl.tdelays_frac_no_append_slow = sl.gen_dm_shifts(sl.DM_trials_slow,sl.freq_axis,config.tsamp_slow,config.nsamps)

        #sl.current_noise = noise_update_all(None,config.gridsize,config.gridsize,sl.DM_trials,sl.widthtrials,readonly=True) #get_noise_dict(config.gridsize,config.gridsize)
        sl.tDM_max_slow = (4.15)*np.max(sl.DM_trials_slow)*((1/np.min(sl.freq_axis)/1e-3)**2 - (1/np.max(sl.freq_axis)/1e-3)**2) #ms
        sl.maxshift_slow = int(np.ceil(sl.tDM_max_slow/config.tsamp_slow))


        printlog("UPDATED MAXSHIFT:" + str(sl.maxshift),output_file=processfile)

    if args.nocutoff:
        sl.default_cutoff = 0
    else:
        printlog(sl.DEC_axis,output_file=processfile)
        sl.default_cutoff = get_RA_cutoff(np.nanmean(sl.DEC_axis),pixsize=np.abs(sl.RA_axis[1]-sl.RA_axis[0]))
        printlog("Initialized pixel cutoff to " + str(sl.default_cutoff) + " pixels",output_file=processfile)

    """
    #move PSF, width trials, DM trials to GPU
    sl.default_PSF_gpu_0 = jax.device_put(np.array(sl.default_PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(sl.default_PSF[:,:,0:1,:].sum(3,keepdims=True))))[int((sl.default_PSF.shape[0]-sl.gridsize)//2):int((sl.default_PSF.shape[0]-sl.gridsize)//2)+sl.gridsize,
                      (int((sl.default_PSF.shape[1]-sl.gridsize)//2))+int(sl.default_cutoff//2):(int((sl.default_PSF.shape[1]-sl.gridsize)//2)+sl.gridsize)+(-(sl.default_cutoff - int(sl.default_cutoff//2)))],jax.devices()[0])
    sl.default_PSF_gpu_1 = jax.device_put(np.array(sl.default_PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(sl.default_PSF[:,:,0:1,:].sum(3,keepdims=True))))[int((sl.default_PSF.shape[0]-sl.gridsize)//2):int((sl.default_PSF.shape[0]-sl.gridsize)//2)+sl.gridsize,
                      (int((sl.default_PSF.shape[1]-sl.gridsize)//2))+int(sl.default_cutoff//2):(int((sl.default_PSF.shape[1]-sl.gridsize)//2)+sl.gridsize)+(-(sl.default_cutoff - int(sl.default_cutoff//2)))],jax.devices()[1])
    sl.corr_shifts_all_gpu_0 = jax.device_put(sl.corr_shifts_all_gpu_0,jax.devices()[0])
    sl.tdelays_frac_gpu_0 = jax.device_put(sl.tdelays_frac_gpu_0,jax.devices()[0])
    sl.corr_shifts_all_gpu_1 = jax.device_put(sl.corr_shifts_all_gpu_1,jax.devices()[1])
    sl.tdelays_frac_gpu_1 = jax.device_put(sl.tdelays_frac_gpu_1,jax.devices()[1])
    sl.full_boxcar_filter_gpu_0 = jax.device_put(sl.full_boxcar_filter_gpu_0,jax.devices()[0])
    sl.full_boxcar_filter_gpu_1 = jax.device_put(sl.full_boxcar_filter_gpu_1,jax.devices()[1])
    """

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
        sl.init_last_frame(args.gridsize,args.gridsize,args.nsamps-sl.maxshift_slow,args.nchans,slow=True)

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

        if args.completeness:
            #append (normal)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_completeness(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,maxshift+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),
                                               #sl.default_PSF_gpu_0,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),
                                               #sl.corr_shifts_all_gpu_0,
                                               jax.device_put(corr_shifts_all,jax.devices()[0]),
                                               #sl.tdelays_frac_gpu_0,
                                               jax.device_put(tdelays_frac,jax.devices()[0]),
                                               #sl.full_boxcar_filter_gpu_0,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_completeness(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,maxshift+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[1]),
                                               #sl.default_PSF_gpu_1,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),
                                               #sl.corr_shifts_all_gpu_1,
                                               jax.device_put(corr_shifts_all,jax.devices()[1]),
                                               #sl.tdelays_frac_gpu_1,
                                               jax.device_put(tdelays_frac,jax.devices()[1]),
                                               #sl.full_boxcar_filter_gpu_1,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)

            #no append (slow)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append_completeness(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),
                                               #sl.default_PSF_gpu_0,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),
                                               #sl.corr_shifts_all_gpu_0,
                                               jax.device_put(sl.corr_shifts_all_no_append_slow,jax.devices()[0]),
                                               #sl.tdelays_frac_gpu_0,
                                               jax.device_put(sl.tdelays_frac_no_append_slow,jax.devices()[0]),
                                               #sl.full_boxcar_filter_gpu_0,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append_completeness(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[1]),
                                               #sl.default_PSF_gpu_1,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),
                                               #sl.corr_shifts_all_gpu_1,
                                               jax.device_put(sl.corr_shifts_all_append,jax.devices()[1]),
                                               #sl.tdelays_frac_gpu_1,
                                               jax.device_put(sl.tdelays_frac_append,jax.devices()[1]),
                                               #sl.full_boxcar_filter_gpu_1,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)


            #append (slow)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append_completeness(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,sl.maxshift_slow+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),
                                               #sl.default_PSF_gpu_0,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),
                                               #sl.corr_shifts_all_gpu_0,
                                               jax.device_put(sl.corr_shifts_all_append_slow,jax.devices()[0]),
                                               #sl.tdelays_frac_gpu_0,
                                               jax.device_put(sl.tdelays_frac_append_slow,jax.devices()[0]),
                                               #sl.full_boxcar_filter_gpu_0,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append_completeness(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,sl.maxshift_slow+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[1]),
                                               #sl.default_PSF_gpu_1,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),
                                               #sl.corr_shifts_all_gpu_1,
                                               jax.device_put(sl.corr_shifts_all_append_slow,jax.devices()[1]),
                                               #sl.tdelays_frac_gpu_1,
                                               jax.device_put(sl.tdelays_frac_append_slow,jax.devices()[1]),
                                               #sl.full_boxcar_filter_gpu_1,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)


            #no append, image differencing
            jax_funcs.img_diff_jit_no_append_completeness(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.imgdiffgulps,1)),dtype=np.float32),jax.devices()[0]),
                                               #jax_funcs.PSF_1,#
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),
                                               jax.device_put(sl.full_boxcar_filter_imgdiff,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.img_diff_jit_no_append_completeness(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.imgdiffgulps,1)),dtype=np.float32),jax.devices()[1]),
                                               #jax_funcs.PSF_2,#
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),
                                               jax.device_put(sl.full_boxcar_filter_imgdiff,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)






        else:
            #append (normal)
            jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,maxshift+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),
                                               #sl.default_PSF_gpu_0,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),
                                               #sl.corr_shifts_all_gpu_0,
                                               jax.device_put(corr_shifts_all,jax.devices()[0]),
                                               #sl.tdelays_frac_gpu_0,
                                               jax.device_put(tdelays_frac,jax.devices()[0]),
                                               #sl.full_boxcar_filter_gpu_0,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,maxshift+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[1]),
                                               #sl.default_PSF_gpu_1,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),
                                               #sl.corr_shifts_all_gpu_1,
                                               jax.device_put(corr_shifts_all,jax.devices()[1]),
                                               #sl.tdelays_frac_gpu_1,
                                               jax.device_put(tdelays_frac,jax.devices()[1]),
                                               #sl.full_boxcar_filter_gpu_1,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)
        
            #no append (slow)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),
                                               #sl.default_PSF_gpu_0,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),
                                               #sl.corr_shifts_all_gpu_0,
                                               jax.device_put(sl.corr_shifts_all_no_append_slow,jax.devices()[0]),
                                               #sl.tdelays_frac_gpu_0,
                                               jax.device_put(sl.tdelays_frac_no_append_slow,jax.devices()[0]),
                                               #sl.full_boxcar_filter_gpu_0,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[1]),
                                               #sl.default_PSF_gpu_1,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),
                                               #sl.corr_shifts_all_gpu_1,
                                               jax.device_put(sl.corr_shifts_all_append,jax.devices()[1]),
                                               #sl.tdelays_frac_gpu_1,
                                               jax.device_put(sl.tdelays_frac_append,jax.devices()[1]),
                                               #sl.full_boxcar_filter_gpu_1,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)


            #append (slow)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,sl.maxshift_slow+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[0]),
                                               #sl.default_PSF_gpu_0,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),
                                               #sl.corr_shifts_all_gpu_0,
                                               jax.device_put(sl.corr_shifts_all_append_slow,jax.devices()[0]),
                                               #sl.tdelays_frac_gpu_0,
                                               jax.device_put(sl.tdelays_frac_append_slow,jax.devices()[0]),
                                               #sl.full_boxcar_filter_gpu_0,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,sl.maxshift_slow+args.nsamps,args.nchans)),dtype=np.float32),jax.devices()[1]),
                                               #sl.default_PSF_gpu_1,
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),
                                               #sl.corr_shifts_all_gpu_1,
                                               jax.device_put(sl.corr_shifts_all_append_slow,jax.devices()[1]),
                                               #sl.tdelays_frac_gpu_1,
                                               jax.device_put(sl.tdelays_frac_append_slow,jax.devices()[1]),
                                               #sl.full_boxcar_filter_gpu_1,
                                               jax.device_put(sl.full_boxcar_filter,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)

        
            #no append, image differencing
            jax_funcs.img_diff_jit_no_append(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.imgdiffgulps,1)),dtype=np.float32),jax.devices()[0]),
                                               #jax_funcs.PSF_1,#
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[0]),
                                               jax.device_put(sl.full_boxcar_filter_imgdiff,jax.devices()[0]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[0]),past_noise_N=1,noiseth=args.noiseth)
            jax_funcs.img_diff_jit_no_append(jax.device_put(np.array(np.random.normal(size=(args.gridsize,args.gridsize-sl.default_cutoff,args.imgdiffgulps,1)),dtype=np.float32),jax.devices()[1]),
                                               #jax_funcs.PSF_2,#
                                               jax.device_put(np.array(np.random.normal(size=(args.kernelsize,args.kernelsize if args.gridsize > args.kernelsize else args.kernelsize-sl.default_cutoff,1,1)),dtype=np.float32),jax.devices()[1]),
                                               jax.device_put(sl.full_boxcar_filter_imgdiff,jax.devices()[1]),
                                               jax.device_put(np.array(np.random.normal(size=len(sl.widthtrials)),dtype=config.noise_data_type),jax.devices()[1]),past_noise_N=1,noiseth=args.noiseth)
        

    printlog("USEFFT = " + str(args.usefft),output_file=processfile)
    TXsubimg = args.TXmode=='subimg'
    TXsubint = args.TXmode=='subint'

    printlog("TX MODE:"+str((args.TXmode,TXsubimg,TXsubint,args.TXnints)),output_file=processfile)
    printlog("SAMPS:"+str((args.nsamps,args.TXnints,args.nsamps//args.TXnints)),output_file=processfile)

    #total expected number of bytes for each sub-band image
    portmapping = dict() #dictionary defining which ports correspond to which sub-int/sub-image
    if TXsubimg:
        if args.datasize%2 != 0:
            maxbytes = SUBIMGPIX*SUBIMGPIX*args.nsamps*(args.datasize-1) + args.headersize #really just payload size
            maxbyteshex = (SUBIMGPIX*SUBIMGPIX*args.nsamps*(args.datasize-1) + args.headersize + 4)*2 + 404 #http header is 404
        else:
            maxbytes = SUBIMGPIX*SUBIMGPIX*args.nsamps*args.datasize + args.headersize #really just payload size
            maxbyteshex = (SUBIMGPIX*SUBIMGPIX*args.nsamps*args.datasize + args.headersize + 4)*2 + 404 #http header is 404m
    elif TXsubint and args.TXnints>1:
        if args.datasize%2 != 0:
            maxbytes = args.gridsize*args.gridsize*(args.nsamps//args.TXnints)*(args.datasize-1) + args.headersize #really just payload size
            maxbyteshex = (args.gridsize*args.gridsize*(args.nsamps//args.Txnints)*(args.datasize-1) + args.headersize + 4)*2 + 404 #http header is 404
        else:
            maxbytes = args.gridsize*args.gridsize*(args.nsamps//args.TXnints)*args.datasize + args.headersize #really just payload size
            maxbyteshex = (args.gridsize*args.gridsize*(args.nsamps//args.TXnints)*args.datasize + args.headersize + 4)*2 + 404 #http header is 404
        i=0
        for p in args.multiport[::-1]:
            portmapping[p]= i//16
            i+=1
            printlog(str((p,i//16)),output_file=processfile)
    else:
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

    if len(args.multiport)==0:
        #create socket
        printlog("creating socket...",output_file=processfile,end='')
        servSockD = socket.socket(socket.AF_INET, socket.SOCK_STREAM if args.protocol=='tcp' else socket.SOCK_DGRAM,0)
        printlog("Done!",output_file=processfile)    

        #bind to port number
        port = args.port
        printlog("binding to port " + str(port) + "...",output_file=processfile,end='')
        servSockD.bind(('', port))
        printlog("Done!",output_file=processfile)

        if args.protocol=='tcp':
            #listen for conections
            printlog("listening for connections...",output_file=processfile,end='')
            servSockD.listen(args.maxconnect)
            printlog("Made connection",output_file=processfile)
    else:
        printlog("Multiport mode")
        servSockD_list = []
        for ii in range(len(args.multiport)):
            #create socket
            printlog("creating socket...",output_file=processfile,end='')
            servSockD_list.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM if args.protocol=='tcp' else socket.SOCK_DGRAM,0))
            printlog("Done!",output_file=processfile)

            #bind to port number
            port = args.multiport[ii]
            printlog("binding socket " + str(ii) + " to port " + str(port) + "...",output_file=processfile,end='')
            servSockD_list[ii].bind(('',port))
            printlog("Done!",output_file=processfile)
            if args.protocol=='tcp':
                #listen for conections
                printlog("listening for connections...",output_file=processfile,end='')
                servSockD_list[ii].listen(args.maxconnect)
                printlog("Made connection",output_file=processfile)
            printlog("")
            #multiport_accepting[ii] = True

    #initialize a pool of processes for concurent execution
    #maxProcesses = 5
    #if "DASKPORT" in os.environ.keys() and QSETUP:
    #    executor = QCLIENT
    #else:
    if len(args.daskaddress)>0:
        printlog("Using DASK scheduler",output_file=processfile)
        executor = Client(args.daskaddress)#,serializers=["msgpack"],deserializers=["msgpack"])
        #search_executor = Client(args.daskaddress)
        #slowlock_ = Lock_DASK(client=executor)
    else:
        executor = ThreadPoolExecutor(args.maxProcesses)
        #search_executor = ThreadPoolExecutor(args.maxProcesses)
        slowlock_ = Lock()
        searchlock_ = Lock()
        sublock_ = Lock()
    #executor = Client(processes=False)#"10.41.0.254:8844")


    task_list = []
    task_timing = []
    multiport_task_list = []
    multiport_num_list = []
    dask_enabled = len(args.daskaddress)>0
    printlog("DASK ENABLED FLAG = " + str(dask_enabled),output_file=processfile)


    #if subimg or subint, store parse results before adding fullimg
    substage = dict()
    for i in range(16):
        substage[i] = dict()
        
    #create a thread for each port
    readtasks = []
    for ii in range(len(servSockD_list)):
        readtasks.append(executor.submit(udpreadtask,servSockD_list[ii],ii,servSockD_list[ii].getsockname()[1],maxbytes,
                                    maxbyteshex,args.timeout,args.chunksize,args.headersize,args.datasize,args.testh23,
                                    args.offline,args.protocol,args.udpchunksize,substage,TXsubimg,TXsubint,args.TXnints,portmapping,dask_enabled,None if dask_enabled else sublock_ ))

    #wait for tasks
    multiporttasks = []
    cnt=0

    if args.debug:
        tracemalloc.start()
        mallocloop = 0
        startmalloc = tracemalloc.take_snapshot()
        f = open(procserver_memory_file,"w")
        print("INITIAL MEMORY ALLOCATION",file=f)
        startstats = startmalloc.statistics('lineno')
        for i in range(len(startstats)):
            print(startstats[i],file=f)
        print("-"*20,file=f)
        print("")
        f.close()
    while True:
        printlog("\nFULLIMG_DICTS---------------------------",output_file=processfile)
        printlog(fullimg_dict,output_file=processfile)
        printlog(slow_fullimg_dict,output_file=processfile)
        printlog(imgdiff_fullimg_dict,output_file=processfile)
        printlog("----------------------------------------",output_file=processfile)
        printlog("SUBSTAGE==>"+str(substage[0].keys()),output_file=processfile)
        printlog("----------------------------------------",output_file=processfile)
        printlog("MULTIPORTTASKS==>"+str([multiporttasks[i].done() for i in range(len(multiporttasks))]),output_file=processfile)
        printlog("----------------------------------------",output_file=processfile)
        printlog("READTASKS RUNNING==>"+str([readtasks[i].running() for i in range(len(readtasks))]),output_file=processfile)
        printlog("----------------------------------------\n",output_file=processfile)
        packet_dict = ETCD.get_dict(ETCDKEY_PACKET) #dict()
        packet_dict["dropped"] = 0
        
        #try:
        tread=time.time()
        qobj=QQUEUE_UDP.get()
        if qobj[0]=='mport':
            corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port = qobj[1:]#QQUEUE_UDP.get()#timeout=0.1)
            printlog("MAIN GOT THIS FROM QUEUE:"+str(('mport',corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,port)),output_file=processfile)
            if corr_node != -1 and img_id_isot in timeout_isots:
                printlog("Data from timed-out image, ignoring",output_file=processfile)
                continue
            printlog("Submitting multiport task>>[TIME]"+str(time.time()),output_file=processfile)
            if dask_enabled:
                multiporttasks.append(executor.submit(multiport_task,corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,
                                    cnt,args.testh23,
                                    args.offline,args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,args.PyTorchDedispersion,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.fprtest,args.fnrtest,args.appendframe,args.DMbatches,
                                    args.SNRbatches,args.usejax,args.noiseth,args.nocutoff,args.realtime,args.nchans,None if dask_enabled else executor,#search_executor,
                                    args.slow,args.imgdiff,args.etcd,dask_enabled,args.attachmode,args.completeness,None if dask_enabled else slowlock_,None if dask_enabled else searchlock_,
                                    args.forfeit,args.rtastrocal,args.testsinglenode,False,False,1))
            else:
                multiporttasks.append(executor.submit(multiport_task,corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,
                                    cnt,args.testh23,
                                    args.offline,args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,args.PyTorchDedispersion,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.fprtest,args.fnrtest,args.appendframe,args.DMbatches,
                                    args.SNRbatches,args.usejax,args.noiseth,args.nocutoff,args.realtime,args.nchans,None if dask_enabled else executor,#search_executor,
                                    args.slow,args.imgdiff,args.etcd,dask_enabled,args.attachmode,args.completeness,None if dask_enabled else slowlock_,None if dask_enabled else searchlock_,
                                    args.forfeit,args.rtastrocal,args.testsinglenode,False,False,1))
            #if type(ret)==list:
            #    multiporttasks += list(ret)        
            packet_dict["read_time"] = time.time()-tread
            printlog("TOTAL READ TIME:"+str( time.time()-tread),output_file=processfile)
            ETCD.put_dict(ETCDKEY_PACKET,packet_dict)

        elif qobj[0]=='srch':
             
            
            srch_isot,srch_thash,srch_slow,srch_imgdiff=qobj[1:]#QQUEUE_SRCH.get()#timeout=0.1)
            printlog("MAIN GOT THIS FROM QUEUE:"+str(('srch',srch_isot,srch_thash,srch_slow,srch_imgdiff)),output_file=processfile)
            if srch_slow:

                srchimg = slow_fullimg_dict[srch_isot]
                srchRA_axis_idx = copy.deepcopy(srchimg.RA_axis)[srchimg.slow_RA_cutoff:]
            elif srch_imgdiff:
                srchimg = imgdiff_fullimg_dict[srch_isot]
                srchRA_axis_idx = copy.deepcopy(srchimg.RA_axis)[srchimg.imgdiff_RA_cutoff:]
            else:
                srchimg = fullimg_dict[srch_isot]
                srchRA_axis_idx = copy.deepcopy(srchimg.RA_axis) #copy.deepcopy(fullimg_array[idx].RA_axis)
            srchDEC_axis_idx= copy.deepcopy(srchimg.DEC_axis) #copy.deepcopy(fullimg_array[idx].DEC_axis)


            printlog("Starting search task>>[TIME]"+str(time.time()),output_file=processfile)
            stask = executor.submit(sl.search_task,searchlock_,srchimg,args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.fprtest,args.fnrtest,args.appendframe,args.DMbatches,
                                    args.SNRbatches,args.usejax,args.noiseth,args.nocutoff,args.realtime,srch_slow,srch_imgdiff,None,None,False,args.completeness,args.forfeit)
            stask.add_done_callback(lambda future: future_callback(future,args.SNRthresh,args.etcd,dask_enabled))


            multiporttasks += [stask]
            printlog("Started search task success",output_file=processfile)
        
        """
        multiporttasks_ = []
        for mt in multiporttasks:
            if not mt.done(): multiporttasks_.append(mt)
        multiporttasks = multiporttasks_
        """
        printlog("***GARBAGE COLLECTION>>"+str(gc.collect()),output_file=processfile)
        if args.debug:
            endstats = tracemalloc.take_snapshot().compare_to(startmalloc,'lineno')
            f = open(procserver_memory_file,"a")
            print("LOOP " + str(mallocloop) + "MEMORY ALLOCATION",file=f)
            for i in range(len(endstats)):
                print(endstats[i],file=f)
            print("-"*20,file=f)
            print("",file=f)
            f.close()
            startmalloc = tracemalloc.take_snapshot()
            mallocloop += 1



        """
        #check if any timed out
        printlog("MAINLOOP> TRYING TO ACQUIRE MUTEX...",output_file=processfile)
        slowlock_.acquire()
        printlog("MAINLOOP> ACQUIRED",output_file=processfile)
        try:
            for k in fullimg_dict.keys():
                if str(k) != img_id_isot and (not fullimg_dict[k].running) and fullimg_dict[k].is_timeout():
                    printlog("Image "+str(k)+" timed out, searching now",output_file=processfile)
                    fullimg_dict[k].running = True
                    timeout_isots.append(str(k))
                    if dask_enabled:
                        multiporttasks.append(search_executor.submit(multiport_task,-1,str(k),fullimg_dict[k].img_id_mjd,fullimg_dict[k].img_uv_diag,fullimg_dict[k].img_dec,fullimg_dict[k].shape[:-1],None,
                                    -cnt,args.testh23,
                                    args.offline,args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,args.PyTorchDedispersion,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.fprtest,args.fnrtest,args.appendframe,args.DMbatches,
                                    args.SNRbatches,args.usejax,args.noiseth,args.nocutoff,args.realtime,args.nchans,None if dask_enabled else search_executor,
                                    args.slow,args.imgdiff,args.etcd,dask_enabled,args.attachmode,args.completeness,None if dask_enabled else slowlock_,None if dask_enabled else searchlock_,
                                    args.forfeit,args.rtastrocal,args.testsinglenode,False,False,1))
                    else:
                        multiporttasks.append(search_executor.submit(multiport_task,-1,str(k),fullimg_dict[k].img_id_mjd,fullimg_dict[k].img_uv_diag,fullimg_dict[k].img_dec,fullimg_dict[k].shape[:-1],None,
                                    -cnt,args.testh23,
                                    args.offline,args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,args.PyTorchDedispersion,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.fprtest,args.fnrtest,args.appendframe,args.DMbatches,
                                    args.SNRbatches,args.usejax,args.noiseth,args.nocutoff,args.realtime,args.nchans,None if dask_enabled else search_executor,
                                    args.slow,args.imgdiff,args.etcd,dask_enabled,args.attachmode,args.completeness,None if dask_enabled else slowlock_,None if dask_enabled else searchlock_,
                                    args.forfeit,args.rtastrocal,args.testsinglenode,False,False,1))
                    #multiport_num_list.append(ii)    
            cnt +=1
        except Exception as exc:
            printlog("|EXCEPTION| MAINLOOP>" + str(exc),output_file=processfile)
            slowlock_.release()
            break
        printlog("MAINLOOP> TRYING TO RELEASE MUTEX...",output_file=processfile)
        slowlock_.release()
        printlog("MAINLOOP> RELEASED",output_file=processfile)

        multiporttasks_ = []
        for mt in multiporttasks:
            if not mt.done(): multiporttasks_.append(mt)
        multiporttasks = multiporttasks_
        """

        """
        for k in slow_fullimg_dict.keys():
            if (not slow_fullimg_dict[k].running) and slow_fullimg_dict[k].is_timeout():
                printlog("Removing slow image:"+str(k),output_file=processfile)
                del slow_fullimg_dict[k]
        for k in imgdiff_fullimg_dict.keys():
            if (not imgdiff_fullimg_dict[k].running) and imgdiff_fullimg_dict[k].is_timeout():
                printlog("Removing imgdiff image:"+str(k),output_file=processfile)
                del imgdiff_fullimg_dict[k]
        """
    executor.shutdown()
    search_executor.shutdown()
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
    parser.add_argument('--PyTorchDedispersion',action='store_true',help='[Deprecated] Uses GPU-accelerated dedispersion code from https://github.com/nkosogor/PyTorchDedispersion')
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
    parser.add_argument('--pixperFWHM',type=float,help='Pixels per FWHM, default 3',default=pixperFWHM)
    parser.add_argument('--multiport',nargs='+',default=[],help='List of port numbers to listen on, default using single port specified in --port',type=int)
    parser.add_argument('--imgdiffgulps',type=int,help='Number of gulps to search at a time with image differencing, default=' + str(config.ngulps_per_file),default=config.ngulps_per_file)
    parser.add_argument('--slow',action='store_true',help='Activate slow search pipeline, which bins data by 5 samples and re-searches')
    parser.add_argument('--imgdiff',action='store_true',help='Activate image differencing search pipeline, which bins data by 25 samples and searches 5-minute chunk at DM=0')
    parser.add_argument('--daskaddress',type=str,help='tcp address of dask scheduler, default does not use scheduler',default="")
    parser.add_argument('--rttimeout',type=float,help='time to wait for search task to complete before cancelling, default=3 seconds',default=3)
    parser.add_argument('--attachmode',action='store_true',help='in attached mode, search tasks for slow and image diff pipelines are combined with normal pipeline to minimize overheads')
    parser.add_argument('--completeness',action='store_true',help='Run a completeness assessment by sending images to the process server and testing recovery')
    parser.add_argument('--forfeit',action='store_true',help='Forfeit searching base resolution data gulp to search slow/imgdiff data; forfeit searching slow data gulp to search imgdiff data; superceded by attach mode')
    parser.add_argument('--rtastrocal',action='store_true',help='Save data for astrometric and flux calibration')
    parser.add_argument('--testsinglenode',action='store_true',help='Receive data from only one corr node and duplicate across 16 nodes')
    parser.add_argument('--TXmode',type=str,choices=['subimg','subint','base'],default='base',help='TX mode')
    parser.add_argument('--TXnints',type=int,help='Number of sub-integrations for TXmode subint',default=5)
    parser.add_argument('--protocol',choices=['tcp','udp'],default='tcp',help='protocol to use to send data to process server,default=tcp')
    parser.add_argument('--udpchunksize',type=int,help='Data chunksize in bytes,default=25886',default=25886)
    parser.add_argument('--debug',action='store_true',help='memory debugging')
    args = parser.parse_args()

    """
    if len(args.daskaddress)>0:
        print("Connecting to dask scheduler "+args.daskaddress)
        client = Client(args.daskaddress)
        client.submit(main,args,pure=False) 
    else:
    """
    main(args)
