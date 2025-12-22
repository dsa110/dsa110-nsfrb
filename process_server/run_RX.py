import numpy as np
from realtime import rtwriter
from threading import Lock, Timer
import glob
import json
from multiprocessing import Manager
from nsfrb.planning import get_RA_cutoff
from threading import Lock
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
from astropy.coordinates import SkyCoord
from astropy import units as u
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
from nsfrb import cerberus as jax_funcs #jax_funcs
"""s
Directory for output data
"""
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,Lon,Lat,tsamp_slow,bin_slow,pixperFWHM,output_file,bin_imgdiff,sslogfile,table_dir

"""
NSFRB modules
"""
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack 
from nsfrb.imaging import uv_to_pix,stack_images
from nsfrb.candcutting import is_injection
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
ETCDKEY_CORRSTAGGER = f'/mon/nsfrbstagger'



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





ECODE_BREAK = -1
ECODE_CONT = -2
ECODE_SUCCESS = 0
nextdata = dict()
def readcorrdata(servSockD,ii,port,maxbytes,maxbyteshex,timeout_SOCKET,chunksize,headersize,datasize,testh23,offline,protocol,udpchunksize,readlock,timeout_SELECT,timeout_LOOP,overdraw,args,dtype,dask_enabled,search_executor,searchlock,slowlock,timeout_SLEEP,timeout_INLOOP):

    t_ = time.time()
    socksuffix = "SOCKET " + str(ii) + " >>"
    printlog(socksuffix + "STARTUP, WAITING FOR LOCK ON "+str(port),output_file=processfile)
    #readlocks_[port].acquire()
    printlog(socksuffix + "ACQUIRED LOCK ON "+str(port),output_file=processfile)
    if protocol=='udp':
        lastbyte=-1
        recdatabytes=bytes(0)
        nchunks = int(maxbytes//udpchunksize)
        nrec = int((maxbytes//nchunks) + headersize) #header includes byte number[8], host[16], isot[23],hdrarray[24]
        servSockD.settimeout(timeout_SOCKET)
        while len(recdatabytes)<maxbytes-headersize:
            if len(recdatabytes)==0:
                printlog(socksuffix + "start UDP read...",output_file=processfile,end='')
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
                        #readlocks_[port].release()
                        return ECODE_BREAK
                elif 'Assertion' in str(exc):
                    printlog(socksuffix + "Bad data order:"+str(exc),output_file=processfile)

                if len(recdatabytes)<maxbytes-headersize:
                    recdatabytes += bytes(int((nrec-headersize)))
                lastbyte+=1
        shape = np.array([shape_0,shape_1,shape_2,shape_3],dtype=int)
        shape = tuple(shape[shape!=0])
        arrData = np.frombuffer(recdatabytes,dtype=np.float64).reshape(shape)
        port = int(host[host.index(":")+1:])
        img_id_mjd=Time(img_id_isot,format='isot').mjd
        printlog(socksuffix +"Done, read "+str(len(recdatabytes))+"/"+str(maxbytes-headersize)+" into array shaped "+str(shape),output_file=processfile)
        #readlocks_[port].release()
        return corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port
    printlog(socksuffix + "accepting connection...",output_file=processfile,end='')
    clientSocket,address = servSockD.accept()
    clientSocket.setblocking(0)
    
    corr_address, tmp = clientSocket.getpeername()
    if corr_address not in corraddrs:
        clientSocket.close()
        printlog(socksuffix + "BAD ADDRESS:"+str(corr_address),output_file=processfile)
        #readlocks_[port].release()
        return ECODE_BREAK
    printlog(socksuffix + "client: " + str(corr_address) + "...",output_file=processfile,end='')
    recstatus = 1
    fullMsg = ""
    printlog(socksuffix + "Done!",output_file=processfile)
    printlog(socksuffix + "Receiving data...",output_file=processfile)

    #set timeout and expected number of bytes to read
    clientSocket.settimeout(timeout_SOCKET)
    totalbytes = 0
    pflag = 0

    #while (recstatus> 0) and (totalbytes < maxbytes):#+maxbytesaddr):
    t_timeout = time.time()
    t_startread = time.time()
    totalbyteshex =0
    printlog(nextdata,output_file=processfile)
    if overdraw and corr_address in nextdata.keys():
        printlog("FOUND LEFTOVER DATA|"+str(corr_address) + "|"+str(len(nextdata[corr_address][0])),output_file=processfile)
        (strData, ancdata, msg_flags, address)=nextdata[corr_address]
        recstatus = len(strData)
        fullMsg += strData.hex()
        totalbytes += recstatus
        totalbyteshex += len(strData.hex())
        del nextdata[corr_address]
        printlog("DONE LEFTOVER DATA-->"+str(totalbyteshex)+"/"+str(maxbyteshex),output_file=processfile)
    readiters=0
    #wait until corrstagger flag set
    """while not ETCD.get_dict(ETCDKEY_CORRSTAGGER)['status'][(ii-1)%16] and time.time()-t_startread<timeout_SOCKET:
        printlog(socksuffix+"waiting for data...",output_file=processfile)
        time.sleep(timeout_SLEEP)
    if time.time()-t_startread<timeout_LOOP: printlog(socksuffix+"READ NOW!",output_file=processfile)
    else: 
        printlog(socksuffix+"NO TIME TO READ!",output_file=processfile)
        clientSocket.close()
        printlog(socksuffix+"TOTAL READ+ERROR TIME:"+str(time.time()-t_)+"s",output_file=processfile)
        #readlocks_[port].release()
        return ECODE_BREAK # break
    """
    #readlock.acquire()
    t_timeout = time.time()
    t_startread = time.time()
    while (totalbyteshex < maxbyteshex) and time.time()-t_startread<timeout_LOOP and time.time()-t_<timeout_INLOOP:# and time.time()-t_timeout<args.timeout:
        #printlog(socksuffix + "NLOOP>>"+str(readiters),output_file=processfile)
        readiters+=1
        try:
            #check if data is ready to read first
            t_ready = time.time()

            
            #readlock.acquire()
            """
            while not select.select([clientSocket],[],[],timeout_SELECT) and time.time()-t_ready<timeout_LOOP:
                continue
            if not select.select([clientSocket],[],[],timeout_SELECT):
                raise socket.timeout
            """
            printlog(socksuffix+ "Data ready at time T="+str(time.time()-t_ready),output_file=processfile)

           
            (strData, ancdata, msg_flags, address) = clientSocket.recvmsg(chunksize)#255)
            recstatus = len(strData)
            printlog(socksuffix+"Read "+ str(recstatus) + " bytes, total "+ str(totalbytes+recstatus) + "|"+str(time.time()-t_ready),output_file=processfile)
            if recstatus > 0:
                t_timeout = time.time()
            #if recstatus == 0 and time.time()-t_timeout>timeout_INLOOP:
            #    raise socket.timeout
            
            elif recstatus == 0:
                time.sleep(timeout_SLEEP)
                if totalbyteshex == 0:
                    printlog(socksuffix + "STILL WAITING...",output_file=processfile)
                    #t_startread = time.time()
                continue
            
            #readlock.release()
            #printlog(socksuffix+"Read "+ str(recstatus) + " bytes, total "+ str(totalbytes+recstatus) + "|"+str(time.time()-t_ready),output_file=processfile)
            printlog(socksuffix+"Message flags:" + str(msg_flags),output_file=processfile)
            printlog(socksuffix+"AncData:" + str(ancdata),output_file=processfile)
            if overdraw and totalbyteshex + len(strData.hex())>maxbyteshex:

                printlog(socksuffix+"OVERDRAWOVERDRAW-->HAVE "+str(totalbyteshex)+" KEEPING "+str(maxbyteshex - totalbyteshex) +" OUT OF "+str(len(strData.hex())),output_file=processfile)
                printlog(socksuffix+"OVERDRAWOVERDRAW-->HAVE "+str(totalbytes)+" KEEPING "+str(maxbytes - totalbytes) +" OUT OF "+str(len(strData)),output_file=processfile)
                nextdata[corr_address] = (strData[int(maxbytes - totalbytes):],ancdata,msg_flags,address)
                fullMsg += strData[:int(maxbytes - totalbytes)].hex()
                totalbyteshex += len(strData[:int(maxbytes - totalbytes)])*2
                totalbytes += len(strData[:int(maxbytes - totalbytes)])
                printlog(socksuffix+"OVERDRAWOVERDRAW-->SAVETHEREST "+str(totalbyteshex),output_file=processfile)
            else:
                fullMsg += strData.hex()
                totalbytes += recstatus
                totalbyteshex += len(strData.hex())
            #don't know how long the header is, so don't start counting until hit NP data
            if "93" in fullMsg:
                printlog(socksuffix+"Found start byte at index " + str(fullMsg.index("93")),output_file=processfile)
                totalbytes = (len(fullMsg) - fullMsg.index("93"))//2
                #totalbyteshex = totalbytes*2
        except Exception as ex:
            if type(ex) == socket.timeout:
                printlog(socksuffix+"Timed out after reading " + str(totalbytes) + " bytes; proceeding...",output_file=processfile)
                break
            else:
                #readlocks_[port].release()
                printlog(socksuffix+"I'm about to scream",output_file=processfile)
                raise
    printlog(socksuffix+"Done! Total bytes read:" + str(totalbytes),output_file=processfile)
    totalbytessend = 0
    #readlock.release()

    #check if data is the size we expect
    try:
        assert(totalbyteshex>=maxbyteshex)
    except AssertionError as exc:
        """readlock.acquire()
        printlog(socksuffix + "READLOCK ACQUIRED",output_file=processfile)
        tmp_ = np.zeros(16)
        tmp_[corraddrs[corr_address]] += 1
        np.save(config.table_dir + "/TCPHELPMONITOR.npy",np.load(config.table_dir + "/TCPHELPMONITOR.npy")+tmp_)
        readlock.release()"""
        printlog(socksuffix + "READLOCK RELEASED",output_file=processfile)
        printlog(socksuffix+"Invalid data size, " + str(totalbytes) + " received when expected at least " + str(maxbytes) + ": " + str(exc),output_file=processfile)

        printlog(socksuffix+"Invalid data size, " + str(totalbyteshex) + " received when expected at least " + str(maxbyteshex) + ": " + str(exc),output_file=processfile)
        printlog(socksuffix+"Setting truncated data size flag...",output_file=processfile,end='')

        pflag = set_pflag_loc("datasize_error")
        if pflag == None:
            clientSocket.close()
            printlog(socksuffix+"Error setting flags, abort",output_file=processfile)
            printlog(socksuffix+"TOTAL READ+ERROR TIME:"+str(time.time()-t_)+"s",output_file=processfile)
            #readlocks_[port].release()
            return ECODE_BREAK # break
        printlog(socksuffix+"Done, continue",output_file=processfile)
    if pflag != 0:
        successmsg = bytes(success + str(pflag) + '\n','utf-8')
        printlog(socksuffix+"Sending response...",output_file=processfile,end='')
        while (totalbytessend < len(successmsg)):
            totalbytessend += clientSocket.send(successmsg)
        clientSocket.close()
        printlog(socksuffix+"Done! Total bytes sent:" + str(totalbytessend) + "/" + str(len(successmsg)),output_file=processfile)
        printlog(socksuffix+"TOTAL READ+ERROR TIME:"+str(time.time()-t_)+"s",output_file=processfile)
        #readlocks_[port].release()
        return ECODE_CONT#continue

    #try to parse to get address
    try:
        corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData = parse_packet(fullMsg=fullMsg,maxbytes=maxbytes,headersize=headersize,datasize=datasize,port=port,corr_address=corr_address,testh23=testh23)
        printlog("PARSE SUCCESS:"+str((corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape)),output_file=processfile)
    except Exception as exc:
        """readlock.acquire()
        printlog(socksuffix + "READLOCK ACQUIRED",output_file=processfile)
        print(corr_address)
        tmp_ = np.zeros(16)
        tmp_[corraddrs[corr_address]] += 1
        np.save(config.table_dir + "/TCPHELPMONITOR.npy",np.load(config.table_dir + "/TCPHELPMONITOR.npy")+tmp_)
        readlock.release()"""
        printlog(socksuffix + "READLOCK RELEASED",output_file=processfile)

        if type(exc) == UnicodeDecodeError:
            printlog(socksuffix+"Error parsing data: " + str(exc),output_file=processfile)
            printlog(socksuffix+"Setting parse error flag...",output_file=processfile,end='')
            pflag = set_pflag_loc("parse_error")
            if pflag == None:
                clientSocket.close()
                printlog(socksuffix+"Error setting flags, abort",processfile=processfile)
                printlog(socksuffix+"TOTAL READ+ERROR TIME:"+str(time.time()-t_)+"s",output_file=processfile)
                #readlocks_[port].release()
                return ECODE_BREAK #break
            printlog(socksuffix+"Done, continue",output_file=processfile)
        if type(exc) == ValueError:
            printlog(socksuffix+"Invalid data size: " + str(exc),output_file=processfile)
            printlog(socksuffix+"Setting datasize flag...",output_file=processfile,end='')
            pflag = set_pflag_loc("datasize_error")
            if pflag == None:
                clientSocket.close()
                printlog(socksuffix+"Error setting flags, abort",processfile=processfile)
                printlog(socksuffix+"TOTAL READ+ERROR TIME:"+str(time.time()-t_)+"s",output_file=processfile)
                #readlocks_[port].release()
                return ECODE_BREAK #break
            printlog(socksuffix+"Done, continue",output_file=processfile)
        else:
            clientSocket.close()
            #readlocks_[port].release()
            raise exc

    successmsg = bytes(success + str(pflag) + '\n','utf-8')
    printlog(socksuffix+"Sending response...",output_file=processfile,end='')
    while (totalbytessend < len(successmsg)):
        totalbytessend += clientSocket.send(successmsg)
    printlog(socksuffix+"Done! Total bytes sent:" + str(totalbytessend) + "/" + str(len(successmsg)),output_file=processfile)
    if pflag != 0:
        clientSocket.close()
        printlog(socksuffix+"TOTAL READ+ERROR TIME:"+str(time.time()-t_)+"s",output_file=processfile)
        #readlocks_[port].release()
        return ECODE_CONT #continue
    printlog(socksuffix+"Data: " + str(arrData),output_file=processfile)

    #reopen
    printlog(socksuffix+"TOTAL READ TIME:"+str(time.time()-t_)+"s",output_file=processfile)



    #PACK mode --> "pack" read and gather tasks into same task
    clientSocket.close()
    #readlocks_[port].release()
    
    """
    if args.pack:
        if img_id_isot not in slowlocks_.keys():
            slowlocks_[img_id_isot] = Lock()
        multiport_subtask(corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,
                                    ii,args.testh23,
                                    args.offline,args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,args.PyTorchDedispersion,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.fprtest,args.fnrtest,args.appendframe,args.DMbatches,
                                    args.SNRbatches,args.usejax,args.noiseth,args.nocutoff,args.realtime,args.nchans,None if dask_enabled else search_executor,
                                    args.slow,args.imgdiff,args.etcd,dask_enabled,args.attachmode,args.completeness,None if dask_enabled else slowlock,
                                    None if dask_enabled else searchlock,args.forfeit,args.rtastrocal,args.testsinglenode,False,False,1,dtype,args.lockdev,args.rejectnoiseoutliers)
    
    """
    #printlog(socksuffix+"--RET--"+str((corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape)),output_file=processfile)
    #imagetoDADA(corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port,key=config.NSFRB_SRCHDADA_KEY)
    printlog(socksuffix+"TOTAL READ+GATHER TIME:"+str(time.time()-t_)+"s",output_file=processfile)
    return corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port

multiport_accepting = dict()
appendinit = False

from realtime import rtwriter
from realtime import rtreader
def imagetoDADA(corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port,key=config.NSFRB_SRCHDADA_KEY):
    numdata = np.array([corr_node,img_id_mjd,img_uv_diag,img_dec,port]+list(shape),dtype=np.float64)
    alldata = np.concatenate([numdata,arrData.flatten()])
    rtwriter.rtwrite(alldata,key=key,addheader=False,dtype=np.float64) #bytes(numdata) + bytes(img_id_isot,encoding='utf-8') + bytes(arrData)
    return


from multiprocessing import Queue
QQUEUE = Queue()
ETCDKEY_CORRSTAGGER = f'/mon/nsfrbstagger'
def etcd_to_stagger(queue=QQUEUE):
    """
    This is a callback function that waits for previous corr node to send data
    """
    etcd_dict = ETCD.get_dict(ETCDKEY_CORRSTAGGER)
    if etcd_dict is None:
        return
    #if ((sb>0 and (etcd_dict['status'][sb-1] and not etcd_dict['status'][sb])) or
    #    (sb==0 and np.all(np.array(etcd_dict['status'])))):
    QQUEUE.put(etcd_dict['status'])
    return


def main(args):

    #ETCD.add_watch(ETCDKEY_CORRSTAGGER, etcd_to_stagger)
    np.save(config.table_dir + "/TCPHELPMONITOR.npy",np.zeros(16))
    #redirect stderr
    sys.stderr = open(error_file,"w")
    
    #if "DASKPORT" in os.environ.keys():
    #    printlog("Using Dask Scheduler on Port " + str(os.environ['DASKPORT']) + " for cand_cutter queue",output_file=processfile)
    if args.etcd:
        printlog("Etcd enabled, will push candidates to " + ETCDKEY,output_file=processfile)

    


    printlog("USEFFT = " + str(args.usefft),output_file=processfile)
    TXsubimg = args.TXmode=='subimg'
    TXsubint = args.TXmode=='subint'

    printlog("TX MODE:"+str((args.TXmode,TXsubimg,TXsubint,args.TXnints)),output_file=processfile)
    printlog("SAMPS:"+str((args.nsamps,args.TXnints,args.nsamps//args.TXnints)),output_file=processfile)

    #total expected number of bytes for each sub-band image
    if args.datasize==4:
        dtype = np.float32
    elif args.datasize==2:
        dtype = np.float16
    elif args.datasize==8:
        dtype = np.float64
    elif args.datasize==16:
        dtype = np.float128
    
    printlog(dtype,output_file=processfile)
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
    """
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
        search_executor = ThreadPoolExecutor(args.maxProcesses)
        read_executor = ThreadPoolExecutor(args.maxProcesses)
        slowlock_ = Lock()
        searchlock_ = Lock()
        readlock_ = Lock()
    #executor = Client(processes=False)#"10.41.0.254:8844")
    """

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
        
    readtasks = []
    TSTARTUP = time.time()
    initflag = False
    readsockets = []
    badcounter = 0
    failsafe_timer = time.time()
    failsafe_counter =0
    skipcorrs_bad = []
    skipcorrs_prev = []
    TNEXT=time.time()
    while True: # want to keep accepting connections
        skipcorrs_prev = skipcorrs_bad
        skipcorrs_bad = []
        print((skipcorrs_prev,skipcorrs_bad))
        if failsafe_counter>=3 and ((time.time()-failsafe_timer)<args.timeout_FAILSAFE): 
            printlog("<"+str((time.time()-failsafe_timer))+">FAILSAFE TRIGGERED, RESTARTING...",output_file=processfile)
            
            for s in servSockD_list: s.close()
            raise RuntimeError("Failsafe condition reached, restarting...")
        elif failsafe_counter>=3 and ((time.time()-failsafe_timer)>=args.timeout_FAILSAFE):
            printlog("FAILSAFE CHECKUP",output_file=processfile)
            failsafe_counter=0
            failsafe_timer=time.time()

        fulldata = np.nan*np.ones((args.gridsize,args.gridsize,args.nsamps,16))
        packet_dict = ETCD.get_dict(ETCDKEY_PACKET) #dict()
        packet_dict["dropped"] = 0
        #SELECT
        printlog(">>>LOOP HERE>>>"+str(time.time()-TSTARTUP),output_file=processfile)
        t0=time.time()
        if False:#badcounter >= 16*args.BADITERS:
            printlog("BAD COUNTER FAILSAFE TRIGGERED, RESTARTING...",output_file=processfile)
            for s in servSockD_list: s.close()
            raise RuntimeError("Failsafe condition reached, restarting...")
        printlog("DATAITER>>>"+str(len(readsockets)),output_file=processfile)
        while len(readsockets)<1 and ((not initflag) or (time.time() - t0 < args.timeout_RESTART)):#16 and (time.time()-t0 < (config.tsamp*config.nsamps/1000)):# and (not initflag or (time.time()-TSTARTUP)<3.1):
            printlog("DATAITER>>>"+str(len(readsockets)),output_file=processfile)
            readsockets,writesockets,errsockets = select.select(servSockD_list[0:1],[],[],args.timeout_SELECT)
        if not initflag: t0 = time.time()
        if (time.time() - t0 >= args.timeout_RESTART):
            printlog(">>>RESTART HAPPENING, WAIT",output_file=processfile)
            time.sleep(600)
            for s in servSockD_list: s.close()
            raise RuntimeError("Failsafe condition reached, restarting...")
        """
        while np.any(QQUEUE.get()):
            continue
        printlog("DATAITER>>>"+str(len(readsockets)),output_file=processfile)
        
        tmpstatus = [False]*16
        while not tmpstatus[0] and np.any(tmpstatus[1:]):
            tmpstatus = QQUEUE.get()
            continue
        printlog("DATAITER>>>"+str(tmpstatus),output_file=processfile)
        """

        
        
        printlog("DATAITER>>>DONE",output_file=processfile)
        printlog("TOTALTIMEINITIAL:"+str(time.time()-TNEXT),output_file=processfile)
        TSTARTUP = time.time()
        #printlog("Data ready on "+str(readsockets)+" ports",output_file=processfile)
        #printlog("Data ready on "+str(len(readsockets))+" ports",output_file=processfile)

        repret = None
        for ii in range(len(servSockD_list)):#readsockets)):
            printlog("ALLSKIPS:"+str(args.skipcorrs),output_file=processfile)
            #if time.time()-TSTARTUP < 1.0*(ii/16): #1.28*((ii+0.8)/16):
            #    printlog("SKIPPING THE REST...",output_file=processfile)
            #    time.sleep(1.0*(16 - ii)/16)
            #    break
            if (ii not in args.skipcorrs) and (ii not in skipcorrs_prev):
                tsel =time.time()
                """
                printlog("WAITDATA>>>"+str(ii)+"--",output_file=processfile)
                tmpstatus = [False]*16
                while not tmpstatus[ii] and (ii==15 or np.any(tmpstatus[ii+1:])):
                    tmpstatus = QQUEUE.get()
                    continue
                """
                #printlog("GOTDATA1>>>"+str(ii)+"--"+str(ETCD.get_dict(ETCDKEY_CORRSTAGGER)),output_file=processfile)
                readsockets = []
                printlog("WAITDATA>>>",output_file=processfile)
                while len(readsockets)<1 and (time.time()-TSTARTUP < args.timeout_RESTART):
                    readsockets,writesockets,errsockets = select.select(servSockD_list[ii:ii+1],[],[],args.timeout_SELECT)
                if ii!=0 and (time.time() - TSTARTUP >= args.timeout_RESTART):
                    printlog(">>>RESTART HAPPENING, WAIT",output_file=processfile)
                    repret = None
                    for s in servSockD_list: s.close()
                    raise RuntimeError("Failsafe condition reached, restarting...")
                    #break
                if ii==0:
                    TSTARTUP = time.time()
                printlog("ITERDATA>>>"+str(readsockets)+"| "+str(time.time()-tsel)+" s",output_file=processfile)
                if len(readsockets)==1: 
                    ret = readcorrdata(readsockets[0],ii,readsockets[0].getsockname()[1],maxbytes,
                                    maxbyteshex,args.timeout_SOCKET,args.chunksize,args.headersize,args.datasize,args.testh23,
                                    args.offline,args.protocol,args.udpchunksize,None,args.timeout_SELECT,args.timeout_LOOP,args.overdraw,
                                    args,dtype,dask_enabled,None,None,None,args.timeout_SLEEP,args.timeout_INLOOP)
                    #if ii==0 and type(ret)==int:
                    #   printlog("<<<<SKIP FULL SET>>>>",output_file=processfile)
                    #    break
            
                    if type(ret) != int: 
                        fulldata[:,:,:,int(ret[0])] = ret[-2]
                        repret = ret#retdats.append(ret)
                        printlog("|ITERDATA MJD>>>"+ret[1]+" "+str(ret[2]),output_file=processfile)

                    else:
                        badcounter += 1
                        #skipcorrs_bad.append(ii)
                        #time.sleep(args.timeout_SELECT)
                        """
                        if args.timeout_INLOOP >= (config.tsamp/1000):
                            for s in servSockD_list:
                                s.close()
                            printlog("RELAX TIME FAILSAFE TRIGGERED, RESTARTING...",output_file=processfile)
                            raise RuntimeError("Failsafe condition reached, restarting...")
                        else:
                            args.timeout_INLOOP *= 1.1 
                            printlog("relaxing select timeout:"+str(args.timeout_SELECT),output_file=processfile)
                        """
                #printlog("GOTDATA2>>>"+str(ii)+"--"+str(ETCD.get_dict(ETCDKEY_CORRSTAGGER)),output_file=processfile)
            else:
                time.sleep(args.timeout_INLOOP)
                printlog("SKIPCORR--"+str(ii),output_file=processfile)

        printlog("TOTALTIMEFINAL:"+str(time.time()-TSTARTUP),output_file=processfile)
        TNEXT=time.time()
        if repret is not None:
            corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port = repret
            imagetoDADA(corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,fulldata,port)
        elif (time.time() - TSTARTUP < args.timeout_RESTART):
            printlog(">NO DATA FOUND, FAILSAFE TRIGGERED, RESTARTING...",output_file=processfile)

            for s in servSockD_list: s.close()
            raise RuntimeError("Failsafe condition reached, restarting...")
        else:
            continue
        initflag = True
        
        readtasks = [] #newreadtasks
        readsockets= []    

        printlog("Data done on "+str(readsockets)+" ports",output_file=processfile)
        printlog("Data done on "+str(len(readsockets))+" ports",output_file=processfile)
        failsafe_counter+=1
    #executor.shutdown()
    #search_executor.shutdown()
    #read_executor.shutdown()
    #clientSocket.close()



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
    parser.add_argument('--timeout_FAILSAFE',type=float,help='Minimum timespan within which 5 reads should be completed before restarting',default=config.tsamp*config.nsamps*5/1000/2)
    parser.add_argument('--timeout_SELECT',type=float,help='Timeout for \'select.select()\' call to block until a socket has data ready; default =0 for polling',default=0)
    parser.add_argument('--timeout_BADCOUNTER',type=float,help='Time to wait after 10 bad reads',default=30)
    parser.add_argument('--timeout_SOCKET',type=float,help='Timeout for \'socket.settimeout()\' call to block until a socket receives data; default=0 for non-blocking',default=0)
    parser.add_argument('--timeout_LOOP',type=float,help='Timeout for TCP read data loop for full data chunk (175x175x25 image); default=3.35',default=3.35)
    parser.add_argument('--timeout_INLOOP',type=float,help='Timeout for TCP read data loop for full data chunk (175x175x25 image); default=3.35',default=0.15)
    parser.add_argument('--timeout',type=float,help='Max time in seconds to wait for more data to be ready to receive, default = 1',default=1)
    parser.add_argument('--timeout_TASK',type=float,help='Timeout for \'executor.submit()\' call to wait for read tasks to complete',default=None)
    parser.add_argument('--timeout_SLEEP',type=float,help='Timeout to wait if no data is available so that other tasks are given priority',default=0)
    parser.add_argument('--timeout_RESTART',type=float,help='Timeout to wait that indicates a system restart is happening',default=10)
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
    #parser.add_argument('--rttimeout',type=float,help='time to wait for search task to complete before cancelling, default=3 seconds',default=3) #--> no longer used b/c timeout set by integration time
    parser.add_argument('--attachmode',action='store_true',help='in attached mode, search tasks for slow and image diff pipelines are combined with normal pipeline to minimize overheads')
    parser.add_argument('--completeness',action='store_true',help='Run a completeness assessment by sending images to the process server and testing recovery')
    parser.add_argument('--forfeit',action='store_true',help='Forfeit searching base resolution data gulp to search slow/imgdiff data; forfeit searching slow data gulp to search imgdiff data; superceded by attach mode')
    parser.add_argument('--rtastrocal',action='store_true',help='Save data for astrometric and flux calibration')
    parser.add_argument('--testsinglenode',action='store_true',help='Receive data from only one corr node and duplicate across 16 nodes')
    parser.add_argument('--TXmode',type=str,choices=['subimg','subint','base'],default='base',help='TX mode')
    parser.add_argument('--TXnints',type=int,help='Number of sub-integrations for TXmode subint',default=5)
    parser.add_argument('--protocol',choices=['tcp','udp'],default='tcp',help='protocol to use to send data to process server,default=tcp')
    parser.add_argument('--udpchunksize',type=int,help='Data chunksize in bytes,default=25886',default=25886)
    parser.add_argument('--lockdev',type=int,help='Locks all search tasks to a single GPU, 0 or 1',default=-1)
    parser.add_argument('--rtastrocalrange',type=int,help='Range in gulps to save images',default=15)
    parser.add_argument('--rejectnoiseoutliers',action='store_true',help='if set, does not add noise estimates to standard deviation if difference from current noise is >3sigma (unless not initializes)')
    parser.add_argument('--wait',action='store_true',help='wait for read tasks to finish')
    parser.add_argument('--overdraw',action='store_true',help='checks if data was read into the next chunk')
    parser.add_argument('--pack',action='store_true',help='pack together read and gather tasks')
    parser.add_argument('--maxloops',type=int,help='max number of read attempts before giving up',default=100)
    parser.add_argument('--psrdadakey',type=str,help='output to psrdada buffer with the specified key; default='+str(config.NSFRB_SRCHDADA_KEY),default="")#config.NSFRB_SRCHDADA_KEY)
    parser.add_argument('--BADITERS',type=float,help='Max number of poor iterations before regroup',default=5)
    parser.add_argument('--skipcorrs',nargs='+',default=[],help='corr nodes we know are lagging, so we skip',type=int)
    args = parser.parse_args()

    """
    if len(args.daskaddress)>0:
        print("Connecting to dask scheduler "+args.daskaddress)
        client = Client(args.daskaddress)
        client.submit(main,args,pure=False) 
    else:
    """
    main(args)
