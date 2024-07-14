import numpy as np
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

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()


import sys
sys.path.append(cwd + "/") #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
import csv
import copy

from nsfrb.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset
from nsfrb.noise import init_noise
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
from nsfrb import searching as sl
from nsfrb import pipeline
from nsfrb import plotting as pl
from nsfrb import config
from nsfrb import jax_funcs
"""s
Directory for output data
"""
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = cwd + "/src/.pipestatus.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = cwd + "/scripts/script_flags/searchlog_flags.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_flags/searchlog_flags.txt"
output_file = cwd + "-logfiles/run_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
processfile = cwd + "-logfiles/process_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt"
flagfile = cwd + "/process_server/process_flags.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_flags.txt"
cand_dir = cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
error_file = cwd + "-logfiles/error_log.txt"
"""
Arguments: data file
"""
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack 
from nsfrb.imaging import uv_to_pix

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
    def __init__(self,img_id_isot,img_id_mjd,shape=(32,32,25,16),dtype=np.float16):
        self.image_tesseract = np.zeros(shape,dtype=dtype)
        self.corrstatus = np.zeros(16,dtype=bool)
        self.img_id_isot = img_id_isot
        self.img_id_mjd = img_id_mjd
        self.shape = shape
        #get ra and dec axes
        self.RA_axis,self.DEC_axis = uv_to_pix(self.img_id_mjd,self.shape[0],Lat=37.23,Lon=-118.2851)
    
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
    img_id_isot = HTTPheaderMsgStr[HTTPheaderMsgStr.index('IMG')+3:HTTPheaderMsgStr.index('.npy')]
    img_id_mjd = Time(img_id_isot,format='isot').mjd
    #corr_address = address#HTTPheaderMsgStr[HTTPheaderMsgStr.index('Host')+6:HTTPheaderMsgStr.index(':'+str(port))]
    corr_node = corraddrs[corr_address]
    content_length = int(HTTPheaderMsgStr[HTTPheaderMsgStr.index('Content-Length')+16:HTTPheaderMsgStr.index('Expect')-2])
    shape = pipeline.get_shape_from_raw(bytes(NPheaderMsgHex,'utf-8'),headersize)#tuple(NPheaderMsgStr[NPheaderMsgStr.index('shape')+8:NPheaderMsgStr.index(')')+1])
    printlog("address:"+str(corr_address),output_file=processfile)
    printlog("corr:"+str(corr_node),output_file=processfile)
    printlog("img_id:" +str(img_id_isot),output_file=processfile)
    printlog("shape:" + str(shape),output_file=processfile)

    #use content length to get just data portion
    #data = fullMsg[fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2):fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(content_length*2)]
    #printlog(fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(content_length*2)-fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2),output_file=processfile)
    #printlog(str(data[:128]),output_file=processfile)
   
    printlog("totaldatasize: " + str(len(fullMsg)),output_file=processfile)
    printlog("without HTTP header: "  + str(len(fullMsg[fullMsg.index(HEADER_DELIM):])))
    printlog("without NP header: " + str(len(fullMsg[fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2):]))) 
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

    return corr_node,img_id_isot,img_id_mjd,shape,img_data


def search_task(fullimg,SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,PyTorchDedispersion,space_filter,kernel_size,exportmaps,savesearch,append_frame,DMbatches,usejax):
    printlog("starting search process " + str(fullimg.img_id_isot) + "...",output_file=processfile,end='')

    #define search params
    gridsize=fullimg.image_tesseract.shape[0]
    RA_axis = fullimg.RA_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
    DEC_axis= fullimg.DEC_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
    nsamps = fullimg.image_tesseract.shape[2]
    nchans = fullimg.image_tesseract.shape[3]
    time_axis = np.arange(nsamps)*sl.tsamp
    canddict = dict()

    #print("starting process " + str(img_id) + "...")
    timing1 = time.time()
    if PyTorchDedispersion: #uses Nikita's dedisp code
        printlog("Using PyTorchDedispersion",output_file=processfile)
        fullimg.candidxs,fullimg.cands,fullimg.image_tesseract_searched,fullimg.image_tesseract_binned,canddict,tmp = sl.run_PyTorchDedisp_search(fullimg.image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,SNRthresh=SNRthresh,canddict=dict(),output_file=sl.output_file,usefft=usefft,space_filter=space_filter)

    else:
        fullimg.candidxs,fullimg.cands,fullimg.image_tesseract_searched,fullimg.image_tesseract_binned,canddict,tmp,tmp,tmp,tmp = sl.run_search_new(fullimg.image_tesseract,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,canddict=dict(),PSF=sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,nchans=nchans),usefft=usefft,multithreading=multithreading,nrows=nrows,ncols=ncols,output_file=sl.output_file,threadDM=threadDM,samenoise=samenoise,cuda=cuda,space_filter=space_filter,kernel_size=kernel_size,exportmaps=exportmaps,append_frame=append_frame,DMbatches=DMbatches,usejax=usejax)
    printlog(fullimg.image_tesseract_searched,output_file=processfile)
    printlog("done, total search time: " + str(np.around(time.time()-timing1,2)) + " s",output_file=processfile)

    if savesearch:
        f = open(cand_dir + fullimg.img_id_isot + ".npy","wb")
        np.save(f,fullimg.image_tesseract_searched)
        f.close()

    #only save if we find candidates
    if len(fullimg.candidxs)==0:
        printlog("No candidates found",output_file=processfile)
        return fullimg.image_tesseract_searched#fullimg.cands,fullimg.candidxs,len(fullimg.cands)



    #clustering with hdbscan
    if cluster:
        printlog("clustering with HDBSCAN...",output_file=processfile)

        #prune candidates with infinite signal-to-noise for clustering
        cands_noninf = []
        for i in fullimg.candidxs:
            if not np.isinf(i[-1]): cands_noninf.append(i)  

        #clustering with hdbscan
        classes,fullimg.cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs = sl.hdbscan_cluster(cands_noninf,min_cluster_size=5,gridsize=gridsize,plot=True,show=False,SNRthresh=SNRthresh)
        printlog("done, made " + str(len(fullimg.cluster_cands)) + " clusters",output_file=processfile)
        printlog(classes,output_file=processfile)
        printlog(fullimg.cluster_cands,output_file=processfile)
    else:
        fullimg.cluster_cands = fullimg.candidxs

    
    printlog("basic clustering in RA, DEC...",output_file=processfile,end='')
    fullimg.unique_cands = [(fullimg.cluster_cands[i][0],fullimg.cluster_cands[i][1],fullimg.cluster_cands[i][3]) for i in range(len(fullimg.cluster_cands))]
    fullimg.unique_cands = list(set(fullimg.unique_cands))
    printlog("{a} unique positions/widths...".format(a=len(fullimg.unique_cands)),output_file=processfile,end='')
    printlog("done",output_file=processfile)

    printlog("obtaining image cutouts...",output_file=processfile,end='')
    fullimg.subimgs = np.zeros((len(fullimg.unique_cands),subimgpix,subimgpix,fullimg.image_tesseract_binned.shape[3]),dtype=np.float16)
    """
    for i in range(len(fullimg.unique_cands)):
        fullimg.subimgs[i,:,:,:] = sl.get_subimage(fullimg.image_tesseract_binned,fullimg.unique_cands[i][0],fullimg.unique_cands[i][1],save=False,subimgpix=subimgpix)[:,:,int(fullimg.unique_cands[i][2]),:]
    """
    data_array = np.nan_to_num(fullimg.subimgs,nan=0.0) #change nans to 0s so that classification works, maybe better to implement something different here


    #*** The classifier only classifies based on RA,DEC,frequency, so we should bin each candidate in time and just send the peak time sample. I'm going to modify sl.get_subimage to output an array that is Ncands x RA x DEC x frequency after binning in time for th emaximum pulse width for each candidate. ***#
    # actually...we already output a de-dispersed and binned image tessearct from the search; let's also output one thats just been binned, then we can get argmax for each time series and use that.


    #print(data_array.shape)
    transposed_array = np.transpose(data_array, (0,3,1,2))#cands x frequencies x RA x DEC
    new_shape = (data_array.shape[0], data_array.shape[3], data_array.shape[1], data_array.shape[2])
    merged_array = transposed_array.reshape(new_shape)

    predictions, probabilities = classify_images(merged_array, model_weights, verbose=verbose)  

    printlog(predictions,output_file=processfile)
    printlog(probabilities,output_file=processfile)
    fullimg.predictions = copy.deepcopy(predictions)
    fullimg.probabilities = copy.deepcopy(probabilities)

    #find candidates most likely to be real; need to ask Nikita about conditions
    finalidxs = np.arange(data_array.shape[0])[~np.array(fullimg.predictions,dtype=bool)]
    
    #only save if we find candidates
    if len(finalidxs)==0: 
        printlog("No candidates found",output_file=processfile)
        return fullimg.image_tesseract_searched #fullimg.cands,fullimg.cluster_cands,len(fullimg.cluster_cands)


    #save predictions/probabilities to a csv
    with open(cand_dir + "/classification_" + fullimg.img_id_isot + ".csv","w") as csvfile:
        wr = csv.writer(csvfile,delimiter=',')
        wr.writerow(np.concatenate([["predictions"],predictions]))
        wr.writerow(np.concatenate([["probabilities"],probabilities]))
    csvfile.close()



    #get candidates most likely to be real; need to ask Nikita about conditions
    finalcands = [fullimg.cluster_cands[i] for i in finalidxs]#[condition]

    #dump sub-images to numpy files and write candidates to csv
    suffix = ".npy"# + fullimg_array[idx].img_id_hex
    prefix = "NSFRB"
    lastname = None	#once we have etcd, change to 'names.get_lastname()'
    csvfile = open(cand_dir + "candidates_" + fullimg.img_id_isot + ".csv","w")
    wr = csv.writer(csvfile,delimiter=',')
    wr.writerow(["candname","RA index","DEC index","WIDTH index", "DM index", "SNR"])
    for i in range(len(finalidxs)):
        lastname = names.increment_name(fullimg.img_id_mjd,lastname=lastname)
        finalcand = finalcands[i]
        wr.writerow(np.concatenate([[lastname],np.array(finalcand,dtype=int)]))
        np.save(cand_dir + prefix + lastname + suffix,data_array[finalidxs[i],:,:,:])       
    csvfile.close()

    #make diagnostic plot with all candidates and push to slack
    if len(finalidxs) > 0:
        #make diagnostic plot
        printlog("making diagnostic plot...",output_file=processfile,end='')
        candplot=pl.search_plots_new(canddict,fullimg.image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,DM_trials=sl.DM_trials,widthtrials=sl.widthtrials,output_dir=cand_dir,show=False)
        printlog("done!",output_file=processfile)

        if toslack:
            printlog("sending plot to slack...",output_file=processfile,end='')
            send_candidate_slack(candplot)
            printlog("done!",output_file=processfile)


    #if args.verbose:
    printlog(fullimg.subimgs.shape,output_file=processfile)
    printlog("done",output_file=processfile)



    

    return fullimg.image_tesseract_searched#, SNRthresh#fullimg.cands,fullimg.cluster_cands,len(fullimg.cluster_cands)

def future_callback(future,SNRthresh,timestepisot,RA_axis,DEC_axis):
    """
    This function prints the result once a thread finishes processing an image
    """
    printlog(future.result(),output_file=processfile)
    pl.binary_plot(future.result(),SNRthresh,timestepisot,RA_axis,DEC_axis)
    printlog("****Thread Completed****",output_file=processfile)
    printlog(future.result(),output_file=processfile)
    printlog("************************",output_file=processfile)
    return

def main():
    #redirect stderr
    sys.stderr = open(error_file,"w")
    
    
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 3000',default=3000)
    parser.add_argument('--port',type=int,help='Port number for receiving data from subclient, default = 8080',default=8080)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, default=300',default=300)
    parser.add_argument('--nsamps',type=int,help='Expected number of time samples (integrations) for each sub-band image, default=25',default=25)
    parser.add_argument('--nchans',type=int,help='Expected number of sub-band images for each full image, default=16',default=16)
    parser.add_argument('--datasize',type=int,help='Expected size of each element in sub-band image in bytes,default=8',default=8,choices=list(dtypelookup.keys()))
    parser.add_argument('--subimgpix',type=int,help='Length of image cutouts in pixels, default=11',default=11)
    parser.add_argument('-T','--testh23',action='store_true')
    parser.add_argument('--maxconnect',type=int,help='Maximum number of connections accepted by the server, default=16',default=16)
    parser.add_argument('--timeout',type=float,help='Max time in seconds to wait for more data to be ready to receive, default = 10',default=10)

    #arguments for classifier from classifier.py
    #parser.add_argument('--npy_file', type=str, required=True, help='Path to the NumPy file containing the images')
    parser.add_argument('--model_weights', type=str, help='Path to the model weights file',default=cwd + "/simulations_and_classifications/model_weights.pth")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of images that can be searched at once, default = 5, maximum is 40',default=5)
    parser.add_argument('--headersize',type=int,help='Number of bytes representing the header; note this varies depending on the data shape, default = 128',default=128)
    parser.add_argument('--spacefilter',action='store_true', help='Use PSF to spatial matched filter the input image')
    parser.add_argument('--kernelsize',type=int,help='Kernel size for PSF spatial matched filter; default is same as image size',default=300)
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
    parser.add_argument('--initnoise',action='store_true',help='Initializes noise statistics for S/N estimates')
    parser.add_argument('--savesearch',action='store_true',help='Saves the searched image as a numpy array')
    parser.add_argument('--appendframe',action='store_true',help='Use the previous image to fill in dedispersion search')
    parser.add_argument('--DMbatches',type=int,help='Number of pixel batches to submit dedispersion to the GPUs with, defauls = 1',default=1)
    parser.add_argument('--usejax',action='store_true',help='Use JAX Just-In-Time compilation for GPU acceleration')
    args = parser.parse_args()    

   
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
        jax_funcs.inner_dedisperse_jit(image_tesseract_point=np.random.normal(size=(args.gridsize//args.DMbatches,args.gridsize//args.DMbatches,args.nsamps,args.nchans)),
                                    DM_trials_in=sl.DM_trials,tsamp=sl.tsamp,freq_axis_in=sl.freq_axis)
        jax_funcs.inner_snr_fft_jit(image_tesseract_filtered_dm=np.random.normal(size=(args.gridsize//args.DMbatches,args.gridsize//args.DMbatches,args.nsamps,len(sl.DM_trials))),
                                    boxcar=np.random.normal(size=(len(sl.widthtrials),args.gridsize//args.DMbatches,args.gridsize//args.DMbatches,args.nsamps,len(sl.DM_trials))))
    #initialize last_frame 
    if args.initframes:
        printlog("Initializing previous frames...",output_file=processfile)
        sl.init_last_frame(args.gridsize,args.gridsize,args.nsamps,args.nchans)

    #initialize noise stats
    if args.initnoise:
        printlog("Initializing noise statistics...",output_file=processfile)
        init_noise()

    printlog("USEFFT = " + str(args.usefft),output_file=processfile)
    #total expected number of bytes for each sub-band image
    if args.datasize%2 != 0:
        maxbytes = args.gridsize*args.gridsize*args.nsamps*(args.datasize-1) + args.headersize
    else:
        maxbytes = args.gridsize*args.gridsize*args.nsamps*args.datasize + args.headersize #+ 42 #35 extra bytes are from the meta-data appended by the persistent RX server, but need to wait to see length of the ip address
    printlog("MAXBYTES: " + str(maxbytes),output_file=processfile)
    printlog("SHAPE: "  + str((args.gridsize,args.gridsize,args.nsamps,args.nchans)),output_file=processfile)
   
    #array to store image ids temporarily
    fullimg_array = np.ndarray(shape=(args.maxProcesses),dtype=fullimg)


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
    executor = ThreadPoolExecutor(args.maxProcesses)#ProcessPoolExecutor(args.maxProcesses)
    task_list = []

    while True: # want to keep accepting connections
        printlog("accepting connection...",output_file=processfile,end='')
        clientSocket,address = servSockD.accept()
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
        while (recstatus> 0) and (totalbytes < maxbytes):#+maxbytesaddr):
            try:
                (strData, ancdata, msg_flags, address) = clientSocket.recvmsg(255)
                #printlog(strData,output_file=processfile)
                recstatus = len(strData)

                #printlog(strData.hex(),output_file=processfile,end='')
                fullMsg += strData.hex()
                totalbytes += recstatus

                #don't know how long the header is, so don't start counting until hit NP data
                if "93" in fullMsg:
                    printlog("Found start byte at index " + str(fullMsg.index("93")),output_file=processfile)
                    totalbytes = (len(fullMsg) - fullMsg.index("93"))//2
                
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
            assert(totalbytes>=maxbytes)
        except AssertionError as exc:
            printlog("Invalid data size, " + str(totalbytes) + " received when expected at least " + str(maxbytes) + ": " + str(exc),output_file=processfile)
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
            corr_node,img_id_isot,img_id_mjd,shape,arrData = parse_packet(fullMsg=fullMsg,maxbytes=maxbytes,headersize=args.headersize,datasize=args.datasize,port=args.port,corr_address=corr_address,testh23=args.testh23)
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

        #if object corresponding to the image is in list
        idx,openidx = find_id(img_id_isot,fullimg_array)
        printlog("FIND_ID: " + str(idx) + ", " + str(openidx),output_file=processfile)#if it's not in the list, but there's an open spot, add it
        if idx == -1 and openidx != -1:
            #need to create new object
            fullimg_array[openidx] = fullimg(img_id_isot,img_id_mjd,shape=tuple(np.concatenate([shape,[16]])))
            idx = openidx
        elif idx == -1 and openidx == -1: # shouldn't reach this case often, but if we don't have space for a new object, busy wait
            while openidx == -1: 
                printlog("Process server image array full, waiting for opening...",output_file=processfile,end='')
                idx,openidx = find_id(img_id_isot,fullimg_array)
        #otherwise, just add to the image at idx
            	
        #add image and update flags
        fullimg_array[idx].add_corr_img(arrData,corr_node,args.testh23)
        #if the image is complete, start the search
        printlog("corrstatus:",output_file=processfile,end='')
        printlog(fullimg_array[idx].corrstatus,output_file=processfile)
        if fullimg_array[idx].is_full():
            #submit a search task to the process pool
            printlog("Submitting new task for image " + str(idx),output_file=processfile)
            RA_axis_idx = copy.deepcopy(fullimg_array[idx].RA_axis)
            DEC_axis_idx= copy.deepcopy(fullimg_array[idx].DEC_axis)
            task_list.append(executor.submit(search_task,fullimg_array[idx],args.SNRthresh,args.subimgpix,args.model_weights,args.verbose,args.usefft,args.cluster,
                                    args.multithreading,args.nrows,args.ncols,args.threadDM,args.samenoise,args.cuda,args.toslack,args.PyTorchDedispersion,
                                    args.spacefilter,args.kernelsize,args.exportmaps,args.savesearch,args.appendframe,args.DMbatches,args.usejax))
            
            #printlog(future.result(),output_file=processfile)
            task_list[-1].add_done_callback(lambda future: future_callback(future,args.SNRthresh,img_id_isot,RA_axis_idx,DEC_axis_idx))
            #after finishes execution, remove from list by setting element to None
            fullimg_array[idx] = None
    

        

        #sys.stdout.flush()
    executor.shutdown()
    clientSocket.close()



if __name__=="__main__":
    main()
