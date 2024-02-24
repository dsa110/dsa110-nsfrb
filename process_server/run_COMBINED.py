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
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.append("/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
import csv
import copy

from simulations_and_classifications.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset

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
sys.path.append("/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb import searching as sl
from nsfrb import pipeline

"""s
Directory for output data
"""
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_flags/searchlog_flags.txt"
output_file = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
processfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt"
flagfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_flags.txt"
cand_dir = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
"""
Arguments: data file
"""
from nsfrb.outputlogging import printlog


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
            '10.41.0.182' : 0, #h23, placeholder
            '10.41.0.94' : 0 #corr20
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
def parse_packet(fullMsg,maxbytes,headersize,datasize,port,testh23=False):
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
    corr_address = HTTPheaderMsgStr[HTTPheaderMsgStr.index('Host')+6:HTTPheaderMsgStr.index(':'+str(port))]
    corr_node = corraddrs[corr_address]
    content_length = int(HTTPheaderMsgStr[HTTPheaderMsgStr.index('Content-Length')+16:HTTPheaderMsgStr.index('Expect')-2])
    shape = pipeline.get_shape_from_raw(bytes(NPheaderMsgHex,'utf-8'),headersize)#tuple(NPheaderMsgStr[NPheaderMsgStr.index('shape')+8:NPheaderMsgStr.index(')')+1])
    printlog("address:"+str(corr_address),output_file=processfile)
    printlog("corr:"+str(corr_node),output_file=processfile)
    printlog("img_id:" +str(img_id_isot),output_file=processfile)
    printlog("shape:" + str(shape),output_file=processfile)

    #use content length to get just data portion
    data = fullMsg[fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(headersize*2):fullMsg.index(HEADER_DELIM)+len(HEADER_DELIM)+(content_length*2)]
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


def search_task(fullimg,SNRthresh,subimgpix,model_weights,verbose):
    printlog("starting search process " + str(fullimg.img_id_isot) + "...",output_file=processfile,end='')

    #define search params
    gridsize=fullimg.image_tesseract.shape[0]
    RA_axis = np.linspace(-gridsize//2,gridsize//2,gridsize)
    DEC_axis=np.linspace(-gridsize//2,gridsize//2,gridsize)
    nsamps = fullimg.image_tesseract.shape[2]
    time_axis = np.arange(nsamps)*sl.tsamp

    #print("starting process " + str(img_id) + "...")
    fullimg.cands,fullimg.cluster_cands,fullimg.image_tesseract_searched,fullimg.image_tesseract_binned = sl.run_search(fullimg.image_tesseract,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis)
    printlog("done",output_file=processfile)
    
    printlog("basic clustering in RA, DEC...",output_file=processfile,end='')
    fullimg.unique_cands = [(fullimg.cluster_cands[i][0],fullimg.cluster_cands[i][1],fullimg.cluster_cands[i][3]) for i in range(len(fullimg.cluster_cands))]
    fullimg.unique_cands = list(set(fullimg.unique_cands))
    printlog("{a} unique positions/widths...".format(a=len(fullimg.unique_cands)),output_file=processfile,end='')
    printlog("done",output_file=processfile)

    printlog("obtaining image cutouts...",output_file=processfile,end='')
    fullimg.subimgs = np.zeros((len(fullimg.unique_cands),subimgpix,subimgpix,fullimg.image_tesseract_binned.shape[3]),dtype=np.float16)
    for i in range(len(fullimg.unique_cands)):
        fullimg.subimgs[i,:,:,:] = sl.get_subimage(fullimg.image_tesseract_binned,fullimg.unique_cands[i][0],fullimg.unique_cands[i][1],save=False,subimgpix=subimgpix)[:,:,fullimg.unique_cands[i][2],:]

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
        return fullimg.cands,fullimg.cluster_cands,len(fullimg.cluster_cands)


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

    #if args.verbose:
    printlog(fullimg.subimgs.shape,output_file=processfile)
    printlog("done",output_file=processfile)



    

    return fullimg.cands,fullimg.cluster_cands,len(fullimg.cluster_cands)


def main():
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 3000',default=3000)
    parser.add_argument('--port',type=int,help='Port number for receiving data from subclient, default = 8843',default=8843)
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
    parser.add_argument('--model_weights', type=str, help='Path to the model weights file',default="/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/simulations_and_classifications/model_weights.pth")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of images that can be searched at once, default = 5, maximum is 40',default=5)
    parser.add_argument('--headersize',type=int,help='Number of bytes representing the header; note this varies depending on the data shape, default = 128',default=128)
    args = parser.parse_args()    
    
    
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
    executor = ProcessPoolExecutor(args.maxProcesses)
   
    while True: # want to keep accepting connections
        printlog("accepting connection...",output_file=processfile,end='')
        clientSocket,address = servSockD.accept()
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
                printlog(strData,output_file=processfile)
                recstatus = len(strData)

                #printlog(strData.hex(),output_file=processfile,end='')
                fullMsg += strData.hex()
                totalbytes += recstatus
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
            printlog("Invalid data size, " + str(totalbytes) + " received when expected at least " + str(maxbytes+maxbytesaddr) + ": " + str(exc),output_file=processfile)
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
            corr_node,img_id_isot,img_id_mjd,shape,arrData = parse_packet(fullMsg=fullMsg,maxbytes=maxbytes,headersize=args.headersize,datasize=args.datasize,port=args.port,testh23=args.testh23)
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
            future = executor.submit(search_task,fullimg_array[idx],args.SNRthresh,args.subimgpix,args.model_weights,args.verbose)
            printlog(future.result(),output_file=processfile)

            #after finishes execution, remove from list by setting element to None
            fullimg_array[idx] = None
            
        #sys.stdout.flush()
    clientSocket.close()



if __name__=="__main__":
    main()
