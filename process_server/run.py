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
        if testmode:
            self.corrstatus[corr_node] = 1
        return
	    
    def is_full(self):
        return np.all(self.corrstatus==1)

def find_id(img_id_isot,fullimg_array):
    if len(fullimg_array) == 0: return -1

    #also want to see if any spaces are open
    openidx = -1

    for i in range(len(fullimg_array)):
        if fullimg_array[i] == None and openidx == -1:
            openidx = i
        elif fullimg_array[i] == None: 
            continue
        elif fullimg_array[i].img_id_isot == img_id_isot: return i,None
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


def parse_packet(fullMsg,headersize=128,testh23=False):
    #first need  bytestring representation to find metadata
    fullMsgStr = str(bytes.fromhex(fullMsg))[2:]
  

    #get metadata
    corr_address = fullMsgStr[:fullMsgStr.index("ENDADDR")]
    corr_node = corraddrs[corr_address]

    #print(fullMsgStr[fullMsgStr.index("ENDADDR"):128])
    #images will be labelled with the datetime in ISOT format
    img_id_isot = fullMsgStr[fullMsgStr.index("ENDADDR")+7:fullMsgStr.index("ENDIMGID")]
    img_id_mjd = Time(img_id_isot,format='isot').mjd

    #img_id_hex = fullMsgStr[fullMsgStr.index("ENDADDR")+7:fullMsgStr.index("ENDIMGID")]
    #img_id = int(fullMsgStr[fullMsgStr.index("ENDADDR")+7:fullMsgStr.index("ENDIMGID")],16)
    data = fullMsg[2*(fullMsgStr.index("ENDIMGID")+8):]
    printlog("address:"+str(corr_address),output_file=processfile)
    printlog("corr:"+str(corr_node),output_file=processfile)
    printlog("img_id:" +str(img_id_isot),output_file=processfile)
     
    headerbytes = bytes(data[:2*headersize],'utf-8')
    printlog(bytes.fromhex(data[:2*headersize]),output_file=processfile)
    imgbytes = bytes.fromhex(data[2*headersize:])
    shape = pipeline.get_shape_from_raw(headerbytes,headersize)#bytedata[:headersize],headersize)
    printlog(shape,output_file=processfile)
    printlog(len(imgbytes),output_file=processfile)
    img_data = np.frombuffer(imgbytes,dtype=np.float64).reshape(shape)
    printlog(shape,output_file=processfile)

    #***only keep this part while we test with h23***
    if testh23:
        corraddrs[corr_address] += 1
        if corraddrs[corr_address] > 15:
            corraddrs[corr_address] = 0

    return corr_node,img_id_isot,img_id_mjd,shape,img_data
    


def search_task(fullimg,SNRthresh,subimgpix,model_weights,verbose):
    printlog("starting search process " + str(fullimg.img_id_isot) + "...",output_file=processfile,end='')
    #print("starting process " + str(img_id) + "...")
    fullimg.cands,fullimg.cluster_cands,fullimg.image_tesseract_searched,fullimg.image_tesseract_binned = sl.run_search(fullimg.image_tesseract,SNRthresh=SNRthresh)
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
    parser.add_argument('--maxbytes',type=int,help='Expected size of sub-band image data in bytes, default = 204958 (for 32x32x25 image)',default=204958)
    parser.add_argument('--subimgpix',type=int,help='Length of image cutouts in pixels, default=11',default=11)
    parser.add_argument('-T','--testh23',action='store_true')
    parser.add_argument('--maxconnect',type=int,help='Maximum number of connections accepted by the server, default=16',default=16)
    parser.add_argument('--timeout',type=float,help='Max time in seconds to wait for more data to be ready to receive, default = 10',default=10)

    #arguments for classifier from classifier.py
    #parser.add_argument('--npy_file', type=str, required=True, help='Path to the NumPy file containing the images')
    parser.add_argument('--model_weights', type=str, help='Path to the model weights file',default="/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/simulations_and_classifications/model_weights.pth")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of images that can be searched at once, default = 5, maximum is 40',default=5)
    args = parser.parse_args()    
       
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
        maxbytes = args.maxbytes #204958
        totalbytes = 0

        while (recstatus> 0) and (totalbytes < maxbytes):
            try:
                (strData, ancdata, msg_flags, address) = clientSocket.recvmsg(255)
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
        #print(fullMsg)
        


        #try to parse to get address
        try:
            corr_node,img_id_isot,img_id_mjd,shape,arrData = parse_packet(fullMsg,testh23=args.testh23)
            if pipeline.set_pflag("all",on=False) == None:
                printlog("Error setting flags, abort",processfile=processfile)
                break
        except UnicodeDecodeError as exc:
            printlog("Error parsing data: " + str(exc),output_file=processfile)
            printlog("Setting retry flag...",output_file=processfile,end='')
            if pipeline.set_pflag("parse_error") == None: 
                printlog("Error setting flags, abort",processfile=processfile)
                break
            printlog("Done, continue",output_file=processfile)
            continue	        
        except ValueError as exc:
            printlog("Invalid data size: " + str(exc),output_file=processfile)
            printlog("Setting truncated data size...",output_file=processfile,end='')
            if pipeline.set_pflag("datasize_error") == None:
                printlog("Error setting flags, abort",processfile=processfile,end='')
                break
            printlog("Done, continue",output_file=processfile)
            continue     
            
        #printlog(arrData,output_file=processfile)
        printlog("Received data from corr " + str(corr_node),output_file=processfile)
        printlog("Shape " + str(shape),output_file=processfile)
        printlog("img_id (isot) " + str(img_id_isot),output_file=processfile)
        printlog("img_id (mjd) " + str(img_id_mjd),output_file=processfile)

        


        #if object corresponding to the image is in list
        idx,openidx = find_id(img_id_isot,fullimg_array)
        #if it's not in the list, but there's an open spot, add it
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
