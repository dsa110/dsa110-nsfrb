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

#from gen_dmtrials_copy import gen_dm
import argparse

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
cand_dir = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
"""
Arguments: data file
"""
from nsfrb.logging import printlog

"""
Create a structure for full image
"""
class fullimg:
    def __init__(self,img_id,img_id_hex,shape=(32,32,25,16),dtype=np.float16):
        self.image_tesseract = np.zeros(shape,dtype=dtype)
        self.corrstatus = np.zeros(16,dtype=bool)
        self.img_id = img_id
        self.img_id_hex = img_id_hex
        self.shape = shape

    def add_corr_img(self,data,corr_node,testmode=False):
        self.image_tesseract[:,:,:,corr_node] = data
        if testmode:
            self.corrstatus[corr_node] = 1
        return
	    
    def is_full(self):
        return np.all(self.corrstatus==1)

fullimg_array = []
def find_id(img_id):
    if len(fullimg_array) == 0: return -1
	
    for i in range(len(fullimg_array)):
	    if fullimg_array[i].img_id == img_id: return i
    return -1

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
            '10.41.0.182' : 0 #h23, placeholder
             }

def parse_packet(fullMsg,headersize=128):
    #first need  bytestring representation to find metadata
    fullMsgStr = str(bytes.fromhex(fullMsg))[2:]
  

    #get metadata
    corr_address = fullMsgStr[:fullMsgStr.index("ENDADDR")]
    corr_node = corraddrs[corr_address]

    #***only keep this part while we test with h23***
    corraddrs[corr_address] += 1
    if corraddrs[corr_address] > 15:
        corraddrs[corr_address] = 0
    #***#
    #print(fullMsgStr[fullMsgStr.index("ENDADDR"):128])
    img_id_hex = fullMsgStr[fullMsgStr.index("ENDADDR")+7:fullMsgStr.index("ENDIMGID")]
    img_id = int(fullMsgStr[fullMsgStr.index("ENDADDR")+7:fullMsgStr.index("ENDIMGID")],16)
    data = fullMsg[2*(fullMsgStr.index("ENDIMGID")+8):]
    printlog("address:"+str(corr_address),output_file=processfile)
    printlog("corr:"+str(corr_node),output_file=processfile)
    printlog("img_id:" +str(img_id),output_file=processfile)
     
    headerbytes = bytes(data[:2*headersize],'utf-8')
    printlog(bytes.fromhex(data[:2*headersize]),output_file=processfile)
    imgbytes = bytes.fromhex(data[2*headersize:])
    shape = pipeline.get_shape_from_raw(headerbytes,headersize)#bytedata[:headersize],headersize)
    print(shape)
    printlog(len(imgbytes),output_file=processfile)
    img_data = np.frombuffer(imgbytes,dtype=np.float64).reshape(shape)
    printlog(shape,output_file=processfile)
    return corr_node,img_id,img_id_hex,shape,img_data
    
maxProcesses = 5


def search_task(image_tesseract,SNRthresh,img_id,idx,subimgpix,model_weights,verbose):
    printlog("starting search process " + str(img_id) + "...",output_file=processfile,end='')
    #print("starting process " + str(img_id) + "...")
    fullimg_array[idx].cands,fullimg_array[idx].cluster_cands,fullimg_array[idx].image_tesseract_searched,fullimg_array[idx].image_tesseract_binned = sl.run_search(image_tesseract,SNRthresh=SNRthresh)
    printlog("done",output_file=processfile)
    
    printlog("basic clustering in RA, DEC...",output_file=processfile,end='')
    fullimg_array[idx].unique_cands = [(fullimg_array[idx].cluster_cands[i][0],fullimg_array[idx].cluster_cands[i][1],fullimg_array[idx].cluster_cands[i][3]) for i in range(len(fullimg_array[idx].cluster_cands))]
    fullimg_array[idx].unique_cands = list(set(fullimg_array[idx].unique_cands))
    printlog("{a} unique positions/widths...".format(a=len(fullimg_array[idx].unique_cands)),output_file=processfile,end='')
    #fullimg_array[idx].unique_cands_dm = [(fullimg_array[idx].cluster_cands[i][0],fullimg_array[idx].cluster_cands[i][1],fullimg_array[idx].cluster_cands[i][2]) for i in range(len(fullimg_array[idx].cluster_cands))]
    #fullimg_array[idx].unique_cands_dm = list(set(fullimg_array[idx].unique_cands_dm))
    #printlog("{a} unique DMs...".format(a=len(fullimg_array[idx].unique_cands_dm)),output_file=processfile,end='')
    printlog("done",output_file=processfile)

    printlog("obtaining image cutouts...",output_file=processfile,end='')
    #fullimg_array[idx].subimgs_dm = np.zeros((len(fullimg_array[idx].unique_cands_dm),subimgpix,subimgpix,fullimg_array[idx].image_tesseract.shape[2],fullimg_array[idx].image_tesseract.shape[3]),dtype=np.float16)
    fullimg_array[idx].subimgs = np.zeros((len(fullimg_array[idx].unique_cands),subimgpix,subimgpix,fullimg_array[idx].image_tesseract_binned.shape[3]),dtype=np.float16)
    for i in range(len(fullimg_array[idx].unique_cands)):
        fullimg_array[idx].subimgs[i,:,:,:] = sl.get_subimage(fullimg_array[idx].image_tesseract_binned,fullimg_array[idx].unique_cands[i][0],fullimg_array[idx].unique_cands[i][1],save=False,subimgpix=subimgpix)[:,:,fullimg_array[idx].unique_cands[i][2],:]
    #for i in range(len(fullimg_array[idx].unique_cands_dm)):
    #    fullimg_array[idx].subimgs_dm[i,:,:,:,:] = sl.get_subimage(fullimg_array[idx].image_tesseract,fullimg_array[idx].unique_cands_dm[i][0],fullimg_array[idx].unique_cands_dm[i][1],dm=sl.DM_trials[fullimg_array[idx].unique_cands_dm[i][2]],save=False,subimgpix=subimgpix)


    data_array = np.nan_to_num(fullimg_array[idx].subimgs,nan=0.0) #change nans to 0s so that classification works, maybe better to implement something different here


    #*** The classifier only classifies based on RA,DEC,frequency, so we should bin each candidate in time and just send the peak time sample. I'm going to modify sl.get_subimage to output an array that is Ncands x RA x DEC x frequency after binning in time for th emaximum pulse width for each candidate. ***#
    # actually...we already output a de-dispersed and binned image tessearct from the search; let's also output one thats just been binned, then we can get argmax for each time series and use that.


    print(data_array.shape)
    transposed_array = np.transpose(data_array, (0,3,1,2))#cands x frequencies x RA x DEC
    print(transposed_array.shape)
    new_shape = (data_array.shape[0], data_array.shape[3], data_array.shape[1], data_array.shape[2])
    merged_array = transposed_array.reshape(new_shape)

    predictions, probabilities = classify_images(merged_array, model_weights, verbose=verbose)  

    print(predictions,probabilities)
    fullimg_array[idx].predictions = copy.deepcopy(predictions)
    fullimg_array[idx].probabilities = copy.deepcopy(probabilities)
    #save predictions/probabilities to a csv
    with open(cand_dir + "/classification_ID" + fullimg_array[idx].img_id_hex + ".csv","w") as csvfile:
        wr = csv.writer(csvfile,delimiter=',')
        wr.writerow(predictions)
        wr.writerow(probabilities)
    csvfile.close()

    #find candidates most likely to be real; need to ask Nikita about conditions
    finalidxs = np.arange(data_array.shape[0])[~np.array(fullimg_array[idx].predictions,dtype=bool)]
    finalcands = [fullimg_array[idx].cluster_cands[i] for i in finalidxs]#[condition]
    with open(cand_dir + "candidates_ID" + fullimg_array[idx].img_id_hex + ".csv","w") as csvfile:
        wr = csv.writer(csvfile,delimiter=',')
        for finalcand in finalcands:
            wr.writerow(np.array(finalcand))
    csvfile.close()

    #dump sub-images to numpy files
    prefix = "candidate_subimage_ID" + fullimg_array[idx].img_id_hex
    for i in finalidxs:
        np.save(cand_dir + prefix + "_RA" + str(fullimg_array[idx].unique_cands[i][0]) + "_DEC" + str(fullimg_array[idx].unique_cands[i][1]) + "_WIDTH" + str(fullimg_array[idx].unique_cands[i][2]) + ".npy",data_array[i,:,:,:])
    #for i in finalidxs:#range(len(fullimg_array[idx].unique_cands)):
    #    fullimg_array[idx].subimgs[i,:,:,:,:] = sl.get_subimage(fullimg_array[idx].image_tesseract,fullimg_array[idx].unique_cands[i][0],fullimg_array[idx].unique_cands[i][1],save=True,prefix="candidate_subimage_ID" + fullimg_array[idx].img_id_hex,subimgpix=subimgpix,output_dir=cand_dir)
    #for i in finalidxs:#range(len(fullimg_array[idx].unique_cands_dm)):
    #    fullimg_array[idx].subimgs_dm[i,:,:,:,:] = sl.get_subimage(fullimg_array[idx].image_tesseract,fullimg_array[idx].unique_cands_dm[i][0],fullimg_array[idx].unique_cands_dm[i][1],dm=sl.DM_trials[fullimg_array[idx].unique_cands_dm[i][2]],save=True,prefix="candidate_subimage_ID" + fullimg_array[idx].img_id_hex,subimgpix=subimgpix,output_dir=cand_dir)


    #if args.verbose:
    #printlog(fullimg_array[idx].subimgs_dm.shape)
    printlog(fullimg_array[idx].subimgs.shape)
    printlog("done",output_file=processfile)



    

    return fullimg_array[idx].cands,fullimg_array[idx].cluster_cands,len(fullimg_array[idx].cluster_cands)#sl.run_search(image_tesseract,SNRthresh=SNRthresh)


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
    args = parser.parse_args()    


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
    executor = ProcessPoolExecutor(maxProcesses)
   
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
                    print("Timed out after reading",totalbytes," bytes; proceeding...")
                    break
                else:
                    raise
        printlog("Done! Total bytes read:" + str(totalbytes),output_file=processfile)
        #print(fullMsg)
        #parse to get address
        corr_node,img_id,img_id_hex,shape,arrData = parse_packet(fullMsg)
        #printlog(arrData,output_file=processfile)
        printlog("Received data from corr " + str(corr_node),output_file=processfile)
        printlog("Shape " + str(shape),output_file=processfile)
        printlog("img_id " + str(img_id),output_file=processfile)
        printlog("img_id_hex " + str(img_id_hex),output_file=processfile)

        
        #if object corresponding to the image is in list
        idx = find_id(img_id)
        if idx == -1:
            #need to create new object
            fullimg_array.append(fullimg(img_id,img_id_hex))
		
        #add image and update flags
        fullimg_array[idx].add_corr_img(arrData,corr_node,args.testh23)
        #if the image is complete, start the search
        print("corrstatus:",fullimg_array[idx].corrstatus)
        if fullimg_array[idx].is_full():
            #submit a search task to the process pool
            future = executor.submit(search_task,fullimg_array[idx].image_tesseract,args.SNRthresh,fullimg_array[idx].img_id,idx,args.subimgpix,args.model_weights,args.verbose)
            fullimg_array[idx].future = future
            print(future.result())
            printlog(future.result(),output_file=processfile)
        sys.stdout.flush()
    clientSocket.close()

        



if __name__=="__main__":
    main()
