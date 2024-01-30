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

import sys
sys.path.append("/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb import searching as sl
from nsfrb import pipeline

"""
Directory for output data
"""
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_flags/searchlog_flags.txt"
output_file = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
processfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt"
"""
Arguments: data file
"""
from nsfrb.logging import printlog

"""
Create a structure for full image
"""
class fullimg:
    def __init__(self,img_id,shape=(32,32,25,16),dtype=np.float16):
        self.image_tesseract = np.zeros(shape,dtype=dtype)
        self.corrstatus = np.zeros(16,dtype=bool)
        self.img_id = img_id
        self.shape = shape

    def add_corr_img(self,data,corr_node):
        self.image_tesseract[:,:,:,corr_node] = data
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
    img_id = int(fullMsgStr[fullMsgStr.index("ENDADDR")+7:fullMsgStr.index("ENDIMGID")],16)
    data = fullMsg[2*(fullMsgStr.index("ENDIMGID")+8):]
    printlog("address:"+str(corr_address),output_file=processfile)
    printlog("corr:"+str(corr_node),output_file=processfile)
    printlog("img_id:" +str(img_id),output_file=processfile)
     
    headerbytes = bytes(data[:2*headersize],'utf-8')
    printlog(bytes.fromhex(data[:2*headersize]),output_file=processfile)
    imgbytes = bytes.fromhex(data[2*headersize:])
    shape = pipeline.get_shape_from_raw(headerbytes,headersize)#bytedata[:headersize],headersize)
    img_data = np.frombuffer(imgbytes,dtype=np.float64).reshape(shape)
    printlog(shape,output_file=processfile)
    return corr_node,img_id,shape,img_data
    



def main():
    #create socket
    printlog("creating socket...",output_file=processfile,end='')
    servSockD = socket.socket(socket.AF_INET, socket.SOCK_STREAM,0)
    printlog("Done!",output_file=processfile)    

    #bind to port number
    port = 8843
    printlog("binding to port " + str(port) + "...",output_file=processfile,end='')
    servSockD.bind(('', port))
    printlog("Done!",output_file=processfile)

    #listen for conections
    printlog("listening for connections...",output_file=processfile,end='')
    servSockD.listen(16)
    printlog("Made connection",output_file=processfile)
    
    while True: # want to keep accepting connections
        printlog("accepting connection...",output_file=processfile,end='')
        clientSocket,address = servSockD.accept()
        recstatus = 1
        fullMsg = ""
        printlog("Done!",output_file=processfile)
        printlog("Receiving data...",output_file=processfile)
        
        #set timeout and expected number of bytes to read
        clientSocket.settimeout(10) 
        maxbytes = 204958
        totalbytes = 0

        while (recstatus > 0) and (totalbytes < maxbytes):
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
        #print(fullMsg[:256])
        #parse to get address
        corr_node,img_id,shape,arrData = parse_packet(fullMsg)
        #printlog(arrData,output_file=processfile)
        printlog("Received data from corr " + str(corr_node),output_file=processfile)
        printlog("Shape " + str(shape),output_file=processfile)
        printlog("img_id " + str(img_id),output_file=processfile)
		

        #if object corresponding to the image is in list
        idx = find_id(img_id)
        if idx == -1:
            #need to create new object
            fullimg_array.append(fullimg(img_id))
		
        #add image and update flags
        fullimg_array[idx].add_corr_img(arrData,corr_node)
        #if the image is complete, start the search
        if fullimg_array[idx].is_full():
            #later want to make this in a new thread, but for now run it here
            printlog("starting search here",output_file=processfile)
            #cands,cluster_cands,image_tesseract_searched = sl.run_search(fullimage_array[idx].image_tesseract,image_tesseract,SNRthresh=30000)

        sys.stdout.flush()
    clientSocket.close()

        



if __name__=="__main__":
    main()
