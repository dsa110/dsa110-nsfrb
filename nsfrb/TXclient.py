import time
import numpy as np
import urllib3
import logging
from http.client import HTTPConnection
import sys
sys.path.append("..")



#from nsfrb.pipeline import pflagdict
pflagdict = dict()
pflagdict['parse_error'] = 1
pflagdict['datasize_error'] = 2
pflagdict['shape_error'] = 4
pflagdict['invalid'] = 8
pflagdict['all'] = 15

HTTPConnection.debuglevel = 0
"""
This script implements the following curl command in python:
    curl --upload-file "${fname}" "http://10.41.0.94:8080/${fname}" -verbose --trace-ascii /media/ubuntu/ssd/sherman/code/here.txt --keepalive-time 15 --http0.9
The function send_data() should be called following imaging to send image data
to the persistent server for searching. For the realtime system, this should be called once on each corr node per 3.25 s integration to send a 256x256x25x1 image cube to T4. For the offline system, this should be called once for each sub-band image file.

"""

#parameters that never change
import os

#http parameters
port = 8080
ipaddress = os.environ['NSFRBIP']
host = ipaddress + ":" + str(port)
#url = "http://" + host + "/" + fname
keepalive_time = 15
httpversion = 0.9


#fakeheader = bytes.fromhex("934e554d5059010076007b276465736372273a20273c6638272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a20283330302c203330302c2032292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a")

#for example of headers, see /media/ubuntu/ssd/sherman/code/here.txt
def make_header(content_length,host=host):

    headers=dict()
    headers['Host'] = host
    headers['User-Agent'] = 'curl/7.78.0'
    headers['Accept'] = '*/*'
    headers['Referer'] = 'rbose'
    headers['Content-Length'] = str(content_length)
    headers['Expect'] = '100-continue'
    return headers

def build_np_header(shape,descr='<f8',fortran_order=False,headersize=128):
    #every header stars with this
    startbytes = bytes.fromhex("93") + bytes("NUMPY",'utf-8') + bytes.fromhex("010000")

    #make a dictionary
    d = dict()
    d['descr'] = descr
    d['fortran_order'] = fortran_order
    d['shape'] = shape
    dstr = str(d)[:-1]
    dstr += ", }"
    dbytes = bytes(dstr,'utf-8')

    #append spaces until we get to 128
    spacebytes = bytes(" "*(headersize - len(startbytes + dbytes) - 1) + "\n",'utf-8')

    #combine
    headerbytes = startbytes + dbytes + spacebytes
    #print(headerbytes)
    return headerbytes
    
import socket
def send_data(timestamp,uv_diag,Dec,array,shape=None,node=23,ENDFILE='',headersize=128,verbose=False,retries=5,keepalive_time=keepalive_time,port=port,ipaddress=ipaddress,protocol='tcp',udpchunksize=90601,udpoffset=0):
    if protocol=='udp':
        #make header
        host = ipaddress + ":" + str(port)
        hdrdata = np.array([uv_diag,Dec,node],dtype=np.float64)
        hdrbytes = bytes(host.encode()) + bytes(timestamp.encode()) + hdrdata.tobytes()
        print("UDP header length:",len(hdrbytes),"bytes")

        databytes = array.tobytes()
        nchunks = len(databytes)//udpchunksize
        print("Sending in ",nchunks,"chunks of ",udpchunksize+len(hdrbytes)+8,"bytes (header + data) each")
        
        sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            for ii in range(nchunks):
                i=ii+udpoffset
                print(i.to_bytes(8,byteorder='big').hex())
                print(len(i.to_bytes(8,byteorder='big')+hdrbytes+databytes[ii*udpchunksize:(ii+1)*udpchunksize]))
                sock.sendto((i.to_bytes(8,byteorder='big')+hdrbytes+databytes[ii*udpchunksize:(ii+1)*udpchunksize]),(ipaddress,port))
            print("Done sending, sleep")
            #time.sleep(60)
        except Exception as exc:

            print(exc)
        finally:
            sock.close()
        print("Done")
        return i
    
    host = ipaddress + ":" + str(port)

    if type(array) != bytes and shape is None:
        shape = array.shape
    elif shape is None:
        print("Invalid shape")
        return
    if verbose:
        print(shape)
    #timestamp is how sub-bands will be associated with each other, so ensure its the same for all sub-bands; should be in isot format: 'yyyy:mm:ddThh:mm:ss'
    #array is numpy array of sub-band data
    if node < 10:
        sb = "sb0" + str(node)
    elif node >= 10 and node < 15:
        sb = "sb" + str(node)
    elif node== 23:
        sb = "_h23"
    else:
        if verbose:
            print("Invalid corr node")

    #make filename
    fname = sb + "_IMG" + str(timestamp) + "_UV" + str(np.float64(uv_diag).tobytes().hex()) +  "_DE" + str(np.float64(Dec).tobytes().hex()) + ".npy"
    url = host + "/" + fname
    
    #make a header
    fakeheader = build_np_header(shape=shape,headersize=headersize)

    #convert numpy data to bytes
    if type(array) == bytes:
        body = fakeheader + array + bytes(ENDFILE,encoding='utf-8') #note we had to remove the newline
    else:
        body = fakeheader + array.tobytes() + bytes(ENDFILE,encoding='utf-8')
    content_length = len(body)


    #print(body.decode('utf-8'))
    if verbose:
        print("Sending " + fname)
        print("Message body of size " + str(content_length))
    """
    f = open("TXoutput.txt","w")
    f.write(body.hex())
    f.close()
    
    if verbose:
        HTTPConnection.debuglevel = 1
        print("Sending " + fname)
        print("Message body of size " + str(content_length))
    

        #logging
        #logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger()
        logger.setLevel(logging.NOTSET)
        handler = logging.FileHandler('urllib3_logs.log')
        #handler.setLevel(logging.NOTSET)
        logger.addHandler(handler)
    """
    
    tries = 0
    http = urllib3.PoolManager()
    while tries < retries:
        r = None
        try:
            r = http.urlopen(method='PUT',
                            url=url,
                            body=body,
                            headers=make_header(content_length,host=host),
                            timeout=keepalive_time,
                            retries=retries)
        
            #check response to see if successful
            pflags = int(r.data.decode('utf-8')[-2])
            if verbose:
                print("number of bytes sent:",r.tell())
            if (pflags & pflagdict['parse_error']): #once moved to git, get the flag value from nsfrb.pipeline.pflagdict
                if verbose:
                    print("Parse Error, Re-sending...")
                tries += 1
            elif (pflags & pflagdict['datasize_error']):
                if verbose:
                    print("Data Loss Error, Re-sending...")
                tries += 1
            elif (pflags & pflagdict['shape_error']):
                if verbose:
                    print("Shape Error, Re-sending...")
                tries += 1
            else:
                if verbose:
                    print("Success")
                
                    print("Received: " + str(r.data.decode('utf-8')))
                break
        except Exception as exc:
            if type(Exception) == AttributeError:
                if verbose:
                    print("Received NoneType response, Re-sending...")
                tries += 1
            else:
                raise exc
        else:
            if verbose:
                print(r.data.decode('utf-8'))
        finally:
            if r is not None:
                r.close()    
    return "send_data complete"
