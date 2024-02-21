import numpy as np
import urllib3
import logging
from http.client import HTTPConnection
import sys
sys.path.append("..")
from nsfrb.pipeline import pflagdict
HTTPConnection.debuglevel = 0
"""
This script implements the following curl command in python:
    curl --upload-file "${fname}" "http://10.41.0.94:8080/${fname}" -verbose --trace-ascii /media/ubuntu/ssd/sherman/code/here.txt --keepalive-time 15 --http0.9
The function send_data() should be called following imaging to send image data
to the persistent server for searching. For the realtime system, this should be called once on each corr node per 3.25 s integration to send a 256x256x25x1 image cube to T4. For the offline system, this should be called once for each sub-band image file.

"""

#parameters that never change

#http parameters
port = 8080
ipaddress = "10.41.0.94" #corr20
host = ipaddress + ":" + str(port)
#url = "http://" + host + "/" + fname
keepalive_time = 15
httpversion = 0.9
retries = 5

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


def send_data(timestamp,array,node=23,ENDFILE='ENDFILE',headersize=128,verbose=False):
    #timestamp is how sub-bands will be associated with each other, so ensure its the same for all sub-bands; should be in isot format: 'yyyy:mm:ddThh:mm:ss'
    #array is numpy array of sub-band data
    if node < 10:
        sb = "sb0" + str(node)
    elif node >= 10 and node < 15:
        sb = "sb" + str(node)
    elif node== 23:
        sb = "_h23"
    else:
        print("Invalid corr node")

    #make filename
    fname = sb + "_IMG" + str(timestamp) + ".npy"
    url = host + "/" + fname

    #convert numpy data to bytes
    if type(array) == bytes:
        body = array[:-1] + bytes(ENDFILE,encoding='utf-8') #note we had to remove the newline
    else:
        body = array.tobytes() + bytes(ENDFILE,encoding='utf-8')
    content_length = len(body)


    #print(body.decode('utf-8'))

    """
    f = open("TXoutput.txt","w")
    f.write(body.hex())
    f.close()
    """
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
    
    tries = 0
    while tries < retries:
        http = urllib3.PoolManager() #https://urllib3.readthedocs.io/en/2.2.0/reference/urllib3.poolmanager.html#urllib3.PoolManager.urlopen
        r=http.urlopen(method='PUT',
            url=url,
            body=body,
            headers=make_header(content_length),
            timeout=keepalive_time,
            retries=retries)

        #check response to see if successful
        pflags = int(r.data.decode('utf-8')[-2])
        if (pflags & pflagdict['parse_error']): #once moved to git, get the flag value from nsfrb.pipeline.pflagdict
            print("Parse Error, Re-sending...")
            tries += 1
        elif (pflags & pflagdict['datasize_error']):
            print("Data Loss Error, Re-sending...")
            tries += 1
        else:
            print("Success")
            break
    #rjson = r.info()
    #print(rjson)
    #f = open("TXoutput.txt","w")
    #f.write(rjson['data'].hex())
    #f.close() 
    #http.close()
    return str(r.data.decode('utf-8'))
