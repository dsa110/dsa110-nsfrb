import numpy as np
import urllib3

"""
This script implements the following curl command in python:
    curl --upload-file "${fname}" "http://10.41.0.94:8080/${fname}" -verbose --trace-ascii /media/ubuntu/ssd/sherman/code/here.txt --keepalive-time 15 --http0.9
"""

#parameters that never change

#http parameters
port = 8080
ipaddress = "10.41.0.94" #corr20
host = ipaddress + ":" + str(port)
#url = "http://" + host + "/" + fname
keepalive_time = 15
httpversion = 0.9

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


def send_data(timestamp,array,node=23,ENDFILE='ENDFILE',headersize=128):
    #timestamp is how sub-bands will be associated with each other, so ensure its the same for all sub-bands; should be in isot format: 'yyyy:mm:ddThh:mm:ss'
    #array is numpy array of sub-band data
    if node < 10:
        sb = "sb0" + str(node)
    elif node >= 10 and node < 15:
        sb = "sb" + str(node)
    elif node== 23:
        sb = "h23"
    else:
        print("Invalid corr node")

    #make filename
    fname = "subband_avg_" + sb + "_" + str(timestamp) + ".npy"
    url = host + "/" + fname

    #convert numpy data to bytes
    if type(array) == bytes:
        body = array + bytes(ENDFILE,encoding='utf-8')
    else:
        body = array.tobytes() + bytes(ENDFILE,encoding='utf-8')
    content_length = len(body)

    print("Sending " + fname)
    print("Message body of size " + str(content_length))
    




    http = urllib3.PoolManager() #https://urllib3.readthedocs.io/en/2.2.0/reference/urllib3.poolmanager.html#urllib3.PoolManager.urlopen
    r=http.urlopen(method='PUT',
        url=url,
        body=body,
        headers=make_header(content_length),
        timeout=keepalive_time)
    return str(r.data.decode('utf-8'))
