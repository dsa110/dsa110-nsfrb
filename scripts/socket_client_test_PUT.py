import urllib3

"""
This script implements the following curl command in python:
    curl --upload-file "${fname}" "http://10.41.0.94:8080/${fname}" -verbose --trace-ascii /media/ubuntu/ssd/sherman/code/here.txt --keepalive-time 15 --http0.9
"""

fname = "subband_avg_1311.39_MHz_ID0000.npy"

#get data parameters
import numpy as np
f = open(fname,"rb")
bdata = f.read()
f.close()


import nsfrb.TXclient as TXC

print(TXC.send_data("ID0000",bdata,ENDFILE='ENDFILE',headersize=128))


"""

datasize = len(bdata)
headersize = 128 #size of 3D numpy array default header in bytes
ENDFILE = 'ENDFILE'
ENDFILEsize = len(ENDFILE) #size of end file message
body = bdata + bytes(ENDFILE,encoding='utf-8')#add end file message
content_length = len(body)


print("Sending " + fname)
print("Message body of size " + str(content_length))



#http parameters
port = 8080
ipaddress = "10.41.0.94" #corr20
host = ipaddress + ":" + str(port)
url = "http://" + host + "/" + fname
keepalive_time = 15
httpversion = 0.9

#for example of headers, see /media/ubuntu/ssd/sherman/code/here.txt
headers=dict()
headers['Host'] = host
headers['User-Agent'] = 'curl/7.78.0'
headers['Accept'] = '*/*'
headers['Referer'] = 'rbose'
headers['Content-Length'] = str(content_length)
headers['Expect'] = '100-continue'


http = urllib3.PoolManager() #https://urllib3.readthedocs.io/en/2.2.0/reference/urllib3.poolmanager.html#urllib3.PoolManager.urlopen
#r = http.request_encode_body(method='PUT',
#        url=url,
#        fields={'--upload-file' : fname}
r=http.urlopen(method='PUT',
        url=url,
        body=body,
        headers=headers,
        timeout=keepalive_time)
print("Response: " + str(r.data.decode('utf-8')))
"""
