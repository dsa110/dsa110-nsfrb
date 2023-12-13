import sys
import numpy as np
import os



##defines function to handle output from server piped to standard in


def server_handler(datasize,headersize,chunksize,output_shape,verbose=False):

    #define empty byte array and status string to keep track of data size
    alldat = np.array([]).tobytes() 
    statusstring = ""

    while len(statusstring) < datasize:
        #read data chunk
        dat = os.read(0,chunksize)
        if verbose:
            print(len(alldat))
        alldat = alldat + dat
        statusstring += str(dat[2:-1])

    #decode hex data
    if verbose:
        print(dat,len(dat),len(alldat))
    alldatstr = alldat.decode('utf-8')

    #convert to bytes
    bytedat = bytes.fromhex(alldatstr)
    if verbose:
        print(len(bytedat))

    #convert to numpy array
    arrdat = np.frombuffer(bytedat[headersize:datasize+headersize]).reshape((32,32,25,16))
    return arrdat








