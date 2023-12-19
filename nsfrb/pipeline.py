import sys
import numpy as np
import os



##defines function to handle output from server piped to standard in


def server_handler(datasize,headersize,chunksize,output_shape,verbose=False):

    #define empty byte array and status string to keep track of data size
    alldat = np.array([]).tobytes()
    statusstring = ""
    dat = np.array([1,2,3]).tobytes() #dummy data
    while len(alldat) < datasize:#len(statusstring) < datasize and len(dat) > 0:
        #read data chunk
        #print(".",end=" ")
        dat = os.read(0,chunksize)
        if verbose:
            #print(len(alldat))
            if len(alldat)%128000 == 0:
                print("...",end="")
            if len(alldat)%512000 == 0:
                print("read " + str(len(alldat))+" bytes",end="")
        alldat = alldat + dat
        statusstring += str(dat[2:-1])
        sys.stdout.flush()


    #decode hex data
    if verbose:
        print(len(dat),len(alldat))
    alldatstr = alldat.decode('utf-8')

    #convert to bytes
    bytedat = bytes.fromhex(alldatstr)
    if verbose:
        print(len(bytedat))
        print(np.frombuffer(bytedat[headersize:datasize+headersize]))
    #convert to numpy array
    arrdat = np.frombuffer(bytedat[headersize:datasize+headersize]).reshape(output_shape)#(32,32,25,16))
    return arrdat



##defines function to convert numpy array to string of hex bytes and prints to stdout

def pipeout(arr):
    if type(arr) != np.ndarray:
        print("must be np.ndarray")
        return -1
    print(arr.tobytes().hex())
    return 0




