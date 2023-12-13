#from pipeline import server_handler
import sys
import numpy as np
import os



##defines function to handle output from server piped to standard in


def server_handler(datasize,headersize,chunksize,output_shape,verbose=False):

    #define empty byte array and status string to keep track of data size
    alldat = np.array([]).tobytes()
    statusstring = ""
    dat = np.array([1,2,3]).tobytes() #dummy data
    while len(statusstring) < datasize and len(dat) > 0:
        #read data chunk
        dat = os.read(0,chunksize)
        #if verbose:
        #    print(len(alldat))
        alldat = alldat + dat
        statusstring += str(dat[2:-1])

    #decode hex data
    if verbose:
        print(alldat,len(dat),len(alldat))
    alldatstr = alldat.decode('utf-8')

    #convert to bytes
    bytedat = bytes.fromhex(alldatstr)
    if verbose:
        print(bytedat)
        print(np.frombuffer(bytedat[headersize:datasize+headersize]))
    #convert to numpy array
    arrdat = np.frombuffer(bytedat[headersize:datasize+headersize]).reshape(output_shape)#(32,32,25,16))
    return arrdat




def main():
    datasize = 209408#6553600#6553600#6553600#6553600#6553600#6553600#6553472#3276928*2#409600
    headersize = 0
    chunksize = 128
    output_shape = (32,32,25,16)

    data = server_handler(datasize=datasize,headersize=headersize,chunksize=chunksize,output_shape=output_shape,verbose=True)
    print(data)
    print(data.shape)
    return

if __name__=="__main__":
    main()
