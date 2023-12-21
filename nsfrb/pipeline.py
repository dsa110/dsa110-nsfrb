import sys
import numpy as np
import os


##defines function to get shape of np array from raw data
def get_shape_from_raw(data,headersize=128):
    #input is data as bytes

    #get header string
    header = data[1:headersize].decode('utf-8')
    print(header)

    if not ('shape' in header): #no shape available
        return -1

    #find shape data
    startidx = header.index('shape') + len('shape')
    startidx = startidx + header[startidx:].index('(') + 1
    endidx = startidx + header[startidx:].index(')')
    #print(startidx,endidx)
    #loop through and get shape
    shapearr = []
    while startidx < endidx:
        if ',' in header[startidx:endidx]:
            upto = startidx + header[startidx:endidx].index(',')
        else:
            upto = endidx
        #print(upto)
        dim = int(header[startidx:upto])
        #print(dim)
        shapearr.append(dim)

        startidx = upto + 1
    print(tuple(shapearr))
    #return shape as tuple
    return tuple(shapearr)




##defines function to handle output from server piped to standard in


def server_handler(datasize,headersize,chunksize,output_shape=-1,verbose=False):

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

    #find shape of array
    if output_shape == -1:
        output_shape = get_shape_from_raw(dat[:headersize],headersize=headersize)
        if output_shape == -1:
            print("Invalid output shape")
            return -1

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






    

