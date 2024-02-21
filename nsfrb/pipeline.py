import sys
import numpy as np
import os

output_file = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt"
f=open(output_file,"w")
f.close()

##defines function to get shape of np array from raw data
def get_shape_from_raw(data,headersize=128,output_file=output_file):
    #input is data as bytes
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #get header string
    header = bytes.fromhex(data[:2*headersize].decode('utf-8'))[1:].decode('utf-8')#data[1:headersize].decode('utf-8')
    print("header: ",header,file=fout)
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
    print("ouptut shape: ",tuple(shapearr),file=fout)
    #return shape as tuple
    if output_file != "":
        fout.close()
    return tuple(shapearr)




##defines function to handle output from server piped to standard in


def server_handler(datasize,headersize,chunksize,output_shape=-1,bytesize=-1,output_file=output_file):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout


    #define empty byte array and status string to keep track of data size
    alldat = np.array([]).tobytes()
    statusstring = ""
    dat = np.array([1,2,3]).tobytes() #dummy data
    while len(alldat) < datasize:#len(statusstring) < datasize and len(dat) > 0:
        #read data chunk
        #print(".",end=" ")
        dat = os.read(0,chunksize)
        #if verbose:
        #print(len(alldat))
        if len(alldat)%128000 == 0:
            print("...",end="",file=fout,flush=True)
        if len(alldat)%512000 == 0:
            print("read " + str(len(alldat))+" bytes",end="",file=fout,flush=True)
        alldat = alldat + dat
        statusstring += str(dat[2:-1])
        sys.stdout.flush()

    #print(bytes.fromhex(alldat[:2*headersize].decode('utf-8')))#find shape of array
    if output_shape == -1:
        output_shape = get_shape_from_raw(alldat[:headersize*2],headersize=headersize)
        if output_shape == -1:
            print("Invalid output shape",output_shape)
            return -1

    #decode hex data
    print("decoding hex data; ", len(dat),len(alldat),file=fout)
    alldatstr = alldat.decode('utf-8')
    print("after decode:",len(alldatstr),file=fout)

    #convert to bytes
    #if bytesize==16:
    #    print(alldatstr)
    bytedat = bytes.fromhex(alldatstr)
    if bytesize == 16:
        dtype = np.float16
    elif bytesize == 32:
        dtype = np.float32
    elif bytesize == 64:
        dtype = np.float64
    else:
        dtype = float
    
    print("after hex to bytes:",len(bytedat),file=fout)
    print(np.frombuffer(bytedat[headersize:],dtype=dtype),file=fout)#datasize+headersize]))
    #convert to numpy array
    arrdat = np.frombuffer(bytedat[headersize:],dtype=dtype).reshape(output_shape)#datasize+headersize]).reshape(output_shape)#(32,32,25,16))
    if output_file != "":
        fout.close()
    return arrdat



##defines function to convert numpy array to string of hex bytes and prints to stdout

def pipeout(arr,output_file=output_file):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    print("piping data of shape ", arr.shape, " to stdout...",end="",file=fout,flush=True)
    if type(arr) != np.ndarray:
        print("must be np.ndarray")
        return -1
    print(arr.tobytes().hex())
    print("Done!",file=fout)
    if output_file != "":
        fout.close()
    return len(arr.tobytes().hex())




##defines function to set flags for process server
pflagdict = dict()
pflagdict['parse_error'] = 1
pflagdict['datasize_error'] = 2
pflagdict['invalid'] = 8
pflagdict['all'] = 15
flagfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_flags.txt"
def set_pflag(flag=None,on=True,reset=False):
    if (flag != None) and (not (flag in pflagdict.keys())): return None
    
    with open(flagfile,"r") as flagfileio:
        pflags = int(flagfileio.read()) 
        flagfileio.close()
    if (flag==None) and (not reset):
        return pflags 

    #make sure the invalid flag is unset
    pflags = pflags & ~pflagdict['invalid']

    if reset: pflags = 8
    elif on: pflags = pflags | pflagdict[flag]
    else: pflags = pflags & ~pflagdict[flag]
    with open(flagfile,"w") as flagfileio:
        flagfileio.write(str(int(pflags)))
        flagfileio.close()
    return pflags

    

