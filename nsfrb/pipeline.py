import csv
import struct
from matplotlib import pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, ICRS,SkyCoord
from astropy.time import Time
import astropy.units as u
from influxdb import DataFrameClient
import os
import sys

#cwd = os.environ['NSFRBDIR']
#sys.path.append(cwd + "/")
from nsfrb.config import *
from nsfrb.noise import noise_update,noise_dir,noise_update_all
from nsfrb import jax_funcs
"""
#output_dir = cwd + "/tmpoutput/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/"
coordfile = cwd + "/DSA110_Station_Coordinates.csv" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/DSA110_Station_Coordinates.csv"
output_file = cwd + "-logfiles/search_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt"
cand_dir = cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
processfile = cwd + "-logfiles/process_log.txt"
frame_dir = cwd + "-frames/"
psf_dir = cwd + "-PSF/"
f=open(output_file,"w")
f.close()
"""
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file





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
pflagdict['shape_error'] = 4
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

    
# functions for reading raw visibility data
influx = DataFrameClient('influxdbservice.pro.pvt', 8086, 'root', 'root', 'dsa110')
def read_raw_vis(fname,datasize=4,nbase=4656,nchan=384,npol=2,nsamps=-1,gulp=0,headersize=16,oldformat=False,get_header=False):
    """
    Read raw visibility data from given file.
    fname: file name
    datasize: size of data in bytes
    headersize: size of header in bytes (1/2 headersize is sub-band number, 1/2 headersize is mjd as float)
    """
    if datasize==4:
        dtype = np.float32
        dtypecomplex = np.complex64
    elif datasize==2:
        dtype = np.float16
        dtypecomplex = np.complex32
    elif datasize==8:
        dtype = np.float64
        dtypecomplex = np.complex128
    elif datasize==16:
        dtype = np.float128
        dtypecomplex = np.complex256
    else:
        print("Invalid datasize")
        return None

    f = open(fname,"rb")

    #first read header
    if headersize != 0:
        if headersize == 12:
            mjd = struct.unpack(('>' if sys.byteorder=='big' else '<') + 'f', f.read(headersize//3))[0]
            sbnum = int.from_bytes(f.read(headersize//3),sys.byteorder,signed=False)
            if oldformat:
                dec = np.nan
            else:
                dec = struct.unpack(('>' if sys.byteorder=='big' else '<') + 'f', f.read(headersize//3))[0]
        elif headersize == 16:
            mjd = struct.unpack(('>' if sys.byteorder=='big' else '<') + 'd', f.read(headersize//2))[0]
            sbnum = int.from_bytes(f.read(headersize//4),sys.byteorder,signed=False)
            if oldformat:
                dec = np.nan
            else:
                dec = struct.unpack(('>' if sys.byteorder=='big' else '<') + 'f', f.read(headersize//4))[0]
        else:
            print("Invalid headersize")
            return None
    if get_header:
        f.close()
        return sbnum,mjd,dec

    if nsamps == -1:
        raw_data = np.frombuffer(f.read(),dtype=dtype) #default reads all time samples
    else:
        print("Reading from byte",headersize + gulp*nsamps*nbase*nchan*npol*2*datasize)
        f.seek(headersize + gulp*nsamps*nbase*nchan*npol*2*datasize)
        raw_data = np.frombuffer(f.read(nsamps*nbase*nchan*npol*2*datasize),dtype=dtype)
    f.close()

    ntimes = int(len(raw_data)/nbase/nchan/npol/2)

    dat = raw_data.reshape((ntimes,nbase,nchan,npol,2))

    #real and imaginary
    dat_complex = np.zeros(dat.shape[:-1],dtype=dtypecomplex)
    dat_complex[:,:,:,:] = dat[:,:,:,:,0] + 1j*dat[:,:,:,:,1]
    if headersize == 0:
        return dat_complex#,0,Time.now().mjd()
    else:
        return dat_complex,sbnum,mjd,dec

