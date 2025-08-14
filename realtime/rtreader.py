from psrdada import Reader
import sys
import time
from nsfrb.config import NSFRB_PSRDADA_KEY,nsamps,NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,IMAGE_SIZE,DSAX_PSRDADA_KEY
import numpy as np

"""
helper functions to read visibilities from a psrdada buffer and converts them to a numpy array.
"""

def read_buffer_multisamp(reader, nbls, nchan, npol,nsamps,dtype=np.float32,dtypecomplex=np.float64,verbose=False,verbosefile=sys.stdout,verbosefile2=sys.stdout):
    """
    Reads a psrdada buffer as float32 and returns the visibilities.

    Parameters
    ----------
    reader : psrdada Reader instance
        An instance of the Reader class for the psrdada buffer to read.
    nbls : int
        The number of baselines.
    nchan : int
        The number of frequency channels.
    npol : int
        The number of polarizations.
    nsamps : int
        Number of time samples

    Returns
    -------
    ndarray
        The data. Dimensions (time, baselines, channels, polarization).
    """
    
    if verbose: t1 = time.time()
    page = reader.getNextPage()
    if verbose: print("...|TIME TO GET NEXT PAGE:",time.time()-t1,file=verbosefile2)
    if verbose: t1 = time.time()
    reader.markCleared()
    if verbose: print("...|TIME TO CLEAR PAGE:",time.time()-t1,file=verbosefile2)
    if verbose: t1 = time.time()
    #print(page,type(page))
    data = np.frombuffer(page.tobytes(),dtype=dtype)
    data = data.view(dtype)
    data = data.reshape(-1, 2).view(dtypecomplex).squeeze(axis=-1)
    try:
        data = data.reshape(nsamps, nbls, nchan, npol)
    except ValueError:
        print(
            f"incomplete data: {data.shape[0]%(nbls*nchan*npol*nsamps)} out of {nbls*nchan*npol*nsamps} samples",file=verbosefile)
        data = data[
            :data.shape[0] // (nbls * nchan * npol*nsamps) * (nbls * nchan * npol*nsamps)
        ].reshape(nsamps, nbls, nchan, npol)
    if verbose: print("...|TIME TO REFORMAT DATA:",time.time()-t1,file=verbosefile2)
    return data

def rtread(key=NSFRB_PSRDADA_KEY,nbls=4656,nchan=8,npol=2,nsamps=nsamps,datasize=4,readheader=False,reader=None,verbose=False,verbosefile=sys.stdout,verbosefile2=sys.stdout):
    """
    reads from psrdada specified by key provided

    key: PSRDada buffer identifier, default 0xdada
    nbls: number of baselines
    nchan: number of channels
    npol: number of polarizations
    nsamps: number of time samples
    """
    
    #datasize
    if verbose:
        t1=time.time()
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
        print("Invalid datasize",file=verbosefile)
        return None


    #make reader
    localreader=False
    if reader is None:
        print(f"Initializing reader: " + str(key),file=verbosefile)
        reader = Reader(key)
        localreader=True
    
    if verbose:
        print("...|SETUP TIME:",time.time()-t1,file=verbosefile2)
    #check its connected
    if not reader.isConnected:
        print("Reaer not connected",file=verbosefile)
        tup = None
    else:
        if readheader:
            #read header
            header = reader.getHeader()
            #print(header)
            mjd = np.float64(header['MJD'])
            sb = int(header['SB'])
            dec = np.float32(header['DEC'])
        
        #read buffer
        tup = read_buffer_multisamp(reader,nbls,nchan,npol,nsamps,dtype=dtype,dtypecomplex=dtypecomplex,verbose=verbose,verbosefile=verbosefile,verbosefile2=verbosefile2)
        if readheader:
            tup = (tup,mjd,sb,dec)
    #disconnect reader
    if localreader:
        try:
            reader.disconnect()
        except Exception as e:
            pass
    return tup


def read_buffer_multisamp_cand(reader, gridsize_dec,gridsize_ra, nsamps,nchans,dtype=np.float32):
    """
    Reads a psrdada buffer as float32 and returns the visibilities.

    Parameters
    ----------
    reader : psrdada Reader instance
        An instance of the Reader class for the psrdada buffer to read.
    nsamps : int
        Number of time samples

    Returns
    -------
    ndarray
        The data. Dimensions (time, baselines, channels, polarization).
    """


    page = reader.getNextPage()
    reader.markCleared()
    print(page,type(page))
    data = np.frombuffer(page.tobytes(),dtype=dtype)
    data = data.view(dtype)
    
    try:
        data = data.reshape((gridsize_dec,gridsize_ra, nsamps,nchans))
    except ValueError:
        print(
            f"incomplete data: {data.shape[0]%(gridsize_dec*gridsize_ra*nsamps*nchans)} out of {gridsize_dec*gridsize_ra*nsamps*nchans} samples")
        data = data[
            :data.shape[0] // (gridsize_dec*gridsize_ra*nsamps*nchans) * (gridsize_dec*gridsize_ra*nsamps*nchans)
        ].reshape(gridsize_dec,gridsize_ra, nsamps,nchans)
    return data

def rtread_cand(key=NSFRB_CANDDADA_KEY,gridsize_dec=IMAGE_SIZE,gridsize_ra=IMAGE_SIZE,nsamps=nsamps,nchans=16,datasize=4):
    """
    reads from psrdada specified by key provided

    key: PSRDada buffer identifier, default 0xdada
    

    """

    #datasize
    if datasize==4:
        dtype = np.float32
    elif datasize==2:
        dtype = np.float16
    elif datasize==8:
        dtype = np.float64
    elif datasize==16:
        dtype = np.float128
    else:
        print("Invalid datasize")
        return None
    
    #make reader
    print(f"Initializing reader: " + str(key))
    reader = Reader(key)

    #check its connected
    if not reader.isConnected:
        print("Reaer not connected")
        data = None
    else:
        """
        #read header
        header = reader.getHeader()
        print(header)
        mjd = np.float64(header['MJD'])
        sb = int(header['SB'])
        dec = np.float32(header['DEC'])
        """
        #read buffer
        data = read_buffer_multisamp_cand(reader,gridsize_dec,gridsize_ra,nsamps,nchans,dtype=dtype)
    #disconnect reader
    try:
        reader.disconnect()
    except Exception as e:
        pass
    return data


def read_imaging_buffer_multisamp(reader, gridsize,nsamps,dtype=np.float32,dtypecomplex=np.float64,verbose=False,verbosefile=sys.stdout,verbosefile2=sys.stdout):
    """
    Reads a psrdada buffer as float32 and returns the visibilities.

    Parameters
    ----------
    reader : psrdada Reader instance
        An instance of the Reader class for the psrdada buffer to read.
    nbls : int
        The number of baselines.
    nchan : int
        The number of frequency channels.
    npol : int
        The number of polarizations.
    nsamps : int
        Number of time samples

    Returns
    -------
    ndarray
        The data. Dimensions (time, baselines, channels, polarization).
    """

    if verbose: t1 = time.time()
    page = reader.getNextPage()
    if verbose: print("...|TIME TO GET NEXT PAGE:",time.time()-t1,file=verbosefile2)
    if verbose: t1 = time.time()
    reader.markCleared()
    if verbose: print("...|TIME TO CLEAR PAGE:",time.time()-t1,file=verbosefile2)
    if verbose: t1 = time.time()
    #print(page,type(page))
    data = np.frombuffer(page.tobytes(),dtype=dtype)
    data = data.view(dtype)
    #data = data.reshape(-1, 2).view(dtypecomplex).squeeze(axis=-1)
    try:
        data = data.reshape(gridsize, gridsize,nsamps )#.transpose((1,2,0))
    except ValueError:
        print(
            f"incomplete data: {data.shape[0]%(nsamps*gridsize*gridsize)} out of {nsamps*gridsize*gridsize} samples",file=verbosefile)
        data = data[
            :data.shape[0] // (gridsize*gridsize*nsamps) * (nsamps*gridsize*gridsize)
        ].reshape( gridsize, gridsize, nsamps)#.transpose((1,2,0))
    if verbose: print("...|TIME TO REFORMAT DATA:",time.time()-t1,file=verbosefile2)
    return data

def rtread_imaging(key=DSAX_PSRDADA_KEY,gridsize=301,nsamps=nsamps,datasize=8,reader=None,verbose=False,verbosefile=sys.stdout,verbosefile2=sys.stdout):
    """
    reads from psrdada specified by key provided

    key: PSRDada buffer identifier, default 0xdada
    nbls: number of baselines
    nchan: number of channels
    npol: number of polarizations
    nsamps: number of time samples
    """

    #datasize
    if verbose:
        t1=time.time()
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
        print("Invalid datasize",file=verbosefile)
        return None


    #make reader
    localreader=False
    if reader is None:
        print(f"Initializing reader: " + str(key),file=verbosefile)
        reader = Reader(key)
        localreader=True

    if verbose:
        print("...|SETUP TIME:",time.time()-t1,file=verbosefile2)
    #check its connected
    if not reader.isConnected:
        print("Reaer not connected",file=verbosefile)
        tup = None
    else:
        #read buffer
        tup = read_imaging_buffer_multisamp(reader,gridsize,nsamps,dtype=dtype,dtypecomplex=dtypecomplex,verbose=verbose,verbosefile=verbosefile,verbosefile2=verbosefile2)
    #disconnect reader
    if localreader:
        try:
            reader.disconnect()
        except Exception as e:
            pass
    return tup
