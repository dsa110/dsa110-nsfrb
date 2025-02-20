from psrdada import Reader
from nsfrb.config import NSFRB_PSRDADA_KEY,nsamps
import numpy as np

"""
helper functions to read visibilities from a psrdada buffer and converts them to a numpy array.
"""

def read_buffer_multisamp(reader, nbls, nchan, npol,nsamps,dtype=np.float32,dtypecomplex=np.float64):
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
    

    page = reader.getNextPage()
    reader.markCleared()
    print(page,type(page))
    data = np.frombuffer(page.tobytes(),dtype=dtype)
    data = data.view(dtype)
    data = data.reshape(-1, 2).view(dtypecomplex).squeeze(axis=-1)
    try:
        data = data.reshape(nsamps, nbls, nchan, npol)
    except ValueError:
        print(
            f"incomplete data: {data.shape[0]%(nbls*nchan*npol*nsamps)} out of {nbls*nchan*npol*nsamps} samples")
        data = data[
            :data.shape[0] // (nbls * nchan * npol*nsamps) * (nbls * nchan * npol*nsamps)
        ].reshape(nsamps, nbls, nchan, npol)
    return data

def rtread(key=NSFRB_PSRDADA_KEY,nbls=4656,nchan=8,npol=2,nsamps=nsamps,datasize=4):
    """
    reads from psrdada specified by key provided

    key: PSRDada buffer identifier, default 0xdada
    nbls: number of baselines
    nchan: number of channels
    npol: number of polarizations
    nsamps: number of time samples
    """
    
    #datasize
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


    #make reader
    print(f"Initializing reader: " + str(key))
    reader = Reader(key)

    #check its connected
    if not reader.isConnected:
        print("Reaer not connected")
        tup = None
    else:
        #read header
        header = reader.getHeader()
        print(header)
        mjd = np.float64(header['MJD'])
        sb = int(header['SB'])
        dec = np.float32(header['DEC'])

        #read buffer
        data = read_buffer_multisamp(reader,nbls,nchan,npol,nsamps,dtype=dtype,dtypecomplex=dtypecomplex)
        tup = (data,mjd,sb,dec)
    #disconnect reader
    try:
        reader.disconnect()
    except Exception as e:
        pass
    return tup




