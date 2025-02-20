#!/usr/bin/env python
"""
Show how to send multiple datasets, separated by EODs.
from https://github.com/TRASAL/psrdada-python/blob/master/examples/write_eod_iterators.py

Version with iterors.
"""
import time
from random import randint, choice
from string import ascii_lowercase
import numpy as np
import os
from psrdada import Writer
from nsfrb.pipeline import read_raw_vis
from nsfrb.outputlogging import printlog
log_file = os.environ['NSFRBDIR'] + "/realtime/realtime_imager_log.txt"
def random_string(length=10):
    """Generate a random string of given length."""
    return ''.join(choice(ascii_lowercase) for i in range(length))


writer = Writer(0xdada)

# send 10 datasets, separated by an EOD
ndataset=0
while True: #for ndataset in range(1):
    npages = 1#randint(1, 10)

    #read data for test
    fname = "/dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/GP_observations_2025-02-20T00:00:00.000/nsfrb_sb12_181517.out"
    test_data_complex,sb,mjd,dec = read_raw_vis(fname,nchan=8,nsamps=25,headersize=16)
    test_data = np.zeros(list(test_data_complex.shape) + [2],dtype=np.float32)
    test_data[:,:,:,:,0] = test_data_complex.real
    test_data[:,:,:,:,1] = test_data_complex.imag
    printlog(test_data.shape,output_file=log_file)

    # setting a new header also resets the buffer: isEndOfData = False
    writer.setHeader({
        'DATASET': str(ndataset),
        'PAGES': str(npages),
        'MAGIC': random_string(20),
        'MJD': str(mjd),
        'SB': str(sb),
        'DEC': str(dec)
    })
    printlog(writer.header,output_file=log_file)

    npage = 0
    for page in writer:
        data = np.asarray(page)
        data[:] = np.frombuffer(bytes(test_data.flatten()),dtype=np.uint8)#data.fill(npage)
        npage += 1

        # at the last iteration, mark the page with EOD;
        # this will also raise a StopIteration exception
        if npage == npages:
            writer.markEndOfData()
        else:
            # like this, you can see better what happens using dada_dbmonitor
            time.sleep(0.5)
    ndataset+=1
# Send a message to the reader that we're done
writer.setHeader({'QUIT': 'YES'})

writer.disconnect()
