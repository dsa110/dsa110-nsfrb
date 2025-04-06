"""
A quick script to write to a psrdada buffer in order to test a psrdada reader.
Copied from dsa110-meridian-fs
"""

import os
import subprocess
from time import sleep
import numpy as np
from psrdada import Writer

KEY_STRING = 'adad'
KEY = 0xadad
NANT = 64
NCHAN = 8#384
NPOL = 2
NBLS = 4656#NANT * (NANT + 1) // 2
NSAMPS = 5

def main():
    """Writes a psrdada buffer for test"""
    vis_temp = np.arange(NSAMPS * NBLS * NCHAN * NPOL * 2, dtype=np.float32)

    # Define the data rate, including the buffer size
    # and the header size
    samples_per_frame = 1
    header_size = 4096
    buffer_size = int(4 * NSAMPS * NBLS * NPOL * NCHAN * samples_per_frame * 2)
    assert buffer_size == vis_temp.nbytes, (
        "Sample data size and buffer size do not match.")

    # Create the buffer
    #os.system(f"dada_db -a {header_size} -b {buffer_size} -k {KEY_STRING}")
    #print("Buffer created")

    # Start the reader
    """
    read = (
        "python ./meridian_fringestop.py /home/ubuntu/data/ "
        "/home/ubuntu/proj/dsa110-shell/dsa110-meridian-fs/dsamfs/data/test_parameters.yaml "
        "/home/ubuntu/proj/dsa110-shell/dsa110-meridian-fs/dsamfs/data/test_header.txt")
    """
    #with open("write.log", 'w', encoding='utf-8') as read_log:
    #with subprocess.Popen(read, shell=True, stdout=read_log, stderr=read_log) as _read_proc:
    #print("Reader started")
    #sleep(5)

    # Write to the buffer
    writer = Writer(KEY)
    print('Writer created')

    #write header
    writer.setHeader({
        'MJD': str(60724.16435968821),
        'SB': str(15),
        'DEC': str(23.665929794311523)
    })
    
    print("wrote header",writer.header)

    #following https://github.com/TRASAL/psrdada-python/blob/master/examples/write_eod_iterators.py
    npage = 0
    npages =1
    for npage in range(1,npages+1):
        page = writer.getNextPage()
        print("start loop")
        data = np.asarray(page)
        data.fill(npage)
        print("got next page",npage)
        writer.markFilled()
        time.sleep(1)

    writer.markEndOfData()
    """
        if npage == npages:
            writer.markEndOfData()
        else:
            writer.markFilled()
        time.sleep(1)
        print(npage)
    
    for i in range(48):
        page = writer.getNextPage()
        data = np.asarray(page)
        data[...] = vis_temp.view(np.int8)
        if i < 9:
            writer.markFilled()
        else:
            writer.markEndOfData()
            vis_temp += 1
            # Wait to allow reader to clear pages
        sleep(1)
    """
    writer.disconnect()

    #os.system(f"dada_db -d -k {KEY_STRING}")


if __name__ == "__main__":
    main()
