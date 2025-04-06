from psrdada import Writer
from time import sleep
import numpy as np
from nsfrb.config import nsamps,NSFRB_CANDDADA_KEY

"""
Functions for writing to psrdada buffer
"""

def rtwrite(image,key=NSFRB_CANDDADA_KEY):
    """
    Writes to a psrdada buffer as float32

    Parameters
    ----------
    image: image data to write (301 x 301 pixels x 25 samples x 16 channels)

    """
    image = image.astype(np.float32).flatten()
    
    #create writer
    writer = Writer(key)

    #write image as single page
    page = writer.getNextPage()
    data = np.asarray(page)
    data[...] = image.view(np.int8)
    writer.markEndOfData()

    sleep(1)
    try:
        writer.disconnect()
        return 0
    except Exception as exc:
        print("Error on writer disconnect:",exc)
    return 1
