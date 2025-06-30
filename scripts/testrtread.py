from psrdada import Reader
from nsfrb.config import NSFRB_PSRDADA_KEY,nsamps,NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,IMAGE_SIZE
import time

#while True:
for i in range(10):
    t1=time.time()
    reader = Reader(NSFRB_PSRDADA_KEY)
    print(reader.isConnected)

    page = reader.getNextPage()
    
    print("Read",len(page.tobytes()),"bytes in",time.time()-t1,"seconds")
    reader.markCleared()


    #disconnect reader
    try:
        reader.disconnect()
    except Exception as e:
        print(e)
