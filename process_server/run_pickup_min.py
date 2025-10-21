from realtime import rtwriter
from astropy.time import Time
import sys
import numpy as np
from nsfrb import config
from realtime import rtreader
def imagetoDADA(corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port,key=config.NSFRB_SRCHDADA_KEY):
    numdata = np.array([corr_node,img_id_mjd,img_uv_diag,img_dec,port]+list(shape),dtype=np.float64)
    alldata = np.concatenate([numdata,arrData.flatten()])
    rtwriter.rtwrite(alldata,key=key,addheader=False,dtype=np.float64) #bytes(numdata) + bytes(img_id_isot,encoding='utf-8') + bytes(arrData)
    return
def imagefromDADA(key=config.NSFRB_SRCHDADA_KEY,reader=None,datasizebytes=6125064):
    data = np.frombuffer(rtreader.rtread_searching(key,reader=None,verbose=True,verbosefile=sys.stdout))
    corr_node,img_id_mjd,img_uv_diag,img_dec,port = data[:5]
    shape = tuple(data[5:8].astype(int))
    img_id_isot = Time(img_id_mjd,format='mjd').isot
    arrData = data[8:].reshape(tuple(list(shape)+[16]))
    return corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port



def main():
    while True:
        print("waiting...")
        corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape,arrData,port = imagefromDADA(datasizebytes=98000064)
        print(corr_node,img_id_isot,img_id_mjd,img_uv_diag,img_dec,shape)


if __name__=="__main__":
    main()
