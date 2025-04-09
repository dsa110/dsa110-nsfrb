import numpy as np
from astropy import units as u
import glob
from nsfrb import planning
from nsfrb import pipeline
from nsfrb import config
def main():
    fpr_set = open(config.table_dir + "/fpr_set.txt","w")
    fnames = np.sort(glob.glob(config.vis_dir + "/lxd110h03/*.out"))
    final_fnums = []
    for f in fnames:
        flag = 1
        #get mjd, dec, ra
        sb,mjd,dec = pipeline.read_raw_vis(f,nchan=8,nsamps=25,gulp=0,headersize=16,get_header=True)
        #look for pulsars, lpts
        for i in [0,45,89]:
            psrcoord,psrnames = planning.atnf_cat(mjd + (i*25*config.tsamp/1000/86400),dec,2.5*u.deg)
            lptnames,lptcoord,tmp,tmp,tmp = planning.LPT_cat(mjd + (i*25*config.tsamp/1000/86400),dec,2.5*u.deg)
            if len(psrcoord)>0 or len(lptnames)>0:
                flag = 0
                break
        if flag == 1:
            print(f)
            fpr_set.write(f + "\n")
    fpr_set.close()
    return
if __name__=="__main__":
    main()
