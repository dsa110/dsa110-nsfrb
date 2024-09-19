import argparse
from nsfrb.simulating import compute_uvw,get_core_coordinates,get_all_coordinates
import h5py
from casatools import table
import numpy as np
from astropy.time import Time
import sys

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()

sys.path.append(cwd+"/nsfrb/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/nsfrb/")
sys.path.append(cwd+"/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c
from nsfrb.imaging import uniform_image, uv_to_pix
from nsfrb.TXclient import send_data
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images
from tqdm import tqdm
import time
from scipy.stats import norm
import nsfrb.searching as sl
from nsfrb.outputlogging import numpy_to_fits
from nsfrb import calibration as cal
vispath = cwd + "-fast-visibilities"
"""
This script reads raw fast visibility data from a file on disk, applies fringe-stopping from a pre-made table,
applies calibration, and images. If specified, the resulting image is transmitted to the process server.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m

#flagged antennas

flagged_antennas = [21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
"""f = open("/home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat","r")
flagged_antennas = np.array(f.read().split("\n")[:-1],dtype=int)
f.close()
"""
def main(args):
    verbose = args.verbose
    #read raw data for each corr node
    dat_all = None
    for i in range(len(corrs)):
        corr = corrs[i]
        fname = args.path + "/lxd110"+ corr + "/" + corr + args.filelabel + ".out"
        fname = args.path + "/3C286_vis/" + corr + args.filelabel + ".out"
        try:
            dat_corr = cal.read_raw_vis(fname,datasize=args.datasize).mean(2,keepdims=True)
            if verbose: print(dat_corr.shape)
            if dat_all is None:
                dat_all = np.nan*np.ones(dat_corr.shape,dtype=dat_corr.dtype).repeat(len(corrs),axis=2)
            dat_all[:,:,i,:] = dat_corr
        except:
            if verbose: print("No data for " + corr)

    
    #send in sub-gulps
    
    num_gulps = int(dat_all.shape[0]//args.num_time_samples)
    if args.num_gulps != -1:
        num_gulps = np.min([args.num_gulps,num_gulps])
    num_chans = int(NUM_CHANNELS//AVERAGING_FACTOR)
    for gulp in range(num_gulps):
        dat = dat_all[gulp*args.num_time_samples:(gulp+1)*args.num_time_samples,:,:,:]
        print("Gulp size:",dat.shape)

        #use MJD to get pointing
        mjd = Time(args.timestamp,format='isot').mjd + (args.num_time_samples/86400)
        time_start_isot = Time(mjd,format='mjd').isot
        HA,Dec =  uv_to_pix(mjd,1,Lat=37.23,Lon=-118.2851)
        HA = HA[0]
        Dec = Dec[0]
        if verbose: print("Coordinates:",HA,Dec)


        #get UVW coordinates
        x_m,y_m,z_m = get_all_coordinates(flagged_antennas)
        U,V,W = compute_uvw(x_m,y_m,z_m,HA,Dec)
        if verbose: print(x_m.shape,y_m.shape,z_m.shape)
        if verbose: print(U.shape,V.shape,W.shape)
    
    
        #fringe stopping
        if args.fringestop:
            ra_ax,dec_ax = uv_to_pix(mjd,dat.shape[0],Lat=37.23,Lon=-118.2851)
            ra_center,dec_center = ra_ax[0],dec_ax[0]
            for i in range(len(ra_ax)):
                ra_point,dec_point = ra_ax[i],dec_ax[i]
                if verbose: print("Pointing:",ra_point,dec_point)
                for j in range(num_chans):
                    for k in range(dat.shape[-1]):
                        phaseterms = cal.make_phase_table(U/wavs[j],V/wavs[j],W/wavs[j],ra_center,dec_center,ra_point,dec_point,verbose=False)
                        dat[i,:,j,k] *= phaseterms
    
        #image
        dirty_img = np.nan*np.ones((IMAGE_SIZE,IMAGE_SIZE,dat.shape[0],num_chans))
        for i in range(dat.shape[0]):
            for j in range(num_chans):
                for k in range(dat.shape[-1]):
                    if k == 0:
                        dirty_img[:,:,i,j] = uniform_image(dat[i, :, j, k],U,V,IMAGE_SIZE)**2
                    else:
                        dirty_img[:,:,i,j] += uniform_image(dat[i, :, j, k],U,V,IMAGE_SIZE)**2

        #send to proc server
        if args.search:
            for i in range(num_chans):
                #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
                msg=send_data(time_start_isot, dirty_images_all[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10)
                if args.verbose: print(msg)
                time.sleep(1)


    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('filelabel')           # positional argument
    parser.add_argument('timestamp',help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49)')
    parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=-1)
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--fringestop', action='store_true', default=False, help='Fringe stop manually')
    #parser.add_argument('--fringetable',type=str,help='Fringe stop manually with specified table in the dsa110-nsfrb-fast-visibilities dir',default='')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=4)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vispath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    args = parser.parse_args()
    main(args)



