import argparse
import csv
from matplotlib import pyplot as plt
from nsfrb.simulating import compute_uvw,get_core_coordinates,get_all_coordinates
from inject import injecting
import h5py
from casatools import table
import numpy as np
from astropy.time import Time
import astropy.units as u
import sys
from dsamfs import utils as pu
from dsautils import cnf
from collections import OrderedDict
my_cnf = cnf.Conf(use_etcd=True)

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()

sys.path.append(cwd+"/nsfrb/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/nsfrb/")
sys.path.append(cwd+"/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize
from nsfrb.imaging import inverse_uniform_image,uniform_image, uv_to_pix
from nsfrb.TXclient import send_data
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images
from tqdm import tqdm
import time
from scipy.stats import norm
import nsfrb.searching as sl
from nsfrb.outputlogging import numpy_to_fits
from nsfrb import calibration as cal
vispath = cwd + "-fast-visibilities"
imgpath = cwd + "-images"
inject_file = cwd + "-injections/injections.csv"
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
    """
    #read raw data for each corr node
    dat_all = None
    for i in range(len(corrs)):
        corr = corrs[i]
        fname = args.path + "/lxd110"+ corr + "/" + corr + args.filelabel + ".out"
        fname = args.path + "/3C286_vis/" + corr + args.filelabel + ".out"
        try:
            tmp = cal.read_raw_vis(fname,datasize=args.datasize)
            #print("tmp",tmp)
            dat_corr = np.nanmean(cal.read_raw_vis(fname,datasize=args.datasize),axis=2,keepdims=True)
            if verbose: print(dat_corr.shape)
            if dat_all is None:
                dat_all = np.nan*np.ones(dat_corr.shape,dtype=dat_corr.dtype).repeat(len(corrs),axis=2)
            #print(dat_all.shape,dat_corr.shape)
            dat_all[:,:,i,:] = dat_corr[:,:,0,:]
            #print("tmp2",dat_all[:,:,i,:],dat_corr)
        except Exception as exc:
            if verbose: print("No data for " + corr)
            if verbose: print(exc)
    """
    #send in sub-gulps
    
    num_gulps = 1#int(dat_all.shape[0]//args.num_time_samples)
    if args.num_gulps != -1:
        num_gulps = args.num_gulps#np.min([args.num_gulps,num_gulps])
    num_chans = int(NUM_CHANNELS//AVERAGING_FACTOR)

    #randomly choose which gulp to inject burst in
    if args.inject:
        inject_gulp = np.random.choice(np.arange(num_gulps,dtype=int))

    #parameters from etcd
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)


    for gulp in range(num_gulps):
        #dat = dat_all[gulp*args.num_time_samples:(gulp+1)*args.num_time_samples,:,:,:]
        
        #read raw data for each corr node
        dat = None
        for i in range(len(corrs)):
            corr = corrs[i]
            fname = args.path + "/lxd110"+ corr + "/" + corr + args.filelabel + ".out"
            fname = args.path + "/3C286_vis/" + corr + args.filelabel + ".out"
            try:
                #tmp = cal.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples)
                #print("tmp",tmp)
                dat_corr = np.nanmean(cal.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples,gulp=gulp),axis=2,keepdims=True)
                if verbose: print(dat_corr.shape)
                if dat is None:
                    dat = np.nan*np.ones(dat_corr.shape,dtype=dat_corr.dtype).repeat(len(corrs),axis=2)
                #print(dat_all.shape,dat_corr.shape)
                dat[:,:,i,:] = dat_corr[:,:,0,:]
                #print("tmp2",dat_all[:,:,i,:],dat_corr)
            except Exception as exc:
                if verbose: print("No data for " + corr)
                if verbose: print(exc)
        
        
        
        print("Gulp size:",dat.shape)

        #use MJD to get pointing
        mjd = Time(args.timestamp,format='isot').mjd + (gulp*args.num_time_samples*tsamp/86400)
        time_start_isot = Time(mjd,format='mjd').isot
        LST = Time(mjd,format='mjd').sidereal_time("mean",longitude=-118.2851).to(u.hourangle).value
        print("Time:",time_start_isot)
        print("LST (hr):",LST)
        RA_axis,Dec_axis = uv_to_pix(mjd,IMAGE_SIZE,Lat=37.23,Lon=-118.2851)
        HA_axis = (LST*15) - RA_axis
        #HA_axis = RA_axis - RA_axis[int(len(RA_axis)//2)] #want to image the central RA, so the hour angle should be 0 here, right?
        RA = RA_axis[int(len(RA_axis)//2)]
        HA = HA_axis[int(len(HA_axis)//2)]
        Dec = Dec_axis[int(len(Dec_axis)//2)]

        if verbose: print("Coordinates (deg):",RA,Dec)
        if verbose: print("Hour angle (deg):",HA)

        #get antenna positions coordinates
        #x_m,y_m,z_m,antenna_names = get_all_coordinates(flagged_antennas,return_names=True) #meters
        """
        #re-order based on antenna order from etcd
        my_cnf = cnf.Conf(use_etcd=True)
        corr_cnf = my_cnf.get('corr')
        antenna_order = list(OrderedDict(sorted(corr_cnf['antenna_order'].items())).values())
        mfs_cnf = my_cnf.get('fringe')
        refmjd = mfs_cnf['refmjd']


        #get UVWs
        #U,V,W = compute_uvw(x_m,y_m,z_m,HA,Dec) #meters
        bname, blen, UVW = pu.baseline_uvw(antenna_order, Dec*np.pi/180, refmjd, casa_order=False,autocorrs=True) #include autocorrelations
        """
        #get UVW from etcd
        #test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
        pt_dec = Dec*np.pi/180.
        bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)

        U = UVW[0,:,0]
        V = UVW[0,:,1]
        W = UVW[0,:,2]
        if verbose: print(antenna_order,len(antenna_order))#x_m.shape,y_m.shape,z_m.shape)
        if verbose: print(UVW.shape,U.shape,V.shape,W.shape)
        if verbose: print(UVW)
        #if verbose: print("core idxs",len(core_idxs),core_idxs)
   
        """
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
                        print(dat[i,:,j,k])
                        print(phaseterms)
                        dat[i,:,j,k] *= phaseterms
    
        """
        #calibrating
        #*** TO DO: INSERT NIKITA'S CALIBRATION CODE HERE***#


        #creating injection
        if args.inject and (gulp == inject_gulp):
            offsetRA,offsetDEC,SNR,width,DM,maxshift = injecting.draw_burst_params(time_start_isot,RA_axis=RA_axis,DEC_axis=Dec_axis,gridsize=IMAGE_SIZE,nsamps=dat.shape[0],nchans=num_chans,tsamp=tsamp*1000)

            if args.snr_inject > 0:
                SNR = args.snr_inject
            if args.dm_inject != -1 and args.dm_inject >= 0:
                DM = args.dm_inject
            if args.width_inject > 0:
                width = args.width_inject
            print("PARAMSFROM OFFLINE IMAGER:",offsetRA,offsetDEC,SNR,width,DM,maxshift,tsamp)
            if args.solo_inject:
                noiseless=False
                dat[:,:,:,:] = 0
            else:
                noiseless=True
            #DM = 0
            #SNR = 10000
            #width = 2
            #offsetRA = offsetDEC = 0
            inject_img = injecting.generate_inject_image(HA=HA_axis[int(len(HA_axis)//2 + offsetRA)],DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,snr=SNR,width=width,loc=0.5,gridsize=IMAGE_SIZE,nchans=num_chans,nsamps=dat.shape[0],DM=DM,maxshift=maxshift,offline=args.offline,noiseless=noiseless)


            #report injection in log file
            with open(inject_file,"a") as csvfile:
                wr = csv.writer(csvfile,delimiter=',')
                wr.writerow([time_start_isot,DM,width,SNR])
            csvfile.close()


        else:
            inject_img = np.zeros((IMAGE_SIZE,IMAGE_SIZE,dat.shape[0],num_chans))
        
        #imaging
        dirty_img = np.nan*np.ones((IMAGE_SIZE,IMAGE_SIZE,dat.shape[0],num_chans))
        for i in range(dat.shape[0]):
            for j in range(num_chans):
                for k in range(dat.shape[-1]):
                    
                    """
                    if i == 0 and j == 2 and k == 0:
                        plt.figure(figsize=(12,12))
                        plt.plot(np.real(dat[i, :, j, k]),np.real(inverse_uniform_image(uniform_image(dat[i:i+1, :, j, k],U,V,IMAGE_SIZE,return_complex=True),U,V)),'o')
                        plt.xscale("log")
                        plt.yscale("log")
                        plt.plot(np.linspace(0,1e5),np.linspace(0,1e5))
                        plt.xlim(1,1e5)
                        plt.ylim(1,1e5)
                        plt.savefig("tmp4.png")

                        plt.close()
                    """
                    if k == 0:
                        dirty_img[:,:,i,j] = uniform_image(dat[i:i+1, :, j, k],U,V,IMAGE_SIZE,inject_img=inject_img[:,:,i,j])
                    else:
                        dirty_img[:,:,i,j] += uniform_image(dat[i:i+1, :, j, k],U,V,IMAGE_SIZE,inject_img=inject_img[:,:,i,j])
        #save image to fits, numpy file
        if args.save:
            np.save(args.outpath + "/" + time_start_isot + ".npy",dirty_img)
            numpy_to_fits(np.nanmean(dirty_img,(2,3)),args.outpath + "/" + time_start_isot + ".fits")

        #send to proc server
        if args.search:
            for i in range(num_chans):
                #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
                msg=send_data(time_start_isot, dirty_img[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10)
                if args.verbose: print(msg)
                time.sleep(1)


    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('filelabel')           # positional argument
    parser.add_argument('timestamp',help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49)')
    parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=-1)
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    #parser.add_argument('--fringestop', action='store_true', default=False, help='Fringe stop manually')
    #parser.add_argument('--fringetable',type=str,help='Fringe stop manually with specified table in the dsa110-nsfrb-fast-visibilities dir',default='')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=4)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vispath)
    parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--inject',action='store_true',default=False,help='Inject a burst into the gridded visibilities. Unless the --solo_inject flag is set, a noiseless injection will be integrated into the data.')
    parser.add_argument('--solo_inject',action='store_true',default=False,help='If set, visibility data will be zeroed and an injection with simulated noise will overwrite the data')
    parser.add_argument('--snr_inject',type=float,help='SNR of injection; default 0 which chooses a random SNR',default=0)
    parser.add_argument('--dm_inject',type=float,help='DM of injection; default -1 which chooses a random DM',default=-1)
    parser.add_argument('--width_inject',type=int,help='Width of injection in samples; default 0 which chooses a random width',default=0)
    parser.add_argument('--offline',action='store_true',default=False,help='Initializes previous frame with noise')
    args = parser.parse_args()
    main(args)



