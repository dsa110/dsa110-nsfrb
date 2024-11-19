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

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()

#sys.path.append(cwd+"/nsfrb/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/nsfrb/")
#sys.path.append(cwd+"/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize
from nsfrb.imaging import inverse_uniform_image,uniform_image, uv_to_pix, robust_image
from nsfrb.TXclient import send_data
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images
from tqdm import tqdm
import time
from scipy.stats import norm,multivariate_normal
import nsfrb.searching as sl
from nsfrb.outputlogging import numpy_to_fits
#from nsfrb import calibration as cal
from nsfrb import pipeline
import os
#vispath = os.environ["NSFRBDATA"] + "dsa110-nsfrb-fast-visibilities" #cwd + "-fast-visibilities"
#imgpath = cwd + "-images"
#inject_file = cwd + "-injections/injections.csv"

from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file


"""
This script reads raw fast visibility data from a file on disk, applies fringe-stopping from a pre-made table,
applies calibration, and images. If specified, the resulting image is transmitted to the process server.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
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
    #send in sub-gulps
    
    num_gulps = 1#int(dat_all.shape[0]//args.num_time_samples)
    if args.num_gulps != -1:
        num_gulps = args.num_gulps#np.min([args.num_gulps,num_gulps])
    #num_chans = int(NUM_CHANNELS//AVERAGING_FACTOR)

    #randomly choose which gulp to inject burst in
    if args.inject:
        inject_gulp = np.random.choice(np.arange(num_gulps,dtype=int))

    #parameters from etcd
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)

    if verbose: print("TIMESTAMP:",tsamp)
    #get timestamp
    if len(args.timestamp) != 0:
        timestamp = args.timestamp

    for gulp in range(num_gulps):
        #dat = dat_all[gulp*args.num_time_samples:(gulp+1)*args.num_time_samples,:,:,:]
        
        #read raw data for each corr node
        dat = None
        for i in range(len(corrs)):
            corr = corrs[i]
            sb = sbs[i]

            if len(args.filedir) == 0:
                fname = args.path + "/lxd110"+ corr + "/" + ("nsfrb_" + sb if args.sb else corr) + args.filelabel + ".out"
            else:
                fname =  args.path + "/" + args.filedir + "/" + ("nsfrb_" + sb if args.sb else corr) + args.filelabel + ".out"

            #fname = args.path + "/lxd110"+ corr + "/" + corr + args.filelabel + ".out"
            #fname = args.path + "/3C286_vis/" + corr + args.filelabel + ".out"
            try:
                #tmp = cal.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples)
                #print("tmp",tmp)

                dat_corr,sbnum,tstamp_mjd = pipeline.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples,gulp=gulp,nchan=int(args.nchans_per_node),headersize=8)
                if len(args.timestamp) == 0: 
                    timestamp = Time(tstamp_mjd,format='mjd').isot
            
                dat_corr = np.nanmean(dat_corr,axis=2,keepdims=True)
                if verbose: print(dat_corr.shape)
                if dat is None:
                    dat = np.nan*np.ones(dat_corr.shape,dtype=dat_corr.dtype).repeat(len(corrs),axis=2)
                #print(dat_all.shape,dat_corr.shape)
                dat[:,:,i,:] = dat_corr[:,:,0,:]
                #print("tmp2",dat_all[:,:,i,:],dat_corr)
            except Exception as exc:
                if verbose: print("No data for " + corr)
                if verbose: print(exc)
        

        
        if verbose: print("Gulp size:",dat.shape)

        #use MJD to get pointing
        mjd = Time(timestamp,format='isot').mjd + (gulp*args.num_time_samples*tsamp/86400)
        time_start_isot = Time(mjd,format='mjd').isot
        LST = Time(mjd,format='mjd').sidereal_time("mean",longitude=-118.2851).to(u.hourangle).value
        if verbose: print("Time:",time_start_isot)
        if verbose: print("LST (hr):",LST)
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
            offsetRA,offsetDEC,SNR,width,DM,maxshift = injecting.draw_burst_params(time_start_isot,RA_axis=RA_axis,DEC_axis=Dec_axis,gridsize=IMAGE_SIZE,nsamps=dat.shape[0],nchans=args.num_chans,tsamp=tsamp*1000)
            #offsetRA = offsetDEC = 0

            if args.snr_inject > 0:
                SNR = args.snr_inject
            if args.dm_inject != -1 and args.dm_inject >= 0:
                DM = args.dm_inject
            if args.width_inject > 0:
                width = args.width_inject
            offsetRA = args.offsetRA_inject
            offsetDEC = args.offsetDEC_inject
            print("PARAMSFROM OFFLINE IMAGER:",offsetRA,offsetDEC,SNR,width,DM,maxshift,tsamp)
            print("OFFSET HOUR ANGLE:",HA_axis[int(len(HA_axis)//2 + offsetRA)])
            noiseless=False
            if args.solo_inject or args.flat_field or args.gauss_field:
                #noiseless=False
                dat[:,:,:,:] = 0
            if args.inject_noiseless:
                noiseless=True
            #noiseless = True
            #DM = 0
            #SNR = 10000
            #width = 2
            #offsetRA = offsetDEC = 0
            inject_img = injecting.generate_inject_image(time_start_isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,snr=SNR,width=width,loc=0.5,gridsize=IMAGE_SIZE,nchans=args.num_chans,nsamps=dat.shape[0],DM=DM,maxshift=maxshift,offline=args.offline,noiseless=noiseless,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=args.inject_noiseonly)

            if args.flat_field:
                inject_img = np.ones_like(inject_img)
            elif args.gauss_field:
                xx,yy = np.meshgrid(np.linspace(-2,2,IMAGE_SIZE),np.linspace(-2,2,IMAGE_SIZE))
                inject_img = multivariate_normal(mean=[0,0],cov=0.5).pdf(np.dstack((xx,yy)))
                inject_img = inject_img[:,:,np.newaxis,np.newaxis].repeat(dat.shape[0],2).repeat(args.num_chans,3)
            elif args.point_field:
                inject_img = np.zeros_like(inject_img)
                inject_img[int(IMAGE_SIZE//2)+offsetDEC,int(IMAGE_SIZE//2)+offsetRA] = 1
            #report injection in log file
            with open(inject_file,"a") as csvfile:
                wr = csv.writer(csvfile,delimiter=',')
                wr.writerow([time_start_isot,DM,width,SNR])
            csvfile.close()


        else:
            inject_img = np.zeros((IMAGE_SIZE,IMAGE_SIZE,dat.shape[0],args.num_chans))
        
        #imaging
        dirty_img = np.nan*np.ones((IMAGE_SIZE,IMAGE_SIZE,dat.shape[0],args.num_chans))
        for i in range(dat.shape[0]):
            for j in range(args.num_chans):
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
                    if args.briggs:
                        if k == 0:
                            dirty_img[:,:,i,j] = robust_image(dat[i:i+1, :, j, k],U,V,IMAGE_SIZE,args.robust,inject_img=inject_img[:,:,i,j]/dat.shape[-1],inject_flat=(args.point_field or args.gauss_field or args.flat_field))
                        else:
                            dirty_img[:,:,i,j] += robust_image(dat[i:i+1, :, j, k],U,V,IMAGE_SIZE,args.robust,inject_img=inject_img[:,:,i,j]/dat.shape[-1],inject_flat=(args.point_field or args.gauss_field or args.flat_field))
                    else:
                        if k == 0:
                            dirty_img[:,:,i,j] = uniform_image(dat[i:i+1, :, j, k],U,V,IMAGE_SIZE,inject_img=inject_img[:,:,i,j]/dat.shape[-1],inject_flat=(args.point_field or args.gauss_field or args.flat_field))
                        else:
                            dirty_img[:,:,i,j] += uniform_image(dat[i:i+1, :, j, k],U,V,IMAGE_SIZE,inject_img=inject_img[:,:,i,j]/dat.shape[-1],inject_flat=(args.point_field or args.gauss_field or args.flat_field))
        #save image to fits, numpy file
        if args.save:
            np.save(args.outpath + "/" + time_start_isot + ".npy",dirty_img)
            numpy_to_fits(np.nanmean(dirty_img,(2,3)),args.outpath + "/" + time_start_isot + ".fits")
            
            if args.inject:
                np.save(args.outpath + "/" + time_start_isot + "_response.npy",dirty_img/inject_img)
                numpy_to_fits(np.nanmean(dirty_img,(2,3))/np.nanmean(inject_img,(2,3)),args.outpath + "/" + time_start_isot + "_response.fits")        

        #send to proc server
        if args.search:
            for i in range(args.num_chans):
                #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
                msg=send_data(time_start_isot, dirty_img[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10)
                if args.verbose: print(msg)
                time.sleep(1)

        time.sleep(args.sleeptime)
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('filelabel')           # positional argument
    parser.add_argument('--timestamp',type=str,help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49); if not given, timestamp is retrieved from sb00 file with os.path.getctime() or from time of rsync',default='')
    parser.add_argument('--filedir',type=str,help='Path to fast visibilities; if not given, the /dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/lxd110h**/ paths are used',default='')
    parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=-1)
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    #parser.add_argument('--fringestop', action='store_true', default=False, help='Fringe stop manually')
    #parser.add_argument('--fringetable',type=str,help='Fringe stop manually with specified table in the dsa110-nsfrb-fast-visibilities dir',default='')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=4)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vis_dir[:-1])
    parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--inject',action='store_true',default=False,help='Inject a burst into the gridded visibilities. Unless the --solo_inject flag is set, a noiseless injection will be integrated into the data.')
    parser.add_argument('--solo_inject',action='store_true',default=False,help='If set, visibility data will be zeroed and an injection with simulated noise will overwrite the data')
    parser.add_argument('--snr_inject',type=float,help='SNR of injection; default -1 which chooses a random SNR',default=-1)
    parser.add_argument('--dm_inject',type=float,help='DM of injection; default -1 which chooses a random DM',default=-1)
    parser.add_argument('--width_inject',type=int,help='Width of injection in samples; default -1 which chooses a random width',default=-1)
    parser.add_argument('--offsetRA_inject',type=int,help='Offset RA of injection in samples; default random', default=int(np.random.choice(np.arange(-IMAGE_SIZE//2,IMAGE_SIZE//2))))
    parser.add_argument('--offsetDEC_inject',type=int,help='Offset DEC of injection in samples; default random', default=int(np.random.choice(np.arange(-IMAGE_SIZE//2,IMAGE_SIZE//2))))
    parser.add_argument('--offline',action='store_true',default=False,help='Initializes previous frame with noise')
    parser.add_argument('--inject_noiseonly',action='store_true',default=False,help='Only inject noise; for use with false positive testing')
    parser.add_argument('--inject_noiseless',action='store_true',default=False,help='Only inject signal')
    parser.add_argument('--sb',action='store_true',default=False,help='Use nsfrb_sbxx names')
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=1)
    parser.add_argument('--flat_field',action='store_true',help='Illuminate all pixels uniformly')
    parser.add_argument('--gauss_field',action='store_true',help='Illuminate a gaussian source')
    parser.add_argument('--point_field',action='store_true',help='Illuminate a point source')
    parser.add_argument('--briggs',action='store_true',help='If set use robust weighted gridding with \'briggs\' weighting')
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging',default=0)
    parser.add_argument('--sleeptime',type=float,help='Time to sleep between processing gulps (seconds)',default=30)
    args = parser.parse_args()
    main(args)



