import argparse
from inject import injecting
from nsfrb.outputlogging import printlog
import glob
import csv
from matplotlib import pyplot as plt
from nsfrb.simulating import compute_uvw,get_core_coordinates,get_all_coordinates
#from inject import injecting
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
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize
from nsfrb.imaging import inverse_uniform_image,uniform_image,inverse_revised_uniform_image,revised_uniform_image, uv_to_pix, revised_robust_image,get_ra
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST
from nsfrb.TXclient import send_data
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images
from tqdm import tqdm
import time
from scipy.stats import norm,multivariate_normal
#import nsfrb.searching as sl
from nsfrb.outputlogging import numpy_to_fits
#from nsfrb import calibration as cal
from nsfrb import pipeline
import os
#vispath = os.environ["NSFRBDATA"] + "dsa110-nsfrb-fast-visibilities" #cwd + "-fast-visibilities"
#imgpath = cwd + "-images"
#inject_file = cwd + "-injections/injections.csv"

from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,flagged_antennas,Lon,Lat,maxrawsamps,flagged_corrs,local_inject_dir

import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
from dsautils import cnf
import rtreader
"""
This script continuously pulls data from memory mapped from the rtwriter, images, and sends to the process
server in realtime.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m

#flagged antennas

#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
"""f = open("/home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat","r")
flagged_antennas = np.array(f.read().split("\n")[:-1],dtype=int)
f.close()
"""
CONF = cnf.Conf(use_etcd=True)
CORR_CONF = CONF.get('corr')
CAL_CONF = CONF.get('cal')
MFS_CONF = CONF.get('fringe')
CORRLIST = list(CORR_CONF['ch0'].keys())
CORRLIST = [CORRLIST[i][6:] for i in range(len(CORRLIST))]
NCORR = len(CORRLIST)
CALTIME = CAL_CONF['caltime_minutes']*u.min
FILELENGTH = MFS_CONF['filelength_minutes']*u.min


# ETCD interface
from multiprocessing import Process, Queue
ETCD = ds.DsaStore()
ETCDKEY = f'/mon/nsfrb/fastvis'
ETCDKEY_INJECT = f'/mon/nsfrb/inject'
QQUEUE = Queue()

logfile = ""
def rt_etcd_to_queue(etcd_dict,queue=QQUEUE):
    """
    ETCD watch callback function for /mon/nsfrb/fastvis
    """
    queue.put(etcd_dict['shmid'])
    queue.put(etcd_dict['datasize'])
    queue.put(etcd_dict['mjd'])
    queue.put(etcd_dict['sb'])
    queue.put(etcd_dict['dec'])
    return

def main(args):

    verbose = args.verbose
    
    #attach callback to etcdkey
    printlog("Adding ETCD watch on key "+ETCDKEY,output_file=logfile)
    ETCD.add_watch(ETCDKEY,rt_etcd_to_queue)

    #make a running count and inject a burst every 90x25sample gulps
    if args.inject:
        inject_count = 90

    while True:

        #read and reshape into np array (25 times x 4656 baselines x 8 chans x 2 pols, complex)
        dat = None
        while (dat is None) or dat.shape[0]<args.num_time_samples:
            #wait for shmid to be added to queue
            printlog("Waiting for fast vis data in queue:" + str(QQUEUE),output_file=logfile)
            shmid = QQUEUE.get()
            datasize = QQUEUE.get()
            mjd = QQUEUE.get()
            sb = QQUEUE.get()
            Dec = QQUEUE.get()
            print(mjd,sb,Dec)
            #check this is the right data
            assert(sb==args.sb)

            dat_hex = rtreader.read(shmid,datasize)
            print("DATA HEX:",dat_hex.hex()[-64:])
            dat_i = np.frombuffer(dat_hex,dtype=np.float32)
            
            ntimes = int(len(dat_i)/args.nbase/args.nchans_per_node/2/2)
            dat_i = dat_i.reshape((ntimes,args.nbase,args.nchans_per_node,2,2))
            
            if dat is None:
                dat = np.zeros(dat_i.shape[:-1],dtype=np.complex64)
                dat[:,:,:,:] = dat_i[:,:,:,:,0] + 1j*dat_i[:,:,:,:,1]
            else:
                dat = np.concatenate([dat,dat_i[:,:,:,:,0] + 1j*dat_i[:,:,:,:,1]],axis=0,dtype=np.complex64)

            print(dat.shape)
        
        np.save(img_dir + "2025-02-16T20:36:48.010_rtvis.npy",dat)

        if verbose: print("Collected 25 samples, imaging...")
        #get timestamp
        timestamp = Time(mjd,format='mjd').isot

        #parameters from etcd
        test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
        ff = 1.53-np.arange(8192)*0.25/8192
        fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
        fobs = np.reshape(fobs,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1)
        print("FREQS(",len(fobs),"): ",len(fobs),fobs)

        #use MJD to get pointing
        time_start_isot = Time(mjd,format='mjd').isot
        print("DEC from file:",Dec)
        pt_dec = Dec*np.pi/180.
        if verbose: print("Pointing dec (deg):",pt_dec*180/np.pi)
        bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)


        #flagging andd baseline cut
        fcts = []
        if args.flagSWAVE:
            fcts.append(fct_SWAVE)
        if args.flagBPASS:
            fcts.append(fct_BPASS)
        if args.flagFRCBAND:
            fcts.append(fct_FRCBAND)
        if args.flagBPASSBURST:
            fcts.append(fct_BPASSBURST)
        dat, bname, blen, UVW, antenna_order = flag_vis(dat, bname, blen, UVW, antenna_order, flagged_antennas, bmin, flagged_corrs, flag_channel_templates = fcts, realtime=True, sb=args.sb)
            
        U = UVW[0,:,0]
        V = UVW[0,:,1]
        W = UVW[0,:,2]
        uv_diag=np.max(np.sqrt(U**2 + V**2))
        pixel_resolution = (0.20 / uv_diag) / 3
        if verbose: print(antenna_order,len(antenna_order))#x_m.shape,y_m.shape,z_m.shape)
        if verbose: print(UVW.shape,U.shape,V.shape,W.shape)
        if verbose: print(UVW)

        print("Print bad channels:",np.isnan(dat.mean((0,1,3))))

        RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,DEC=Dec)
        HA_axis = RA_axis[int(len(RA_axis)//2)] - RA_axis
        RA = RA_axis[int(len(RA_axis)//2)]
        HA = HA_axis[int(len(HA_axis)//2)]
        print(HA_axis[len(HA_axis)//2-10:len(HA_axis)//2+10])
        if verbose: print("Time:",time_start_isot)
        if verbose: print("Coordinates (deg):",RA,Dec)
        if verbose: print("Hour angle (deg):",HA)


        #creating injection
        if args.inject and (inject_count>=90):
            inject_count = 0
            print("Injecting pulse")

            #look for an injection in etcd
            injection_params = ETCD.get_dict(ETCDKEY_INJECT)
            if injection_params is None:
                print("Injection not ready, postponing")
                inject_count = 90
            else:
                #update dict
                if 'ISOT' not in injection_params.keys():
                    injection_params['ISOT'] = time_start_isot

                #acknowledge receipt
                injection_params["ack"] = True

                #check if correct time
                if time_start_isot == injection_params['ISOT']:
                    injection_params["injected"][args.sb] = True
                ETCD.put_dict(ETCDKEY_INJECT,injection_params)
                
                if time_start_isot == injection_params['ISOT']:
                    print("Injection",injection_params['ID'],"found")
                    fname = "injection_" + str(injection_params['ID']) + "_sb" +str("0" if args.sb<10 else "")+ str(args.sb) + ".npy"
                    #copy
                    os.system("scp h24.pro.pvt:" + inject_dir + "realtime_staging/" + "injection_" + str(injection_params['ID']) + "_sb" +str("0" if args.sb<10 else "")+ str(args.sb) + ".npy " + local_inject_dir)
                    #read
                    inject_img = np.load(local_inject_dir + fname)
                    #clear data if we only want the injection
                    if injection_params['inject_only']: dat[:,:,:,:] = 0
                    inject_flat = injection_params['inject_flat'] 
                    print("Done injecting")
        else:
            inject_img = np.zeros((args.gridsize,args.gridsize,dat.shape[0]))
        dat[np.isnan(dat)]= 0 

        #imaging
        print("Start imaging")
        if args.wstack: print("W-stacking with ",args.Nlayers," layers")
        dirty_img = np.nan*np.ones((args.gridsize,args.gridsize,dat.shape[0],1))
        for i in range(dat.shape[0]):
            for k in range(dat.shape[-1]):
                    
                for jj in range(args.nchans_per_node):
                    if args.briggs:
                        if k == 0 and jj == 0:
                            dirty_img[:,:,i,0] = revised_robust_image(dat[i:i+1, :, jj, k],U/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),V/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),args.gridsize,inject_img=None if np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/args.nchans_per_node,robust=args.robust,inject_flat=inject_flat,pixel_resolution=pixel_resolution,wstack=args.wstack,w=None if not args.wstack else W/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),Nlayers_w=args.Nlayers)

                        else:
                            dirty_img[:,:,i,0] += revised_robust_image(dat[i:i+1, :, jj, k],U/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),V/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),args.gridsize,inject_img=None if np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/args.nchans_per_node,robust=args.robust,inject_flat=inject_flat,pixel_resolution=pixel_resolution,wstack=args.wstack,w=None if not args.wstack else W/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),Nlayers_w=args.Nlayers)
                    else:
                        if k == 0 and jj == 0:
                            dirty_img[:,:,i,0] = revised_uniform_image(dat[i:i+1, :, jj, k],U/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),V/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),args.gridsize,inject_img=None if np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/args.nchans_per_node,inject_flat=inject_flat,pixel_resolution=pixel_resolution,wstack=args.wstack,w=None if not args.wstack else W/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),Nlayers_w=args.Nlayers)
                        else:
                            dirty_img[:,:,i,0] += revised_uniform_image(dat[i:i+1, :, jj, k],U/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),V/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),args.gridsize,inject_img=None if np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/args.nchans_per_node,inject_flat=inject_flat,pixel_resolution=pixel_resolution,wstack=args.wstack,w=None if not args.wstack else W/(2.998e8/fobs[(args.nchans_per_node*args.sb)+jj]/1e9),Nlayers_w=args.Nlayers)


        print("Imaging complete")            
        print(dirty_img)


        #send to proc server
        if args.save:
            np.save(img_dir + time_start_isot + "_rtimage.npy",dirty_img)
        if args.search:
                
            msg=send_data(time_start_isot, uv_diag, Dec, dirty_img[:,:,:,0] ,verbose=args.verbose,retries=5,keepalive_time=10)
            if args.verbose: print(msg)
                
        if args.inject:
            inject_count += 1
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    #parser.add_argument('filelabel')           # positional argument
    parser.add_argument('sb',type=int)
    parser.add_argument('--timestamp',type=str,help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49); if not given, timestamp is retrieved from sb00 file with os.path.getctime() or from time of rsync',default='')
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=raw_datasize)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vis_dir[:-1])
    parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--inject',action='store_true',default=False,help='Inject a burst into the gridded visibilities. Unless the --solo_inject flag is set, a noiseless injection will be integrated into the data.')
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=1)
    parser.add_argument('--briggs',action='store_true',help='If set use robust weighted gridding with \'briggs\' weighting')
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging',default=0)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--wstack',action='store_true',help='If set use w-stacking algorithm with --Nlayers layers')
    parser.add_argument('--Nlayers',type=int,help='Number of layers for w-stacking',default=18)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--flagSWAVE',action='store_true',help='Flag channels when SWAVE template RFI is detected, which manifests as a 2 Hz sin wave over ~5 minutes of data')
    parser.add_argument('--flagBPASS',action='store_true',help='Flag channels when BPASS template RFI is detected, which is simpl comparison to bandpass mean in visibilities')
    parser.add_argument('--flagFRCBAND',action='store_true',help='Flag channels in FRC miltiary allocation 1435-1525 MHz')
    parser.add_argument('--flagBPASSBURST',action='store_true',help='Flag channel when BPASS template RFI is detected in any timestep, i.e. should detect pulsed narrowband RFI')
    parser.add_argument('--nbase',type=int,help='Expected number of baselines',default=4656)
    args = parser.parse_args()
    main(args)



