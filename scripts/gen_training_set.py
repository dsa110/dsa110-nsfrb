import argparse
from scipy.stats import uniform
import random
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
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,tsamp
#from nsfrb.imaging import inverse_uniform_image,uniform_image,inverse_revised_uniform_image,revised_uniform_image, uv_to_pix, revised_robust_image,get_ra
from nsfrb.imaging import uv_to_pix
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

from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,flagged_antennas,Lon,Lat,maxrawsamps,flagged_corrs,inject_log_file

import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
from dsautils import cnf
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
ETCDKEY = f'/mon/nsfrb/inject'

"""
This service will run on h24 and create injections thqt can be rsynced to each corr node for use in the realtime system.
"""
from nsfrb.searching import DM_trials as default_DMtrials
from nsfrb.searching import widthtrials as default_widthtrials
from nsfrb.searching import maxshift
def main(args):

    verbose = args.verbose
    print("Generating Dec=",args.dec,"injections")

    os.system("mkdir " + str(args.outputdir))
    print("will output training set to "+args.outputdir)

    print("Creating "+str(args.N) + " images")
    for inum in range(args.N):

        DM = 0 if args.RFI else default_DMtrials[np.random.choice(np.arange(len(default_DMtrials),dtype=int))]
        width = default_widthtrials[np.random.choice(np.arange(len(default_widthtrials[:-1]),dtype=int))]
        offsetRA = np.random.choice(np.arange(-args.gridsize//4,args.gridsize//4,dtype=int))
        offsetDEC = np.random.choice(np.arange(-args.gridsize//4,args.gridsize//4,dtype=int))
        if len(args.rfi_type)==0:
            args.rfi_type = np.random.choice(['far','near'])
        if args.RFI:
            print(args.rfi_type+"-field RFI")
        SNR = uniform.rvs(loc=10,scale=1000)
        print("DM=",DM)
        print("W=",width)
        print("offsets=",offsetRA,offsetDEC)
        
        #RA, dec axes
        t_now = Time.now()
        mjd = t_now.mjd
        time_start_isot = t_now.isot
        RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,two_dim=False,manual=False,DEC=args.dec)
        HA_axis = RA_axis - RA_axis[int(args.gridsize//2)]
        cleardataflag = args.solo_inject or args.flat_field or args.gauss_field
        injectflatflag = args.point_field or args.gauss_field or args.flat_field
        HA = 0
        RA = RA_axis[int(args.gridsize//2)]
        Dec = Dec_axis[int(args.gridsize//2)]
        noiseless=False
        if args.inject_noiseless:
            noiseless=True
        inject_img = injecting.generate_inject_image(time_start_isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,snr=SNR*1e-9,width=width,loc=0.5,gridsize=args.gridsize,nchans=args.num_chans,nsamps=args.num_time_samples,DM=DM,maxshift=maxshift,offline=args.offline,noiseless=noiseless,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=args.inject_noiseonly,bmin=args.bmin,robust=args.robust if args.briggs else -2,outputdir=args.outputdir,RFI=args.RFI,rfi_type=args.rfi_type)
        if args.flat_field:
            inject_img = np.ones_like(inject_img)
        elif args.gauss_field:
            xx,yy = np.meshgrid(np.linspace(-2,2,args.gridsize),np.linspace(-2,2,args.gridsize))
            inject_img = multivariate_normal(mean=[0,0],cov=0.5).pdf(np.dstack((xx,yy)))
            inject_img = inject_img[:,:,np.newaxis,np.newaxis].repeat(args.num_time_samples,2).repeat(args.num_chans,3)
        elif args.point_field:
            inject_img = np.zeros_like(inject_img)
            inject_img[int(args.gridsize//2)+offsetDEC,int(args.gridsize//2)+offsetRA] = 1

        #np.save(args.outputdir + "/train_DM"+str(DM)+"_W"+str(width)+"_DEC"+str(args.dec)+".npy",inject_img)
        np.save(args.outputdir + "/dataset_"+time_start_isot+"/train_"+str("RFI" if args.RFI else "SRC")+"_"+time_start_isot+".npy",inject_img)
        print("")

        
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate injections for the realtime pipeline')
    parser.add_argument('--N',type=int,default=1,help='Number to generate')
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--solo_inject',action='store_true',default=False,help='If set, visibility data will be zeroed and an injection with simulated noise will overwrite the data')
    parser.add_argument('--offline',action='store_true',default=False,help='Initializes previous frame with noise')
    parser.add_argument('--inject_noiseonly',action='store_true',default=False,help='Only inject noise; for use with false positive testing')
    parser.add_argument('--inject_noiseless',action='store_true',default=False,help='Only inject signal')
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--flat_field',action='store_true',help='Illuminate all pixels uniformly')
    parser.add_argument('--gauss_field',action='store_true',help='Illuminate a gaussian source')
    parser.add_argument('--point_field',action='store_true',help='Illuminate a point source')
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--briggs',action='store_true',help='If set use robust weighted gridding with \'briggs\' weighting')
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging',default=0)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--dec',type=float,help='Declination',default=71.6)
    parser.add_argument('--outputdir',type=str,default=inject_dir + "TMPDATASET/",help='output directory; defaults to temporary directory')
    parser.add_argument('--RFI',action='store_true',help='generate near or far field RFI simulation')
    parser.add_argument('--rfi_type',type=str,choices=['far','near',''],default='',help='Specify type of RFI (near or far)')
    args = parser.parse_args()
    main(args)
