import argparse
from realtime import rtwriter
from dsacalib import constants as ct
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait
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
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,pixperFWHM,chanbw,freq_axis_fullres,lambdaref,c,NSFRB_PSRDADA_KEY
from nsfrb.imaging import inverse_revised_uniform_image,uv_to_pix, revised_robust_image,get_ra,briggs_weighting,uniform_grid
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST
from nsfrb.TXclient import send_data,ipaddress
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

from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,flagged_antennas,Lon,Lat,maxrawsamps,flagged_corrs,timelogfile


"""
This script reads raw fast visibility data from a file on disk, applies fringe-stopping from a pre-made table,
applies calibration, and images. If specified, the resulting image is transmitted to the process server.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","hh16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m

#flagged antennas/
TXtask_list = []
def offline_image_task(dat, U_wavs, V_wavs, i_indices_all, j_indices_all, i_conj_indices_all, j_conj_indices_all, bweights_all, gridsize,  pixel_resolution, nchans_per_node, fobs_j, j, briggs=False, robust= 0.0, return_complex=False, inject_img=None, inject_flat=False, wstack=False, W_wavs=None, k_indices_all=None, k_conj_indices_all=None, Nlayers_w=18,pixperFWHM=pixperFWHM,wstack_parallel=False):#,port=-1,ipaddress="",time_start_isot="", uv_diag=-1, Dec=-1, TXexecutor=None, stagger=0):

    outimage = np.zeros((args.gridsize,args.gridsize,args.num_time_samples))
    for jj in range(nchans_per_node):
        #if briggs:
        #print("INPUT SHAPE",dat[:,:,jj,:].mean(2))#dat[:,:,jj,:].transpose((0,2,1)).shape)
        outimage += revised_robust_image(dat[:,:,jj,:].mean(2),#.transpose((0,2,1)),#dat[i:i+1, :, jj, k],
                                            U_wavs[:,jj],
                                            V_wavs[:,jj],
                                            gridsize,
                                            inject_img=None if np.all(inject_img==0) else inject_img/dat.shape[-1]/nchans_per_node,
                                            robust=robust,
                                            uniform=(not briggs),
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,
                                            wstack=wstack,
                                            w=None if W_wavs is None else W_wavs[:,jj],
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM,
                                            briggs_weights=None if (not briggs) else bweights_all[:,jj],
                                            i_indices=i_indices_all[:,jj],
                                            j_indices=j_indices_all[:,jj],
                                            i_conj_indices=i_conj_indices_all[:,jj],
                                            j_conj_indices=j_conj_indices_all[:,jj],
                                            clipuv=False,keeptime=True,wstack_parallel=wstack_parallel)
    return outimage,j




#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
def main(args):

    verbose = args.verbose
    #send in sub-gulps
    
    num_gulps = 1#int(dat_all.shape[0]//args.num_time_samples)
    if args.num_gulps != -1:
        num_gulps = args.num_gulps#np.min([args.num_gulps,num_gulps])

    for gulp in range(args.gulp_offset,args.gulp_offset + num_gulps):
        
        #if searching, also need to find the previous integration set so we can initialize previous frame
        filelabels = [args.filelabel]
        #read raw data for each corr node
        for g in range(len(filelabels)):

            #parameters from etcd
            test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
            #ff = 1.53-np.arange(8192)*0.25/8192
            #fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
            fobs = (np.reshape(freq_axis_fullres,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1))*1e-3
            #dat = dat_all[gulp*args.num_time_samples:(gulp+1)*args.num_time_samples,:,:,:]


            if filelabels[g] != args.filelabel:
                print("Making image to initialize last frame")
            dat = None
            Dec = None
            for i in range(len(corrs)):
                corr = corrs[i]
                sb = sbs[i]

                if len(args.filedir) == 0:
                    fname = args.path + "/lxd110"+ corr + "/" + ("nsfrb_" + sb if args.sb else corr) + filelabels[g] + ".out"
                else:
                    fname =  args.filedir + "/" + ("nsfrb_" + sb if args.sb else corr) + filelabels[g] + ".out"
                print(fname)
                #fname = args.path + "/lxd110"+ corr + "/" + corr + args.filelabel + ".out"
                #fname = args.path + "/3C286_vis/" + corr + args.filelabel + ".out"
                try:
                    #tmp = cal.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples)
                    #print("tmp",tmp)

                    dat_corr,sbnum,tstamp_mjd,Dec = pipeline.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples,gulp=(gulp if filelabels[g]==args.filelabel else (maxrawsamps//args.num_time_samples)-1),nchan=int(args.nchans_per_node),headersize=16)
            
                    #send
                    hdr = {'MJD': str(tstamp_mjd),
                            'SB': str(sbnum),
                            'DEC': str(Dec)
                            }
                    print(hdr)
                    rtwriter.rtwrite(np.concatenate([np.real(dat_corr)[:,:,:,:,np.newaxis],np.imag(dat_corr)[:,:,:,:,np.newaxis]],axis=4),key=NSFRB_PSRDADA_KEY,addheader=True,header=hdr,dtype=np.float32)
                except Exception as exc:
                    if verbose: print("No data for " + corr)
                    if verbose: print(exc)
        time.sleep(args.sleeptime)
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('filelabel')           # positional argument
    parser.add_argument('--timestamp',type=str,help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49); if not given, timestamp is retrieved from sb00 file with os.path.getctime() or from time of rsync',default='')
    parser.add_argument('--filedir',type=str,help='Path to fast visibilities; if not given, the /dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/lxd110h**/ paths are used',default='')
    parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=-1)
    parser.add_argument('--gulp_offset',type=int,help='Gulp offset to start from, default = 0', default=0)
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=raw_datasize)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vis_dir[:-1])
    parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--num_inject',type=int,help='Number of injections, must be less than number of gulps',default=1)
    parser.add_argument('--sb',action='store_true',default=False,help='Use nsfrb_sbxx names')
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=8)
    parser.add_argument('--sleeptime',type=float,help='Time to sleep between processing gulps (seconds)',default=3)
    args = parser.parse_args()
    main(args)



