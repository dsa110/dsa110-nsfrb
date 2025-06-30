# config.py
import numpy as np
from dsacalib import constants as ct

# Constants
NUM_CHANNELS = 768
AVERAGING_FACTOR = 48
IMAGE_SIZE = 301#301#300  # pixels
pixperFWHM = 3

# Speed of light
c = ct.C_GHZ_M*1E9  # m/s

# Channel information
CH0 = 1311.387  # MHz
CH_WIDTH = 0.244141  # MHz

# Time parameters
tsamp = 0.1342182159423828*1000 #130 #ms
bin_slow = 5 #number of samples to bin by
tsamp_slow = 0.1342182159423828*1000*bin_slow #ms
baseband_tsamp = 256e-3 #ms
nsamps = 25
T = tsamp*nsamps #3250 #ms
bin_imgdiff = 3 #number of samples to bin by
tsamp_imgdiff = bin_imgdiff*T
ngulps_per_file = 90
#nsamps = int(T/tsamp)

# Image channel information
"""
nchans = int(NUM_CHANNELS/AVERAGING_FACTOR)
chanbw = CH_WIDTH*AVERAGING_FACTOR
fmax  = CH0 + CH_WIDTH * (nchans-1) * AVERAGING_FACTOR #1530 #MHz
fmin = CH0 #1280  #MHz
fc = (fmin+fmax)/2#1400 #MHz
"""
nchans = int(NUM_CHANNELS//AVERAGING_FACTOR)
freq_axis_fullres = 1000*((1.53-np.arange(8192)*0.25/8192)[1024:1024+int(nchans*NUM_CHANNELS/2)]) #MHz
freq_axis = np.reshape(freq_axis_fullres,(nchans,int(NUM_CHANNELS/2))).mean(axis=1) #MHz
chanbw = np.abs(freq_axis[0]-freq_axis[1]) #MHz
fmax = np.max(freq_axis) #MHz
fmin = np.min(freq_axis) #MHz
fc =  (fmin+fmax)/2 #MHz


lambdamin = (c/(fmax*1e6)) #m
lambdamax = (c/(fmin*1e6)) #m
lambdac = (c/(fc*1e6)) #m
lambdaref = (c/(freq_axis_fullres[0]*1e6))
#nchans = 16 #16 coarse channels
#chanbw = (fmax-fmin)/nchans #MHz
telescope_diameter = 4.65 #m
DM_tol = 1.6
DM_tol_slow = 1.2

#resolution parameters
pixsize = 0.002962513099862611#(48/3600)*np.pi/180 #rad
gridsize = IMAGE_SIZE#301#301#300#256
RA_point = 0 #rad
DEC_point = 0 #rad
UVMAX = 2316.5744224010487 #maximum UV extent for uniform gridding

vis_to_img_slope_not_binned = 6.320421766399212e-05 #slope relating noise in visibilities to std noise in image estimated from simulation
vis_to_img_slope = 0.0025062597643136777 #same, but with PSF smoothing; noise increases with contribution from PSF, but so does signal

#outrigger flagging and short baseline flagging
bmin=20 #meters
robust=-2 #default to uniform weighting
#flagged_antennas = [48,103,104,105,106,107,108,109,110,111,112,113,114,115,116]
outrigger_antennas = [103,104,105,106,107,108,109,110,111,112,113,114,115,116]
flagged_corrs = []
bad_antennas = [48]#,85,76,77,78,48,36,37,30]
flagged_antennas = bad_antennas+outrigger_antennas

import numpy as np
noise_data_type = np.float64

#file system
import os
import sys

#directories
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
frame_dir = cwd + "-frames/"
psf_dir = cwd + "-PSF/"
img_dir = cwd + "-images/"
if 'NSFRBDATA' in os.environ:
    cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/"
    vis_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/"
    raw_cand_dir = cand_dir + "raw_cands/"
    backup_cand_dir = cand_dir + "backup_raw_cands/"#cwd + "-candidates/backup_raw_cands/"
    final_cand_dir = cand_dir + "final_cands/"#cwd + "-candidates/final_cands/"
    training_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-training/"
remote_cand_dir = cwd +"-tmp-candidates/"
sslogfile = cwd + "-logfiles/srchstartstoptime_log.txt"
inject_dir = inject_file = cwd + "-injections/"
local_inject_dir = cwd + "/inject/realtime_staging_sb/"
noise_dir = cwd + "-noise/"
imgpath = cwd + "-images"
plan_dir = cwd + "-plans/"
table_dir = cwd + "-tables/"
candplotfile = cwd + "-candplotserver/lastcandplot.png"
candplotfile_slow = cwd + "-candplotserver/lastcandplot_slow.png"
candplotfile_imgdiff = cwd + "-candplotserver/lastcandplot_imgdiff.png"
candplotupdatefile = cwd + "-candplotserver/cand.txt"
psr_dir = cwd + "-pulsar/"
#data files
coordfile = cwd + "/DSA110_Station_Coordinates.csv" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/DSA110_Station_Coordinates.csv"

#log files

output_file = cwd + "-logfiles/search_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt"
processfile = cwd + "-logfiles/process_log.txt"
timelogfile = cwd + "-logfiles/time_log.txt"
cutterfile = cwd + "-logfiles/candcutter_log.txt"
pipestatusfile = cwd + "/src/.pipestatus.txt"
searchflagsfile = cwd + "/scripts/script_flags/searchlog_flags.txt"
run_file = cwd + "-logfiles/run_log.txt"
processfile = cwd + "-logfiles/process_log.txt"
cutterfile = cwd + "-logfiles/candcutter_log.txt"
cuttertaskfile = cwd + "-logfiles/candcuttertask_log.txt"
flagfile = cwd + "/process_server/process_flags.txt"
error_file = cwd + "-logfiles/error_log.txt"
inject_file = cwd + "-injections/injections.csv"
recover_file = cwd + "-injections/recoveries.csv"
binary_file = cwd + "-logfiles/binary_log.txt"
inject_log_file = cwd + "-logfiles/inject_log.txt"
rtbench_file = cwd + "-logfiles/rttimes_log.txt"
rttx_file = cwd + "-logfiles/rttx_log.txt"
srchtx_file = cwd + "-logfiles/srchtx_log.txt"
srchtime_file = cwd + "-logfiles/srchtime_log.txt"
candcutter_memory_file = cwd + "-logfiles/candmem_log.txt"
candcutter_time_file = cwd + "-logfiles/candtime_log.txt"

import casatools as cc
me = cc.measures()
obs=me.observatory("OVRO_MMA")
Lat=obs['m1']['value']*180/np.pi#37.23
Lon=obs['m0']['value']*180/np.pi#-118.2851
Height = obs['m2']['value'] #m
az_offset = 0
raw_datasize = 4 #bytes


#astrometry parameters
crpix_dict = {16.27:{"source":"3C138",
                     "ID":76798,
                     "crval":[80.29119335,16.63945791],
                     "crpix":[150,150],#[150+69,150],
                     "mjd":60653.33203125
                     }
             }

maxrawsamps = 2250

#psrdada key
DSAX_PSRDADA_KEY = 0xcaea #0xbada
DSAX_FSTOPDADA_KEY = 0xcafa
NSFRB_PSRDADA_KEY = 0xcaba
NSFRB_PSRDADA_TESTKEYS = {0:0xcab0,
                          1:0xcab1,
                          2:0xcab2,
                          3:0xcab3,
                          4:0xcab4,
                          5:0xcab5,
                          6:0xcab6,
                          7:0xcab7,
                          8:0xcab8,
                          9:0xcab9,
                          10:0xcaba,
                          11:0xcabb,
                          12:0xcabc,
                          13:0xcabd,
                          14:0xcabe,
                          15:0xcabf}
#main
NSFRB_CANDDADA_KEY = 0xcada
NSFRB_SRCHDADA_KEY = 0xcaea
NSFRB_TOADADA_KEY = 0xcafa
#slow
NSFRB_CANDDADA_SLOW_KEY = 0xcadb
NSFRB_SRCHDADA_SLOW_KEY = 0xcaeb
NSFRB_TOADADA_SLOW_KEY = 0xcafb
#imgdiff
NSFRB_CANDDADA_IMGDIFF_KEY = 0xcadc
NSFRB_SRCHDADA_IMGDIFF_KEY = 0xcaec
NSFRB_TOADADA_IMGDIFF_KEY = 0xcafc




NSFRB_PSRDADA_BYTES = 14899200
NSFRB_CANDDADA_BYTES = 144961600
NSFRB_SRCHDADA_BYTES = 28992320
NSFRB_TOADADA_BYTES = 28992320

minDM = 171
maxDM = 4000
