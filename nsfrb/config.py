# config.py

# Constants
NUM_CHANNELS = 768
AVERAGING_FACTOR = 48
IMAGE_SIZE = 300  # pixels

# Speed of light
c = 299792458  # m/s

# Channel information
CH0 = 1311.387  # MHz
CH_WIDTH = 0.244141  # MHz

# Time parameters
tsamp = 0.1342182159423828*1000 #130 #ms
baseband_tsamp = 256e-3 #ms
nsamps = 25
T = tsamp*nsamps #3250 #ms
#nsamps = int(T/tsamp)

# Image channel information
nchans = int(NUM_CHANNELS/AVERAGING_FACTOR)
chanbw = CH_WIDTH*AVERAGING_FACTOR
fmax  = CH0 + CH_WIDTH * (nchans-1) * AVERAGING_FACTOR #1530 #MHz
fmin = CH0 #1280  #MHz
fc = (fmin+fmax)/2#1400 #MHz
lambdamin = (c/(fmax*1e6)) #m
lambdamax = (c/(fmin*1e6)) #m
lambdac = (c/(fc*1e6)) #m
#nchans = 16 #16 coarse channels
#chanbw = (fmax-fmin)/nchans #MHz
telescope_diameter = 4.65 #m


#resolution parameters
pixsize = 0.002962513099862611#(48/3600)*np.pi/180 #rad
gridsize = 300#256
RA_point = 0 #rad
DEC_point = 0 #rad
UVMAX = 2316.5744224010487 #maximum UV extent for uniform gridding

vis_to_img_slope_not_binned = 6.320421766399212e-05 #slope relating noise in visibilities to std noise in image estimated from simulation
vis_to_img_slope = 0.0025062597643136777 #same, but with PSF smoothing; noise increases with contribution from PSF, but so does signal

#outrigger flagging and short baseline flagging
bmin=20 #meters
flagged_antennas = []#[103,104,105,106,107,108,109,110,111,112,113,114,115,116]

import numpy as np
noise_data_type = np.float64

#file system
import os
import sys

#directories
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/"
frame_dir = cwd + "-frames/"
psf_dir = cwd + "-PSF/"
img_dir = cwd + "-images/"
vis_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/"
raw_cand_dir = cand_dir + "raw_cands/"
backup_cand_dir = cand_dir + "backup_raw_cands/"#cwd + "-candidates/backup_raw_cands/"
final_cand_dir = cand_dir + "final_cands/"#cwd + "-candidates/final_cands/"
inject_dir = inject_file = cwd + "-injections/"
training_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-training/"
noise_dir = cwd + "-noise/"
imgpath = cwd + "-images"
plan_dir = cwd + "-plans/"
table_dir = cwd + "-tables/"

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

import casatools as cc
me = cc.measures()
obs=me.observatory("OVRO_MMA")
Lat=obs['m1']['value']*180/np.pi#37.23
Lon=obs['m0']['value']*180/np.pi#-118.2851

