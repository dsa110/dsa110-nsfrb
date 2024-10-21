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
import numpy as np
noise_data_type = np.float64
