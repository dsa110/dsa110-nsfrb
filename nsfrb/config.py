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
tsamp = 130 #ms
T = 3250 #ms
nsamps = int(T/tsamp)

# Image channel information
fmax  = 1530 #MHz
fmin = 1280  #MHz
fc = 1400 #MHz
lambdamin = (c/(fmax*1e6)) #m
lambdamax = (c/(fmin*1e6)) #m
lambdac = (c/(fc*1e6)) #m
nchans = 16 #16 coarse channels
chanbw = (fmax-fmin)/nchans #MHz
telescope_diameter = 4.65 #m


#resolution parameters
pixsize = 0.002962513099862611#(48/3600)*np.pi/180 #rad
gridsize = 300#256
RA_point = 0 #rad
DEC_point = 0 #rad
UVMAX = 2316.5744224010487 #maximum UV extent for uniform gridding

"""
#for jax use only: pre-defined shift values for dedisp
tdelays_frac = None
corr_shifts_all_hi = None
corr_shifts_all_low = None
image_tesseract_point_DM = None
"""
