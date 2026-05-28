from nsfrb import jax_funcs
import jax

from nsfrb.searching import corr_shifts_all_append, tdelays_frac_append,full_boxcar_filter
import jax.numpy as jnp
#get all fast vis dates
import numpy as np
from nsfrb import pipeline,config,planning
import glob
from matplotlib import pyplot as plt
from astropy.time import Time
import os

usedev=0
gpdir = "/dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/sensitivitydata/"
allfs = np.sort(glob.glob(gpdir+"/*sb01*"))
allinfo = []
allisots = []
alldecs = []
allkeeps = []
for i in range(len(allfs)):
    #check that all files are there
    if len(glob.glob(gpdir + os.path.basename(allfs[i])[:-9] + "*"))==16:
        allkeeps.append(i)
        info=pipeline.read_raw_vis(allfs[i],get_header=True)
        #if np.abs(info[1]-61092.86449934028)<(2*60/86400):
        print("good--",info)
        allinfo.append(tuple(list(info)+[allfs[i]]))
        allisots.append(os.path.basename(allinfo[-1][-1])[:-9])
    else:
        print("bad--",allfs[i])
alltimes = Time([allinfo[i][1] for i in range(len(allinfo))],format='mjd')
allfs = allfs[np.array(allkeeps)]
print("final list: "+str(len(allfs)))

# find nvss sources > 1 Jy
from astropy import units as u
"""
checksrcs = dict()
checksrcs_nums = []
fmin=0
for i in range(len(allfs)):
    srcs = planning.nvss_cat(allinfo[i][1],allinfo[i][2],sep=0.7*u.deg)
    srcs_ = (srcs[0][srcs[1]>fmin],srcs[1][srcs[1]>fmin],srcs[2][srcs[1]>fmin])
    if len(srcs_[0])>0:
        checksrcs[allisots[i]] = srcs_
        if i not in checksrcs_nums:
            checksrcs_nums.append(i)
print(checksrcs),print(checksrcs_nums)
"""


from dsamfs import utils as pu
from dsacalib import constants as ct
from nsfrb import imaging
ii_ff = 0
nchans_per_node=8
test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order_, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None)
pt_dec = allinfo[ii_ff][2]*np.pi/180
Dec = pt_dec*180/np.pi
nchans_per_node = 8
fobs = (1e-3)*(np.reshape(config.freq_axis_fullres,(16*nchans_per_node,int(config.NUM_CHANNELS/2/nchans_per_node))).mean(axis=1))


#pt_dec = Dec*np.pi/180.
#if verbose: printlog("Pointing dec (deg):" + str(pt_dec*180/np.pi),output_file=logfile)
bname_, blen_, UVW_ = pu.baseline_uvw(antenna_order_, pt_dec, refmjd, casa_order=False)


#flagging andd baseline cut
nsamps=25
gridsize=175
bmin=20
tmp, bname, blen, UVW, antenna_order,keep = imaging.flag_vis(np.zeros((nsamps,UVW_.shape[1],nchans_per_node,2)), bname_, blen_, UVW_, antenna_order_, [49,101,71,107,116,102,105,84]+list(config.outrigger_antennas), bmin, [], flag_channel_templates = [], flagged_chans=[], flagged_baseline_idxs=[], bmax=np.inf, returnidxs=True)
#keep = np.sqrt(UVW[0,:,1]**2 + UVW[0,:,0]**2)>args.bmin
U = UVW[0,:,1]
V = UVW[0,:,0]
W = UVW[0,:,2]
uv_diag=np.max(np.sqrt(U**2 + V**2))
pixel_resolution = (config.lambdaref / uv_diag) / config.pixperFWHM

U_wavs = U[:,np.newaxis]/(ct.C_GHZ_M/fobs)
V_wavs = V[:,np.newaxis]/(ct.C_GHZ_M/fobs)
i_indices_all,j_indices_all,i_conj_indices_all,j_conj_indices_all = imaging.uniform_grid(U_wavs, V_wavs, gridsize, pixel_resolution, config.pixperFWHM)
bweights_all = np.zeros(U_wavs.shape)
for jj in range(bweights_all.shape[1]):
    bweights_all[:,jj] = imaging.briggs_weighting(U_wavs[:,jj], V_wavs[:,jj], config.gridsize, robust=-2,pixel_resolution=pixel_resolution)


from nsfrb import imaging
noisevis = np.zeros((len(allfs),len(keep),8*16,2))
pnoisevis = np.zeros((len(allfs),len(keep),8*16,2))
flagchans = np.array([6,(8*1)+1,(8*2)+0,(8*2)+5,(8*2)+6,(8*2)+7,
                     (8*3)+0,(8*3)+1,(8*3)+2,(8*3)+3,(8*3)+4,
                     (8*4)+1,(8*5)+5,(8*6)+5,(8*8)+0,(8*8)+1,
                     (8*9)+7]+list(np.arange(8)*10)+[(8*11)+0,
                                                     (8*11)+1,(8*11)+2,(8*11)+3,(8*11)+4,
                                                    (8*12)+5,(8*13)+0,(8*14)+1,(8*14)+2,
                                                    (8*14)+3,(8*14)+4,(8*14)+5,(8*14)+6,(8*15)+1])
dirty_img = np.zeros((gridsize,gridsize,25*len(allfs),16*8))
import jax
t_indices_gpu = np.repeat(np.arange(nsamps,dtype=int),U.shape[0])
_offset = 0
"""
for ii_ff in range(len(allfs)):#range(45,55):#len(allinfo)):

    for sb_i_ in range(16):
        dat_,sb_i,mjd_i,dec_i = pipeline.read_raw_vis(gpdir+allisots[ii_ff]+"_sb{:02d}.out".format(sb_i_),nsamps=25,nchan=nchans_per_node,gulp=0,get_header=False)
        dat = dat_[:,keep,:,:]
        print(dat_.shape,dat.shape)
        noisevis[ii_ff-_offset,:,sb_i*8:(sb_i+1)*8,:] = np.std(np.real(dat_)[:,keep,:,:]-np.nanmedian(np.real(dat_)[:,keep,:,:],0),0)
        pnoisevis[ii_ff-_offset,:,sb_i*8:(sb_i+1)*8,:] = np.std(np.imag(dat_)[:,keep,:,:]-np.nanmedian(np.imag(dat_)[:,keep,:,:],0),0)
        for j in range(nchans_per_node):
            jj = (nchans_per_node*sb_i)+j
            if jj in flagchans:
                dirty_img[:,:,:,jj] = np.nan
            else:
                dirty_img[:,:,25*(ii_ff-_offset):25*(1+ii_ff-_offset),jj] += imaging.realtime_robust_image(dat[:,:,j,0],#np.nanmean(dat[:,:,j,:],2),
                                                    U_wavs[:,jj],
                                                    V_wavs[:,jj],
                                                    gridsize,
                                                    True,
                                                    None ,
                                                    pixel_resolution,
                                                    config.pixperFWHM,
                                                    bweights_all[:,jj],
                                                    i_indices_all[:,jj],
                                                    j_indices_all[:,jj],
                                                    i_conj_indices_all[:,jj],
                                                    j_conj_indices_all[:,jj],
                                                    0)[0]
    np.save(config.img_dir + allisots[ii_ff]+"_senstestimage.npy",dirty_img[:,:,25*(ii_ff-_offset):25*(1+ii_ff-_offset),:])

np.save(config.img_dir +"_senstestimage.npy",dirty_img)
print("done")
"""
dirty_img = np.load(config.img_dir +"_senstestimage.npy",mmap_mode='r')


noise = np.zeros(5)
past_noise_N=0
noiseth=3
RA_cutoff = planning.get_RA_cutoff(Dec,T=(config.tsamp)*config.nsamps)
snrsamps = np.zeros((len(allfs),gridsize,gridsize-RA_cutoff,5,16))
"""
for i in range(1,len(allfs)):#1,10):
    print(i)
    image_tesseract_filtered_cut = np.nanmean(np.concatenate([dirty_img[:,:-RA_cutoff,25*(i-1):25*i,:],dirty_img[:,RA_cutoff:,25*i:25*(i+1),:]],axis=2).reshape((gridsize,gridsize-RA_cutoff,nsamps*2,16,nchans_per_node)),-1)
    image_tesseract_filtered_cut[np.isnan(image_tesseract_filtered_cut)] = 0

    image_tesseract_binned_new,noise,tmp = jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                      jax.device_put(np.array(dirty_img[:,:,0:1,0],dtype=np.float32),jax.devices()[usedev]),
                                      jax.device_put(corr_shifts_all_append,jax.devices()[usedev]),
                                      jax.device_put(tdelays_frac_append,jax.devices()[usedev]),
                                      jax.device_put(full_boxcar_filter,jax.devices()[usedev]),
                                      jax.device_put(noise,jax.devices()[usedev]),past_noise_N,noiseth)
    snrsamps[i-1,:,:,:,:] = image_tesseract_binned_new

np.save(config.img_dir + "senstest_snrsamps.npy",snrsamps)
np.save(config.img_dir + "senstest_noise.npy",noise)
"""

#snrsamps = np.load(config.img_dir + "senstest_snrsamps.npy")
noise = np.load(config.img_dir + "senstest_noise.npy")


#now add injections
from nsfrb.searching import maxshift
import copy
#dirty_img_inj = copy.deepcopy(dirty_img)
from nsfrb.planning import uv_to_pix
from inject import injecting
HA = 0
Dec = 16.27
offsetRA = offsetDEC = 0
SNR = 6
width=1
DM=0
t_now = Time(allisots[_offset],format='isot')
mjd = t_now.mjd
time_start_isot = t_now.isot
RA_axis,Dec_axis,elev = uv_to_pix(mjd,gridsize,two_dim=False,manual=False,DEC=Dec)
HA_axis = RA_axis - RA_axis[int(gridsize//2)]
injloc=0.5

noise_inj = np.load(config.img_dir + "senstest_noise.npy") #np.zeros(5)
past_noise_N=0
noiseth=3
RA_cutoff = planning.get_RA_cutoff(Dec,T=(config.tsamp)*config.nsamps)
snrsamps_inj_full = np.zeros((8,5,16,gridsize,gridsize-RA_cutoff,5,16))
from nsfrb.searching import DM_trials,gen_dm_shifts
#DM_trials = np.concatenate([np.arange(7),np.logspace(np.log2(7),np.log2(25),9,base=2).astype(int)])*config.tsamp/(4.15*(1/(config.fmin/1000)**2 - 1/(config.fmax/1000)**2))#np.array(gen_dm(253,config.maxDM,1.45,config.fc*1e-3,config.nchans,config.tsamp,config.chanbw,config.nsamps,ZERO=True))
corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append = gen_dm_shifts(DM_trials,config.freq_axis,config.tsamp,config.nsamps)
"""
toamask = np.zeros((nsamps+maxshift,nsamps+maxshift),dtype=bool)
for i in range(nsamps+maxshift):
    toamask[i,i:] = 1
NORMAXIS=np.load("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/scripts/normax20.npy")
import pickle as pkl
with open("injection_calibration_functions.pkl","rb") as pfile:
    allcalfuncs = pkl.load(pfile)
"""
import pickle as pkl
with open("dm0calcurve.pkl","rb") as p:
    cc = pkl.load(p)
for SNR in np.arange(1,12)[::-1]:
    noise_inj = np.load(config.img_dir + "senstest_noise.npy") #np.zeros(5):
    #snrsamps_inj_full = np.zeros((8,5,16,gridsize,gridsize-RA_cutoff,5,16))
    snrsamps_inj_full = np.zeros((8,5,16,5,16))

    for width_i in range(1):
        injloc = 0.5 - ((2**width_i)/nsamps/2)
        for dm_i in range(1,2):

            inject_img = injecting.generate_inject_image(Time.now().isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,
                                             snr=SNR,width=2**width_i,loc=injloc,gridsize=gridsize,nchans=16,
                                             nsamps=25,DM=DM_trials[dm_i],maxshift=maxshift,offline=False,
                                             noiseless=True,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=False,
                                             bmin=config.bmin,robust= -2)

            for i in range(int((18+8*SNR)%(len(allfs)-8))+1,int((18+8*SNR)%(len(allfs)-8))+9):#(40,48):#1,len(allfs)//6):
                print(">>>",i)
                #NORM = (allcalfuncs['m'](width_i,dm_i)*SNR) + allcalfuncs['b'](width_i,dm_i)
                tmpimg = np.nanmean(dirty_img[:,:,25*i:25*(i+1),:].reshape((gridsize,gridsize,25,16,8)),-1) + (inject_img/cc(SNR))#*SNR*(SNR*noise[0]/np.nanmax(inject_img)/NORM))
                image_tesseract_filtered_cut = np.concatenate([np.nanmean(dirty_img[:,:-RA_cutoff,maxshift*(i-1):maxshift*i,:].reshape((gridsize,gridsize-RA_cutoff,maxshift,16,8)),-1),tmpimg[:,RA_cutoff:,:,:]],axis=2)
                image_tesseract_filtered_cut[np.isnan(image_tesseract_filtered_cut)] = 0
                
                image_tesseract_binned_new,noise_inj,tmp = jax_funcs.matched_filter_dedisp_snr_fft_jit_dm0(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                      jax.device_put(np.array(dirty_img[:,:,0:1,0],dtype=bool),jax.devices()[usedev]),
                                      jax.device_put(corr_shifts_all_append,jax.devices()[usedev]),
                                      jax.device_put(tdelays_frac_append,jax.devices()[usedev]),
                                      jax.device_put(full_boxcar_filter,jax.devices()[usedev]),
                                      jax.device_put(noise_inj,jax.devices()[usedev]),past_noise_N,noiseth)
                #noise_inj = noise_inj[0]*np.sqrt(2**np.arange(5))
                #snrsamps_inj_full[i-1-(int(8*SNR%(len(allfs)-8))+1),width_i,dm_i,:,:,:,:] = image_tesseract_binned_new
                snrsamps_inj_full[i-1-(int((18+8*SNR)%(len(allfs)-8))+1),width_i,dm_i,:,:] = image_tesseract_binned_new[90,87,:,:]
                print("expected:",SNR)
                print("recovered:",image_tesseract_binned_new[90,87,width_i,dm_i])
                print("noise:",noise_inj)
                print("")
                
                #np.save(config.img_dir + "senstest_snrsamps_inj_SNR"+str(SNR)+"_W"+str(width_i)+"D"+str(dm_i)+"test.npy",image_tesseract_filtered_cut)
    #np.save(config.img_dir + "senstest_snrsamps_inj.npy",snrsamps_inj_full)
    np.save(config.img_dir + "senstest_snrsamps_inj_SNR"+str(SNR)+"_W1test.npy",snrsamps_inj_full)
    np.save(config.img_dir + "senstest_noise_inj_SNR"+str(SNR)+"_W1test.npy",noise_inj)

