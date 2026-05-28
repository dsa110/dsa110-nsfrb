import glob
from matplotlib import pyplot as plt
from nsfrb import pipeline
from nsfrb.searching import DM_trials,maxshift

import numpy as np
from inject import injecting
from nsfrb import config

dirty_img = np.load(config.img_dir +"_senstestimage.npy",mmap_mode='r')
print(dirty_img.shape)


from astropy.time import Time
import os
from nsfrb.imaging import uv_to_pix
from nsfrb.searching import maxshift
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


HA = 0
Dec = 16.27
offsetRA = offsetDEC = 0
SNR = 25
width=1
DM=0
_offset=i=20
gridsize=175
#noise = np.load(config.img_dir + "senstest_noise.npy")
t_now = Time(allisots[_offset],format='isot')
mjd = t_now.mjd
time_start_isot = t_now.isot
RA_axis,Dec_axis,elev = uv_to_pix(mjd,gridsize,two_dim=False,manual=False,DEC=Dec)
HA_axis = RA_axis - RA_axis[int(gridsize//2)]
wi=0
di=0
injloc=0.5

xinj=injecting.generate_inject_image(Time.now().isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,
                                                 snr=SNR,width=int(2**wi),loc=injloc,gridsize=gridsize,nchans=16,
                                                 nsamps=25,DM=DM_trials[di],maxshift=maxshift,offline=False,
                                                 noiseless=True,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=False,
                                                 bmin=config.bmin,robust= -2)

import pickle as pkl
with open("dm0calcurve.pkl","rb") as p:
    cc = pkl.load(p)
allnunnorm = []

for i in range(18,48):#dirty_img.shape[0]//25):
    print(tuple(list(dirty_img.shape)[:-1]+[16,8]))
    x = np.nansum(dirty_img[:,:,25*i:25*(i+1),:].reshape(tuple(list(dirty_img[:,:,25*i:25*(i+1),:].shape)[:-1]+[16,8])),-1)
    #print(x.shape)
    n = (np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2))
    print(np.nanmedian(n))
    allnunnorm.append(np.nanmedian(n))
    SN = np.nanmax(np.nansum(xinj[90,87,:,:],1))/cc(SNR)/np.nanmedian(n)#np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2))
    SNstd = SN*(np.nanstd(n)/np.nanmedian(n))#np.nanstd(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2))/np.nanmedian(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2))
    SNerr = SNstd/np.sqrt(np.product(n.shape))
    #print(np.nanmax(np.nansum(xinj[90,87,:,:],1))/cc(SNR)/np.nanmean(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2)))
    #print(np.nanmax(np.nansum(xinj[90,87,:,:],1))/cc(SNR)/np.nanmedian(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2)))
    #print(np.nanmax(np.nansum(xinj[90,87,:,:],1))/cc(SNR)/np.nanstd(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2)))
    print(SN,SNstd,SNerr)
#allnunnorm = np.array(allnunnorm)
#print("avg unnormalized noise:",np.nanmedian(allnunnorm))
#np.save("unnormalized_noise_samps.npy",allnunnorm)
"""
plt.figure()
plt.hist(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2).flatten(),100)
plt.axvline(np.nanmean(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2)))
plt.axvline(np.nanmedian(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2)))
plt.axvspan(np.nanmedian(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2))-np.nanstd(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2)),np.nanmedian(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2))+np.nanstd(np.nanstd(np.nansum(x-np.nanmedian(x,axis=2,keepdims=True),3),2)),alpha=0.5)
plt.yscale("log")
plt.savefig("test.png")
plt.close()
"""
