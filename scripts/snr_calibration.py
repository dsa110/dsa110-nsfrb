import numpy as np
import numpy as jnp
from matplotlib import pyplot as plt
from inject import injecting
from nsfrb.searching import DM_trials,maxshift
from nsfrb.config import IMAGE_SIZE,nchans
from astropy.time import Time
from nsfrb.imaging import inverse_uniform_image,uniform_image, uv_to_pix
"""
This script generates injections of different S/N and DM (width=1) and estimates the matchedd filter S/N to calibrate
the input vs output S/N relation
"""
psf_dir = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-PSF/"
t1 = Time.now()
time_start_isot = t1.isot
mjd = t1.mjd
RA_axis,Dec_axis = uv_to_pix(mjd,IMAGE_SIZE,Lat=37.23,Lon=-118.2851)
HA_axis = RA_axis - RA_axis[len(RA_axis)//2]
HA = 0
DEC = 0
Dec_axis = Dec_axis - Dec_axis[len(Dec_axis)//2]
PSFFILTER = np.load(psf_dir+"gridsize_300/PSF_300_0.00_deg.npy")[:,:,np.newaxis,:]


trial_SN = np.logspace(1,5,5)
num_chans = 16
nsamps = 25
gridsize = 300
gridsize_RA = gridsize_DEC = 600
padby_DEC = (gridsize_DEC - gridsize)//2
padby_RA = (gridsize_RA - gridsize)//2
padby_DEC_img = (gridsize_DEC - gridsize)//2
padby_RA_img = (gridsize_RA - gridsize)//2
output_SN = []
DMtrials = [0]#DM_trials


for DM in DMtrials:
    for SNR in trial_SN:
        print("INPUT SNR:",SNR)
        print("INPUT DM:",DM)
        #generate injection
        offsetRA = offsetDEC = 0
        width = 1
        t1 = Time.now()
        time_start_isot = t1.isot
        mjd = t1.mjd

        """
        inject_img = injecting.generate_inject_image(time_start_isot,HA=HA,DEC=DEC,offsetRA=offsetRA,offsetDEC=offsetDEC,snr=SNR,width=width,loc=0.5,
                                                gridsize=IMAGE_SIZE,nchans=num_chans,nsamps=nsamps,DM=DM,maxshift=maxshift,offline=False,
                                                noiseless=False,HA_axis=HA_axis,DEC_axis=Dec_axis)

        #PSF filtering
        inject_img_filtered = jnp.real(
                                jnp.fft.fftshift(jnp.fft.ifft2(
                                    jnp.fft.fft2(jnp.pad(inject_img,#image_tesseract,
                                            ((padby_DEC_img,padby_DEC_img),(padby_RA_img,padby_RA_img),(0,0),(0,0))),
                                        axes=(0,1),s=(gridsize_DEC,gridsize_RA))*jnp.fft.fft2(jnp.pad(
                                            (PSFFILTER.repeat(inject_img.shape[2],axis=2)),
                                            ((padby_DEC,padby_DEC),(padby_RA,padby_RA),(0,0),(0,0))),
                                            axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                    ,axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                ,axes=(0,1)))[gridsize_DEC//4:(gridsize_DEC//4) + gridsize_DEC//2,gridsize_RA//4:(gridsize_RA//4) + gridsize_RA//2,:,:]
        """
        inject_img_filtered = np.load("SNRtest_img" + str(SNR) + ".npy")


        #get snr
        peaksamp = np.argmax(np.nansum(inject_img_filtered[150,150,:,:],axis=1))
        noise = np.nanmedian(np.nanstd(inject_img_filtered[:,:,:peaksamp,:].sum(3),axis=2))
        print("NOISE:",noise)
        out_SNR = (np.nansum(inject_img_filtered[150,150,peaksamp,:])-np.nanmedian(inject_img_filtered[150,150,:peaksamp,:].sum(1)))/noise
        print("SNR:",out_SNR)
        output_SN.append(out_SNR)
        
        #np.save("SNRtest_img" + str(SNR) + ".npy",(inject_img_filtered-np.nanmedian(inject_img_filtered[:,:,:peaksamp,:].sum(3),axis=2))/noise)
        
        
        plt.imsave("SNRtest_img" + str(SNR) + ".png",(inject_img_filtered[:,:,peaksamp-1,:].sum(2)-np.nanmedian(inject_img_filtered[:,:,:peaksamp,:].sum(3),axis=2))/noise,vmin=7,vmax=out_SNR)
"""
plt.figure(figsize=(12,6))
for i in range(len(DMtrials)):
    plt.plot(trial_SN,output_SN[i*len(trial_SN):(i+1)*len(trial_SN)])
plt.xlabel("Input S/N")
plt.ylabel("Output S/N")
plt.xscale("log")
plt.yscale("log")
plt.savefig("SNR_test.png")
plt.close()
"""
