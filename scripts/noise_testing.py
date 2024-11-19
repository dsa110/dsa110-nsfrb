from simulations_and_classifications import generate_PSF_images as scPSF
import numpy as np
import numpy as jnp
from matplotlib import pyplot as plt





noise_samps = np.linspace(0,100,10)#np.logspace(0,10,10)#np.linspace(1,100,100)
psf_dir = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-PSF/"
plt.figure(figsize=(12,6))
s = []
s_psf = []
s_psf_binned = []
PSFFILTER = np.load(psf_dir+"gridsize_300/PSF_300_0.00_deg.npy")[:,:,np.newaxis,:]
gridsize = 300
gridsize_RA = gridsize_DEC = 600
padby_DEC = (gridsize_DEC - gridsize)//2
padby_RA = (gridsize_RA - gridsize)//2
padby_DEC_img = (gridsize_DEC - gridsize)//2
padby_RA_img = (gridsize_RA - gridsize)//2
for n in noise_samps:
    #make 10 sample PSF
    PSF = scPSF.generate_PSF_images(psf_dir,0,150,True,10,dtype=np.float64,HA=0,injectnoise=n,noise_only=True)

    #get std for each pixel and channel
    PSFstd=np.nanstd(PSF,axis=(2))

    #mean over pixel
    PSFstd_mean = np.nanmean(PSFstd,axis=(0,1))
    PSFstd_err = np.nanstd(PSFstd,axis=(0,1))


    #after PSF filtering
    PSF_binned = jnp.real(
                                jnp.fft.fftshift(jnp.fft.ifft2(
                                    jnp.fft.fft2(jnp.pad(PSF,#image_tesseract,
                                            ((padby_DEC_img,padby_DEC_img),(padby_RA_img,padby_RA_img),(0,0),(0,0))),
                                        axes=(0,1),s=(gridsize_DEC,gridsize_RA))*jnp.fft.fft2(jnp.pad(
                                            (PSFFILTER.repeat(PSF.shape[2],axis=2)),
                                            ((padby_DEC,padby_DEC),(padby_RA,padby_RA),(0,0),(0,0))),
                                            axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                    ,axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                ,axes=(0,1)))[gridsize_DEC//4:(gridsize_DEC//4) + gridsize_DEC//2,gridsize_RA//4:(gridsize_RA//4) + gridsize_RA//2,:,:]
    PSFstd_mean_binned = np.nanmean(np.nanstd(PSF_binned,axis=2),axis=(0,1))
    PSFstd_err_binned = np.nanstd(np.nanstd(PSF_binned,axis=2),axis=(0,1))


    #plot istribution
    plt.subplot(311)
    plt.hist(PSF[:,:,0,:].flatten(),bins=np.linspace(0,1e-1,100),alpha=0.5,label=n)
    s.append(np.nanstd(PSF[:,:,0,:].flatten()))

    plt.subplot(312)
    plt.errorbar(np.arange(16),PSFstd_mean,yerr=PSFstd_err,marker='o',capsize=10)
    s_psf.append(np.nanmean(PSFstd_mean))

    plt.errorbar(np.arange(16),PSFstd_mean_binned,yerr=PSFstd_err_binned,marker='o',capsize=10)
    s_psf_binned.append(np.nanmean(PSFstd_mean_binned))

plt.subplot(311)
plt.yscale("log")
plt.legend()

plt.subplot(313)
plt.plot(noise_samps,s,'o--')
popt = np.polyfit(noise_samps,s,deg=1)
print("Best-fit slope:",popt[0])
plt.plot(np.linspace(0,100),popt[0]*np.linspace(0,100) + popt[1])

plt.plot(noise_samps,s_psf,'s--')
popt = np.polyfit(noise_samps,s_psf,deg=1)
print("Best-fit slope -- off-pulse:",popt[0])
plt.plot(np.linspace(0,100),popt[0]*np.linspace(0,100) + popt[1])

plt.plot(noise_samps,s_psf_binned,'v--')
popt = np.polyfit(noise_samps,s_psf_binned,deg=1)
print("Best-fit slope -- off-pulse,binned:",popt[0])
plt.plot(np.linspace(0,100),popt[0]*np.linspace(0,100) + popt[1])


plt.savefig("noisedist.png")
plt.close()


