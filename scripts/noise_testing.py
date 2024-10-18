from simulations_and_classifications import generate_PSF_images as scPSF
import numpy as np
from matplotlib import pyplot as plt





noise_samps = np.linspace(0,100,10)#np.logspace(0,10,10)#np.linspace(1,100,100)
psf_dir = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-PSF/"
plt.figure(figsize=(12,6))
s = []
s_psf = []
for n in noise_samps:
    #make 10 sample PSF
    PSF = scPSF.generate_PSF_images(psf_dir,0,150,True,10,dtype=np.float64,HA=0,injectnoise=n,noise_only=True)

    #get std for each pixel and channel
    PSFstd=np.nanstd(PSF,axis=(2))

    #mean over pixel
    PSFstd_mean = np.nanmean(PSFstd,axis=(0,1))
    PSFstd_err = np.nanstd(PSFstd,axis=(0,1))

    #plot istribution
    plt.subplot(311)
    plt.hist(PSF[:,:,0,:].flatten(),bins=np.linspace(0,1e-1,100),alpha=0.5,label=n)
    s.append(np.nanstd(PSF[:,:,0,:].flatten()))

    plt.subplot(312)
    plt.errorbar(np.arange(16),PSFstd_mean,yerr=PSFstd_err,marker='o',capsize=10)
    s_psf.append(np.nanmean(PSFstd_mean))
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

plt.savefig("noisedist.png")
plt.close()


