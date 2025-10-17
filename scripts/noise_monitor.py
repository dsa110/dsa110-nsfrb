import numpy as np
import pickle as pkl
import csv
from matplotlib import pyplot as plt
import time
from scipy.stats import norm
from nsfrb import config
from astropy.time import Time

"""
tstart = "2025-08-30T21:19:39.651"
noise_monitor = np.load(config.noise_dir+"/"+tstart + "_long_term_noise_test.npy")
noise_count = np.load(config.noise_dir+"/"+tstart + "_long_term_noise_test_count.npy")
"""
noise_monitor = []
noise_count = []
nsamps_noise = 60 #5*6*60 #8640
ninterval = 10 #s
tstart = Time.now().isot
print("Monitoring noise every "+str(ninterval) + " seconds for "+str(nsamps_noise*ninterval/60) + " minutes")
for i in range(nsamps_noise):
    try:
        f=open(config.noise_dir+"/noise_175x175.pkl","rb")
        nn = pkl.load(f)
        f.close()
        noise_monitor.append(nn[0][1][1])
        noise_count.append(nn[0][1][0])
    except Exception as exc:
        print(exc)
        noise_monitor.append(np.nan)
        noise_count.append(np.nan)
    time.sleep(ninterval)

print("plotting...")
plt.figure(figsize=(12,6))
plt.plot(config.tsamp*config.nsamps*(np.array(noise_count) - noise_count[0])/60/1000,noise_monitor,'o',label='Noise Data')
plt.axhline(noise_monitor[-1],color='black',label='Last noise value')
plt.xlabel("Time(min)")
plt.ylabel("Noise (Arb. Units)")
plt.legend(loc='upper right')
plt.savefig(config.noise_dir + "/"+tstart + "_long_term_noise.pdf")
plt.close()
print("saving...")

np.save(config.noise_dir+"/"+tstart + "_long_term_noise_test.npy",np.array(noise_monitor))
np.save(config.noise_dir+"/"+tstart + "_long_term_noise_test_count.npy",np.array(noise_count))

print("done")
