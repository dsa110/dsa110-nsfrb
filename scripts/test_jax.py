from nsfrb import jax_funcs
import jax
import numpy as np
from nsfrb.searching import corr_shifts_all_append,tdelays_frac_append,full_boxcar_filter
from scipy.stats import norm


trials = 50
for jj in range(trials):
    gridsize=175
    image_tesseract_filtered_cut = norm.rvs(loc=0,scale=1,size=((gridsize,gridsize,45,16)))
    PSF = np.zeros((gridsize,gridsize,0,16))
    prev_noise = np.zeros((5,16))
    prev_noise_N = 0
    usedev=0
    noise_data_type=np.float16
    noiseth = 3

    outtup=jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                                                #(default_PSF_gpu_0 if usedev==0 else default_PSF_gpu_1),
                                                                jax.device_put(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                #(corr_shifts_all_gpu_0 if usedev==0 else corr_shifts_all_gpu_1),
                                                                jax.device_put(corr_shifts_all_append,jax.devices()[usedev]),
                                                                #(tdelays_frac_gpu_0 if usedev==0 else tdelays_frac_gpu_1),
                                                                jax.device_put(tdelays_frac_append,jax.devices()[usedev]),
                                                                #(full_boxcar_filter_gpu_0 if usedev==0 else full_boxcar_filter_gpu_1),
                                                                jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[usedev]),
                                                                jax.device_put(np.array(prev_noise[:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                prev_noise_N,noiseth)
    image_tesseract_binned,total_noise,TOAs = np.array(outtup[0]),np.array(outtup[1])[:,np.newaxis].repeat(16,1),np.array(outtup[2])
    print(image_tesseract_binned.shape)
    print(total_noise)
    np.save("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/jax_tests_input"+str(jj)+".npy",image_tesseract_filtered_cut)
    np.save("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/jax_tests_output"+str(jj)+".npy",image_tesseract_binned)
    np.save("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/jax_tests_noise"+str(jj)+".npy",total_noise)
