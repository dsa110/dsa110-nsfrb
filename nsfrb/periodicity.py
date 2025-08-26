import numpy as np
import argparse
import csv
from matplotlib import pyplot as plt
import os
from scipy.fftpack import ifftshift, ifft2,fftshift,fft2,fftfreq
import sys
import struct
import os, glob
from scipy.fftpack import ifftshift, ifft2, fftshift, fft2

from nsfrb.config import tsamp as tsamp_ms
from nsfrb.config import nsamps,ngulps_per_file,bin_imgdiff


"""
Simple periodicity search algorithms
"""

def gen_p_trials(nsamps,trial_p_samp):
    trial_p_samp_idxs = np.zeros((nsamps,len(trial_p_samp)),dtype=int)
    trial_p_folds_factor = np.zeros((nsamps,len(trial_p_samp)))
    for i in range(len(trial_p_samp)):
        nfolds = int(nsamps//trial_p_samp[i])
        trial_p_samp_idxs[:,i] = np.array(list(np.arange(trial_p_samp[i],dtype=int))*nfolds + list(np.arange(nsamps - (nfolds*trial_p_samp[i]),dtype=int)),dtype=int)
        trial_p_folds_factor[:trial_p_samp[i],i] +=  nfolds
        trial_p_folds_factor[:int(nsamps%trial_p_samp[i]),i] +=1
        trial_p_folds_factor[:,i] = (trial_p_folds_factor[:,i])**(3/2)
    trial_p_folds_factor[trial_p_folds_factor==0] = np.nan
    return trial_p_samp_idxs,trial_p_folds_factor

def ffa_slow(timeseries,trial_p_samp):
    """
    DM=0 FFA brute force implementation
    """
    
    snrs = np.zeros(list(timeseries.shape[:-1]) + [len(trial_p_samp)])
    n_init = np.nanmedian(np.nanstd(timeseries,-1))
    print("noise init:",n_init)
    for i in range(len(trial_p_samp)):
        maxidx = timeseries.shape[-1] - (timeseries.shape[-1]%trial_p_samp[i])
        snrs[...,i] = np.nanmax(timeseries[...,:maxidx].reshape(list(timeseries.shape[:-1]) + [maxidx//trial_p_samp[i],trial_p_samp[i]]).mean(-2),axis=-1)/(n_init/np.sqrt(maxidx//trial_p_samp[i]))
        
    return snrs


def ffa_timing(timeseries,trial_p_samp,ref_p=None):
    """
    RMS search of periodogram over fine set of period trials
    """
    #make periodogram for reference period
    if ref_p is None:
        ref_p = int(np.nanmax(trial_p_samp))
    maxidx = timeseries.shape[-1] - (timeseries.shape[-1]%ref_p)
    timeseries = timeseries[...,:maxidx]
    taxis = np.arange(timeseries.shape[-1])
    #phasebins = (taxis%ref_p)/ref_p
    #rows = ref_p*(taxis//ref_p)
    
    #get expected indices for trial periods
    rmss = np.zeros(list(timeseries.shape[:-1])+[len(trial_p_samp),int(ref_p)])
    for p in range(len(trial_p_samp)):
        #get expected indices for trial periods
        idxs = np.arange(int(taxis.shape[-1])//trial_p_samp[p])*trial_p_samp[p]
        #print(np.sort(np.argsort(timeseries,axis=-1)[::-1][:len(idxs)]))
        #print(idxs)
        #loop through trial phases
        for q in range(int(ref_p)):
            idxsq = np.clip(q + idxs,0,taxis.shape[0])
            
            #get timing residuals using the peak indices
            idxs_dat = np.sort(np.argsort(timeseries,axis=-1)[::-1][:len(idxsq)])
            rmss[...,p,q] = np.sqrt(np.nanmean((idxs_dat-idxsq)**2))

    return rmss



def gen_psamp_trials(trial_p_samp,nsamp=int(ngulps_per_file//bin_imgdiff)):
    idxs_full = np.zeros((1,1,nsamp,nsamp,len(trial_p_samp)))#np.zeros_like(timeseries,dtype=int)[...,np.newaxis,np.newaxis].repeat(timeseries.shape[-1],-2).repeat(len(trial_p_samp),-1)
    for i in range(len(trial_p_samp)):
        idxs = (np.array([trial_p_samp[i]*np.arange(0,nsamp//trial_p_samp[i],dtype=int)]*trial_p_samp[i],dtype=int) + np.arange(trial_p_samp[i])[:,np.newaxis])
        idxs_full[...,:idxs.shape[0],:idxs.shape[1],i] = idxs
    return idxs_full, idxs_full!=0



def ffa_faster(timeseries,idxs_full,bool_idxs_full,periodogram=False):
    """
    FFA using pre-computed indices
    """
    if periodogram:
        
        pgrams = []
        for p in range(idxs_full.shape[-1]):
            pgrams.append(np.take_along_axis(timeseries[...,np.newaxis],idxs_full[...,p],axis=-2))
    #snrs = np.nanmax(pgrams.sum(-2,where=bool_idxs_full),-2)
    snrs = np.nanmax((np.take_along_axis(timeseries[...,np.newaxis,np.newaxis],idxs_full,axis=-3)).mean(-2,where=bool_idxs_full),-2)
    return snrs,(None if not periodogram else pgrams)

    

#simple fast folding
def fast_fold_search(dynspec,trial_P,trial_phase,t_samp=40.96e-6,plot=False,suffix='',path=''):
    #average over freq
    timeseries = dynspec.mean(1)
    times = np.arange(len(timeseries))*t_samp

    #fold for each trial

    snrs = np.zeros((len(trial_P),len(trial_phase)))
    if plot:
        plt.figure(figsize=(12,12))
        plt.subplot(2,1,1)
        plt.plot(times,timeseries)


    for i in range(len(trial_P)):
        for j in range(len(trial_phase)):
            fold_samps = np.argmin(np.abs(times-trial_P[i]))
            #print(fold_samps)
            idxs = np.arange(0,len(timeseries),fold_samps,dtype=int) + int(trial_phase[j]*trial_P[i]/(2*np.pi)/t_samp)
            idxs = idxs[idxs<len(timeseries)]
            #print(idxs)
            snrs[i,j] = np.sum(timeseries[idxs])/len(idxs)
            if plot:
                plt.plot(times[idxs],timeseries[idxs],'o',alpha=0.5,markersize=10)
                plt.axvline(trial_P[i],color='red')
    if plot:
        plt.xlabel("Time (s)")
        plt.ylabel("Timeseries")
        #plt.show()

        plt.subplot(2,1,2)
        plt.imshow(snrs,aspect='auto',extent=(trial_P[0],trial_P[-1],trial_phase[0],trial_phase[-1]))
        plt.xlabel("Trial Period (s)")
        plt.ylabel("Trial Phase (rad)")

        plt.savefig(path + "fastfold_plot_" + str(suffix) + ".pdf")
        plt.close()
    return snrs


#simple spectrum folding
def spec_fold_search(dynspec,trial_P,t_samp=40.96e-6,plot=False,maxharms=3,suffix='',path=''):
    #average over freq
    timeseries = dynspec.mean(1)

    #FFT
    fft_sig = np.fft.fft(timeseries)
    fft_ax = np.fft.fftfreq(n=len(timeseries),d=t_samp)

    #add pos and negative frequencies
    fft_pos = fft_sig[fft_ax>=0][1:]
    fft_neg = fft_sig[fft_ax<0][1:][::-1]
    fft_tot = fft_pos + fft_neg
    fft_ax_tot = fft_ax[fft_ax>=0][1:]

    #power spectrum
    pspec = np.abs(fft_tot)**2

    #fold for each trial
    snrs = np.zeros(len(trial_P))
    if plot:
        plt.figure(figsize=(12,12))
        plt.subplot(2,1,1)
        plt.plot(fft_ax_tot,pspec)
        plt.xscale("log")

    #print(1/fft_ax_tot)
    for i in range(len(trial_P)):
        fold_samps = np.argmin(np.abs(fft_ax_tot-(1/trial_P[i])))
        #print(fold_samps)
        idxs = np.arange(fold_samps,len(pspec),fold_samps,dtype=int)
        if len(idxs) > maxharms: idxs = idxs[:maxharms]
        #print(fft_ax_tot[idxs])
        snrs[i] = np.sum(pspec[idxs])/len(idxs)/len(timeseries)
        if plot:
            plt.plot(fft_ax_tot[idxs],pspec[idxs],'o',alpha=0.5,markersize=10)
            plt.axvline(1/trial_P[i],color='red')
    if plot:
        plt.xlabel("Pulse Frequency (Hz)")
        plt.ylabel("Power Spectrum")
        plt.yscale("log")

        plt.subplot(2,1,2)
        plt.plot(trial_P,snrs)
        plt.xlabel("Trial Period (s)")
        plt.ylabel("S/N")
        #plt.show()
        plt.savefig(path + "specfold_plot_" + str(suffix) + ".pdf")
        plt.close()
    return snrs




def main(args):
    #read file
    dynspec = np.load(args.filename)

    #normalize by off-pulse noise
    noise = np.nanstd(dynspec.mean(1))
    dynspec = dynspec/noise

    trial_P = np.linspace(args.minP,args.maxP,args.numP)
    path = os.path.dirname(args.filename) + "/"
    suffix = str(args.suffix if len(args.outputname)>0 else os.path.basename(args.filename)[:-4])

    if args.FFA:
        print("Running Fast Folding Algorithm with " + str(len(trial_P)) + " period trials from " + str(args.minP) + "-" + str(args.maxP) + " seconds")
        trial_phase = np.linspace(0,2*np.pi,args.numPhase)
        snrs = fast_fold_search(dynspec,trial_P,trial_phase,t_samp=args.tsamp,plot=True,suffix=suffix,path=path)
        candidxs = np.arange(len(snrs.flatten()),dtype=int)[snrs.flatten()>args.sigma]
        candPs = [trial_P[np.unravel_index(i,snrs.shape)[0]] for i in candidxs]
        candPhases = [trial_phase[np.unravel_index(i,snrs.shape)[1]] for i in candidxs]
        candsnrs = snrs.flatten()[candidxs]
    else:
        print("Running Spectrum Folding Algorithm with " + str(len(trial_P)) + " period trials from " + str(args.minP) + "-" + str(args.maxP) + " seconds")
        snrs = spec_fold_search(dynspec,trial_P,t_samp=args.tsamp,plot=True,maxharms=args.numharms,suffix=suffix,path=path)
        candidxs =np.arange(len(snrs),dtype=int)[snrs>args.sigma]
        candPs = trial_P[snrs>args.sigma]
        candsnrs = snrs[snrs>args.sigma]
    
    print("Found " + str(len(candidxs)) + " candidates, writing to " + path + suffix + str("_fastfold" if args.FFA else "_specfold") + ".csv")
    
    #write to csv
    with open(path + suffix + str("_fastfold" if args.FFA else "_specfold") + ".csv","w") as csvfile:
        wr =  csv.writer(csvfile,delimiter=',')
        if args.FFA:
            wr.writerow(['CAND','P(s)','phase(rad)','S/N'])
            for i in range(len(candsnrs)):
                wr.writerow([i,candPs[i],candPhases[i],candsnrs[i]])
        else:
            wr.writerow(['CAND','P(s)','S/N'])
            for i in range(len(candsnrs)):
                wr.writerow([i,candPs[i],candsnrs[i]])

    return
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Images and averages fast visibilities to create periodicity search mode data')
    parser.add_argument('filename')           # positional argument
    parser.add_argument('--minP',type=float,help='Minimum search period (s),default='+str(tsamp_ms*4/1000),default=tsamp_ms*4/1000)
    parser.add_argument('--maxP',type=float,help='Maximum search period (s),default='+str(tsamp_ms*nsamps*45/1000),default=tsamp_ms*nsamps*45/1000)
    parser.add_argument('--numP',type=int,help='Number of trial periods, default=10',default=10)
    parser.add_argument('--numPhase',type=int,help='Number of trial phase for FFA, default=5',default=5)
    parser.add_argument('--FFA',action='store_true',help='Use fast folding algorithm')
    parser.add_argument('--numharms',type=int,help='Max number of harmonics to sum',default=3)
    parser.add_argument('--tsamp',type=float,help='Sampling time (s), default='+str(tsamp_ms/1000),default=tsamp_ms/1000)
    parser.add_argument('--outputname',type=str,help='label for output files',default='')
    parser.add_argument('--sigma',type=float,help='S/N threshold, default=2.0',default=2.0)
    args = parser.parse_args()
    main(args)
