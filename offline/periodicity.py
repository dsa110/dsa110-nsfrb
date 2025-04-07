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
from nsfrb.config import nsamps


"""
Simple periodicity search algorithms
"""



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
