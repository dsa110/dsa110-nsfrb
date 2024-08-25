import numpy as np
import os
import jax
import socket
import time
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import random
import copy
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import truncnorm
from scipy.signal import peak_widths
from scipy.stats import norm
from event import names
#from gen_dmtrials_copy import gen_dm
import argparse
from astropy.time import Time
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import glob
import csv
import copy

from nsfrb.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset
from nsfrb.noise import init_noise,noise_update_all,get_noise_dict
import hdbscan
import copy
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from nsfrb.config import tsamp,fmin,fmax,nchans
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack
from nsfrb.imaging import uv_to_pix
from nsfrb import plotting as pl

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
import os
import sys
cwd = os.environ['NSFRBDIR']
cand_dir = cwd + "-candidates/"
cutterfile = cwd + "-logfiles/candcutter_log.txt"
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = cwd + "/src/.pipestatus.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = cwd + "/scripts/script_flags/searchlog_flags.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_flags/searchlog_flags.txt"
output_file = cwd + "-logfiles/run_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
processfile = cwd + "-logfiles/process_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt"
cutterfile = cwd + "-logfiles/candcutter_log.txt"
flagfile = cwd + "/process_server/process_flags.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_flags.txt"
cand_dir = cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
raw_cand_dir = cwd + "-candidates/raw_cands/"
backup_cand_dir = cwd + "-candidates/backup_raw_cands/"
final_cand_dir = cwd + "-candidates/final_cands/"
error_file = cwd + "-logfiles/error_log.txt"
inject_file = cwd + "-injections/injections.csv"
recover_file = cwd + "-injections/recoveries.csv"
sys.path.append(cwd + "/") #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")

#hdbscan clustering function; clusters in DM, width, RA, DEC space
def hdbscan_cluster(cands,min_cluster_size=50,dmt=[0]*16,wt=[0]*5,SNRthresh=1,plot=False,show=False,output_file=cutterfile):
    f = open(output_file,"a")
    print(str(len(cands)) + " candidates",file=f)

    #make list for each param
    raidxs = []
    decidxs = []
    dmidxs = []
    widthidxs = []
    snridxs = []
    for i in range(len(cands)):
        raidxs.append(cands[i][0])
        decidxs.append(cands[i][1])
        dmidxs.append(cands[i][3])
        widthidxs.append(cands[i][2])
        snridxs.append(cands[i][4])
    raidxs = np.array(raidxs)
    decidxs = np.array(decidxs)
    dmidxs = np.array(dmidxs)
    widthidxs = np.array(widthidxs)
    snridxs = np.array(snridxs)

    test_data=np.array([raidxs,decidxs,dmidxs,widthidxs]).transpose()


    #create clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)

    #cluster data
    clusterer.fit(test_data)


    #print number of noise points
    noisepoints = np.sum(clusterer.labels_==-1)
    print(str(noisepoints) + " noise points",file=f)

    nclasses = len(np.unique(clusterer.labels_))
    classnames = np.unique(clusterer.labels_)
    classes = clusterer.labels_
    if -1 in clusterer.labels_:
        nclasses -= 1

    print(str(nclasses) + " unique classes",file=f)

    #get centroids
    #fcsv = open(cand_dir + "hdbscan_cluster_cands.csv","w")
    #csvwriter = csv.writer(fcsv)
    centroid_ras = []
    centroid_decs = []
    centroid_dms = []
    centroid_widths = []
    centroid_snrs = []
    for k in classnames:
        if k != -1:
            centroid_ras.append((np.nansum((snridxs*raidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_decs.append((np.nansum((snridxs*decidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_dms.append((np.nansum((snridxs*dmidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_widths.append((np.nansum((snridxs*widthidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_snrs.append(np.nansum((snridxs*snridxs)[classes==k])/np.nansum(snridxs[classes==k]))
            #csvwriter.writerow([centroid_ras[-1],centroid_decs[-1],centroid_widths[-1],centroid_dms[-1],centroid_snrs[-1]])
    #fcsv.close()
    centroid_ras = np.array(centroid_ras)
    centroid_decs = np.array(centroid_decs)
    centroid_dms = np.array(centroid_dms)
    centroid_widths = np.array(centroid_widths)
    centroid_snrs = np.array(centroid_snrs)

    centroid_cands = [(centroid_ras[i],centroid_decs[i],centroid_widths[i],centroid_dms[i],centroid_snrs[i]) for i in range(len(centroid_ras))]

    if plot:
        cands_noninf = []
        for i in cands:
            if not np.isinf(i[-1]): cands_noninf.append(i)

        plt.figure(figsize=(40,12))
        plt.subplot(121)
        for i in range(-1,len(np.unique(classes))-int(-1 in classes)):
            if i == -1:
                plt.scatter(np.array(cands_noninf)[classes==i,0],np.array(cands_noninf)[classes==i,1],alpha=0.5,s=1000*(np.array(cands_noninf)[classes==i,-1] - SNRthresh)/(2*SNRthresh - SNRthresh),label='Not Classified',color='grey')
            else:
                c=plt.plot(centroid_ras[i],centroid_decs[i],'x',markersize=50,markerfacecolor="none",markeredgewidth=4)
                plt.scatter(np.array(cands_noninf)[classes==i,0],np.array(cands_noninf)[classes==i,1],alpha=0.5,s=1000*(np.array(cands_noninf)[classes==i,-1] - SNRthresh)/(2*SNRthresh - SNRthresh),label='Class ' + str(i),c=c[0].get_color())

        plt.xlim(0,32)
        plt.ylim(0,32)
        plt.xlabel("RA index")
        plt.ylabel("DEC index")
        plt.legend(loc='upper right')

        plt.subplot(122)
        wtinterp = interp1d(np.arange(len(wt)),wt,fill_value='extrapolate')
        dmtinterp = interp1d(np.arange(len(dmt)),dmt,fill_value='extrapolate')
        for i in range(-1,len(np.unique(classes))-int(-1 in classes)):
            if i == -1:
                plt.scatter(wt[np.array(cands_noninf,dtype=int)[classes==i,2]],dmt[np.array(cands_noninf,dtype=int)[classes==i,3]],alpha=0.5,s=1000*(np.array(cands_noninf)[classes==i,-1] - SNRthresh)/(2*SNRthresh - SNRthresh),label='Not Classified',color='grey')
            else:
                c=plt.plot(int(wtinterp(centroid_widths[i])),int(dmtinterp(centroid_dms[i])),'x',markersize=50,markerfacecolor="none",markeredgewidth=4)
                plt.scatter(wt[np.array(cands_noninf,dtype=int)[classes==i,2]],dmt[np.array(cands_noninf,dtype=int)[classes==i,3]],alpha=0.5,s=1000*(np.array(cands_noninf)[classes==i,-1] - SNRthresh)/(2*SNRthresh - SNRthresh),label='Class ' + str(i),c=c[0].get_color())
        plt.xlabel("Width (Samples)")
        plt.ylabel("DM (pc/cc)")
        plt.legend(loc='upper right',frameon=True)


        plt.savefig(cand_dir + "hdbscan_cluster_plot.png")
        if show:
            plt.show()
        else:
            plt.close()
    f.close()
    return classes,centroid_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs



#code to cutout subimages
freq_axis = np.linspace(fmin,fmax,nchans)
def get_subimage(image_tesseract,ra_idx,dec_idx,subimgpix=11,save=False,prefix="candidate_stamp",plot=False,output_file=cutterfile,output_dir=cand_dir,corr_shift=None,tdelay_frac=None,dm=None,tsamp=tsamp,freq_axis=freq_axis):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    gridsize = image_tesseract.shape[0]
    fname = output_dir + prefix + "_" + str(ra_idx) + "_" + str(dec_idx)
    if subimgpix%2 == 0:
        print("subimgpix must be odd",file=fout)
        if output_file != "":
            fout.close()
        return None


    #dedisperse if given a dm
    if (corr_shift is not None) and (tdelay_frac is not None): 
        image_tesseract_dm = quick_dedisp(image_tesseract,corr_shift,tdelay_frac)
    elif dm is not None:
        fname = fname + "_dedisp" + str(dm) + ".npy"
        image_tesseract_dm = copy.deepcopy(image_tesseract)
        for i in range(gridsize):
            for j in range(gridsize):
                tdelays = dm*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
                tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
                tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
                tdelays_frac = tdelays/tsamp - tdelays_idx_low

                for k in range(nchans):
                    #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac)
                    arrlow =  np.pad(image_tesseract[i,j,:,k],((0,tdelays_idx_low[k])),mode="constant",constant_values=0)[tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                    arrhi =  np.pad(image_tesseract[i,j,:,k],((0,tdelays_idx_hi[k])),mode="constant",constant_values=0)[tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

                    image_tesseract_dm[i,j,:,k] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])

    else:
        fname = fname + ".npy"
        image_tesseract_dm = copy.deepcopy(image_tesseract)


    #pad w/ nans
    image_tesseract_dm = np.pad(image_tesseract_dm,((gridsize,gridsize),
                                                   (gridsize,gridsize),
                                                   (0,0),
                                                   (0,0)),
                                                   mode='constant',
                                                   constant_values=np.nan)

    #cut out subimage
    minraidx = int(gridsize + ra_idx - subimgpix//2)#np.max([ra_idx - subimgpix//2,0])
    maxraidx = int(gridsize + ra_idx + subimgpix//2 + 1)#np.min([ra_idx + subimgpix//2 + 1,gridsize-1])
    mindecidx = int(gridsize + dec_idx - subimgpix//2)#np.max([dec_idx - subimgpix//2,0])
    maxdecidx = int(gridsize + dec_idx + subimgpix//2 + 1)#np.min([dec_idx + subimgpix//2 + 1,gridsize-1])

    #print(minraidx_cut,maxraidx_cut,mindecidx_cut,maxdecidx_cut)
    print(minraidx,maxraidx,mindecidx,maxdecidx,file=fout)

    image_cutout = image_tesseract_dm[minraidx:maxraidx,mindecidx:maxdecidx,:,:]

    if save:
        np.save(fname,image_cutout)

    if plot:
        plt.figure(figsize=(12,12))
        plt.imshow(image_cutout.mean((2,3)),aspect='auto')
        plt.show()
    if output_file != "":
        fout.close()
    return image_cutout


#this is a quick implementation of dedispersion meant only for cutouts, classification, and injection
def quick_dedisp(image_pixel,corr_shift,tdelay_frac):
    return (((np.take_along_axis(image_pixel.repeat(2,axis=3),indices=corr_shift,axis=2))*tdelay_frac).reshape((image_pixel.shape[0],image_pixel.shape[1],image_pixel.shape[2],image_pixel.shape[3],2))).mean(4)

#this is a copy of the jax binning function which runs on a single pixel on the CPU. it does not normalize by off-pulse noise and is
#only meant for classification purposes
def quick_snr_fft(image_pixel,width):
    boxcar = np.zeros((1,1,image_pixel.shape[0],1))
    boxcar[:,:,len(boxcar)//2 -width//2 - 2:len(boxcar)//2 -width//2 +width- 2,:] = 1
    return np.nan_to_num(np.real(np.fft.ifftshift(
                                            np.fft.ifft(
                                                np.fft.fft(image_pixel,
                                                            n=image_pixel.shape[2],
                                                            axis=2,norm='backward')*np.fft.fft(boxcar,
                                                            n=image_pixel.shape[2],axis=2,norm='backward'),
                                                        n=image_pixel.shape[2],
                                                        axis=2,norm='backward'),axes=2)),
                                            nan=0,posinf=0,neginf=0)



#checks injection file to see if a candidate is an injection
def is_injection(isot,inject_file=inject_file):
    #check if the candidate is an injection
    injection = False
    with open(inject_file,"r") as csvfile:
        re = csv.reader(csvfile,delimiter=',')
        i = 0
        for row in re:
            if i != 0:
                if row[0] == isot:
                    injection = True
                    break
            i += 1
    csvfile.close()
    return injection

#main cand cutter task function
def candcutter_task(fname,args):
    """
    Main task to obtain cutouts
    """
    #for each candidate get the isot and find the corresponding image
    cand_isot = fname[fname.index("candidates_")+11:fname.index(".csv")]
    cand_mjd = Time(cand_isot,format='isot').mjd
    #read cand file
    finalcands = []
    raw_cand_names = []
    with open(fname,"r") as csvfile:
        re = csv.reader(csvfile,delimiter=',')
        for r in re:
            if 'candname' not in r:
                finalcands.append(np.array(r[1:],dtype=float))
                raw_cand_names.append(r[0])
    csvfile.close()
    finalidxs = np.arange(len(finalcands),dtype=int)

    #if getting cutouts, read image
    try:
        image = np.load(raw_cand_dir + cand_isot + ".npy")
    except Exception as e:
        printlog("No image found for candidate " + cand_isot,output_file=cutterfile)
        return

    #get DM trials from file
    DMtrials = np.load(cand_dir + "DMtrials.npy")
    widthtrials = np.load(cand_dir + "widthtrials.npy")
    SNRthresh = float(np.load(cand_dir +"SNRthresh.npy"))
    corr_shifts = np.load(cand_dir+"DMcorr_shifts.npy")
    tdelays_frac = np.load(cand_dir+"DMdelays_frac.npy")

    #start clustering
    if args['cluster']:
        printlog("clustering with HDBSCAN...",output_file=cutterfile)

        #prune candidates with infinite signal-to-noise for clustering
        cands_noninf = []
        for fcand in finalcands:
            if not np.isinf(fcand[-1]): cands_noninf.append(fcand)


        #clustering with hdbscan
        classes,cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs = hdbscan_cluster(cands_noninf,min_cluster_size=args['mincluster'],dmt=DMtrials,wt=widthtrials,plot=True,show=False,SNRthresh=SNRthresh)
        printlog("done, made " + str(len(cluster_cands)) + " clusters",output_file=cutterfile)
        printlog(classes,output_file=cutterfile)
        printlog(cluster_cands,output_file=cutterfile)

        finalidxs = np.arange(len(cluster_cands),dtype=int)
        finalcands = cluster_cands


    if args['classify']:
        #make a binned copy for each candidate
        data_array = np.zeros((len(finalcands),args['subimgpix'],args['subimgpix'],image.shape[3]),dtype=image.dtype)
        for j in range(len(finalcands)):
            printlog(finalcands[j],output_file=cutterfile)
            subimg = quick_snr_fft(get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args['subimgpix'],corr_shift=corr_shifts[:,:,:,int(finalcands[j][3]),:],tdelay_frac=tdelays_frac[:,:,:,int(finalcands[j][3]),:]),widthtrials[int(finalcands[j][2])])
            data_array[j,:,:,:] = subimg[:,:,np.argmin(subimg.sum((0,1,3))),:]

        #reformat for classifier
        transposed_array = np.transpose(data_array, (0,3,1,2))#cands x frequencies x RA x DEC
        new_shape = (data_array.shape[0], data_array.shape[3], data_array.shape[1], data_array.shape[2])
        merged_array = transposed_array.reshape(new_shape)

        #run classifier
        predictions, probabilities = classify_images(merged_array, args['model_weights'], verbose=args['verbose'])
        printlog(predictions,output_file=cutterfile)
        printlog(probabilities,output_file=cutterfile)

        #only save bursts likely to be real
        #finalidxs = finalidxs[~np.array(predictions,dtype=bool)]

    #if its an injection write the highest SNR candidate to the injection tracker
    injection_flag = is_injection(cand_isot)
    if injection_flag:
        with open(recover_file,"a") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            for j in finalidxs:
                wr.writerow([cand_isot,DMtrials[int(finalcands[j][3])],widthtrials[int(finalcands[j][2])],finalcands[j][4]])
        csvfile.close()


    #write final candidates to csv
    prefix = "NSFRB"
    lastname = None     #once we have etcd, change to 'names.get_lastname()'
    allcandnames = []
    csvfile = open(final_cand_dir + "final_candidates_" + cand_isot + ".csv","w")
    wr = csv.writer(csvfile,delimiter=',')
    if args['classify']:
        wr.writerow(["candname","RA index","DEC index","WIDTH index", "DM index", "SNR","PROB"])
    else:
        wr.writerow(["candname","RA index","DEC index","WIDTH index", "DM index", "SNR"])
    sysstdout = sys.stdout
    for j in finalidxs:#range(len(finalidxs)):
        with open(cutterfile,"a") as sys.stdout:
            lastname = names.increment_name(cand_mjd,lastname=lastname)
        sys.stdout = sysstdout
        if args['classify']:
            wr.writerow(np.concatenate([[lastname],np.array(finalcands[j],dtype=int),[probabilities[j]]]))
        else:
            wr.writerow(np.concatenate([[lastname],np.array(finalcands[j],dtype=int)]))
        allcandnames.append(lastname)
    csvfile.close()

    #get image cutouts and write to file
    if args['cutout']:
        for j in finalidxs:#range(len(finalidxs)):
            subimg = get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args['subimgpix'])#[:,:,int(finalcands[j][2]),:]
            lastname = allcandnames[j]
            np.save(final_cand_dir + prefix + lastname + ".npy",subimg)
    #send candidates to slack
    if len(finalidxs) > 0:
        #make diagnostic plot
        printlog("making diagnostic plot...",output_file=cutterfile,end='')
        canddict = dict()
        canddict['ra_idxs'] = [finalcands[j][0] for j in finalidxs]
        canddict['dec_idxs'] = [finalcands[j][1] for j in finalidxs]
        canddict['wid_idxs'] = [finalcands[j][2] for j in finalidxs]
        canddict['dm_idxs'] = [finalcands[j][3] for j in finalidxs]
        canddict['snrs'] = [finalcands[j][-1] for j in finalidxs]
        RA_axis,DEC_axis = uv_to_pix(cand_mjd,image.shape[0],Lat=37.23,Lon=-118.2851)
        candplot=pl.search_plots_new(canddict,image,cand_isot,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                            DM_trials=DMtrials,widthtrials=widthtrials,
                                            output_dir=final_cand_dir,show=False,s100=SNRthresh/2,injection=injection_flag,vmax=2*SNRthresh)
        printlog("done!",output_file=cutterfile)

        if args['toslack']:
            printlog("sending plot to slack...",output_file=cutterfile,end='')
            send_candidate_slack(candplot,filedir=final_cand_dir)
            printlog("done!",output_file=cutterfile)


    #once finished, move raw data to backup directory (at some point, make this an scp to dsastorage)
    os.system("mv " + raw_cand_dir + "*" + cand_isot + "* " + backup_cand_dir)
    printlog("Done! Total Remaining Candidates: " + str(len(finalidxs)),output_file=cutterfile)
    return


