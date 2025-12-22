import argparse
import copy
from nsfrb import plotting as pl
from event import names
import csv
import json
from nsfrb import candcutting as cc
from nsfrb.outputlogging import numpy_to_fits
from nsfrb import noise
from nsfrb.planning import get_RA_cutoff
from nsfrb import searching as sl
import etcd3
import tracemalloc
from dsacalib import constants as ct
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait
import glob
from matplotlib import pyplot as plt
import h5py
from casatools import table
import numpy as np
from astropy.time import Time
import astropy.units as u
import sys
from dsamfs import utils as pu
from dsautils import cnf
from collections import OrderedDict
my_cnf = cnf.Conf(use_etcd=True)


from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,pixperFWHM,chanbw,freq_axis, freq_axis_fullres,lambdaref,c,NSFRB_PSRDADA_KEY,NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,rttx_file,rtbench_file,nsamps,T,bad_antennas,flagged_antennas,Lon,Lat,Height,maxrawsamps,flagged_corrs,inject_dir,local_inject_dir,rtmemory_file,vis_dir,frame_dir,cand_dir,cwd,table_dir
from nsfrb.config import tsamp as tsamp_ms
from nsfrb.config import NROWSUBIMG,NSUBIMG,SUBIMGPIX,SUBIMGORDER,baseband_tsamp
from nsfrb.imaging import inverse_revised_uniform_image,uv_to_pix, revised_robust_image,get_ra,briggs_weighting,uniform_grid,realtime_robust_image
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST
from nsfrb.TXclient import send_data,ipaddress
import time
from scipy.stats import norm,multivariate_normal
from nsfrb import pipeline
import os

from nsfrb.config import local_inject_dir,rtbench_file,rttx_file,GPofflinestagefile
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
from dsautils import cnf
from realtime import rtreader
from realtime import rtwriter
from psrdada import Reader
from nsfrb.config import NSFRB_PSRDADA_KEY,nsamps,NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,IMAGE_SIZE,DSAX_PSRDADA_KEY
"""
This script reads raw fast visibility data from a file on disk, applies fringe-stopping from a pre-made table,
applies calibration, and images. If specified, the resulting image is transmitted to the process server.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","hh16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m

#logger
logfile = "realtime_imager_log.txt"

# ETCD interface
from multiprocessing import Process, Queue
ETCD = ds.DsaStore()
ETCDKEY = f'/mon/nsfrb/fastvis'
ETCDKEY_INJECT = f'/mon/nsfrb/inject'
ETCDKEY_TIMING = f'/mon/nsfrb/timing'
ETCDKEY_TIMING_LIST = [f'/mon/nsfrbtiming/'+str(i+1) for i in range(len(corrs))]
ETCDKEY_CORRSTAGGER = f'/mon/nsfrbstagger'

#flagged antennas/
TXtask_list = []

def printlog(txt,output_file,end='\n'):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    print(txt,file=fout,end=end,flush=True)
    if output_file != "":
        fout.close()
    return


from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
from nsfrb.config import lambdaref

from multiprocessing import Queue

#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
def main(args,GPcurrentnoise,GPlastframe):
    os.system("> " + rtbench_file)
    os.system("> " + rtmemory_file)
    if len(args.rtlog)>0:
        os.system("> "+ args.rtlog)
    if len(args.rterr)>0:
        os.system("> "+ args.rterr)

    #read and reshape into np array (25 times x 4656 baselines x 8 chans x 2 pols, complex)
    gulp_counter = 0
    tasklist = []



    #set the dec, sb, and mjd
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None)
    datadir = vis_dir + args.GPdir + "/"
    sb,mjd_init,dec = pipeline.read_raw_vis(datadir+"nsfrb_sb00_{b}.out".format(b=args.fnum),get_header=True,nsamps=args.num_time_samples,nchan=args.nchans_per_node,gulp=0,headersize=16)
    fobs = (1e-3)*(np.reshape(freq_axis_fullres,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1))


    pt_dec = dec*np.pi/180.
    #if verbose: printlog("Pointing dec (deg):" + str(pt_dec*180/np.pi),output_file=logfile)
    bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)


    #flagging andd baseline cut
    tmp, bname, blen, UVW, antenna_order,keep = flag_vis(np.zeros((nsamps,UVW.shape[1],args.nchans_per_node,2)), bname, blen, UVW, antenna_order, list(flagged_antennas) + list(args.flagants), bmin, [], flag_channel_templates = [], flagged_chans=[], flagged_baseline_idxs=args.flagbase, bmax=args.bmax, returnidxs=True)
    #keep = np.sqrt(UVW[0,:,1]**2 + UVW[0,:,0]**2)>args.bmin
    U = UVW[0,:,1]
    V = UVW[0,:,0]
    W = UVW[0,:,2]
    uv_diag=np.max(np.sqrt(U**2 + V**2))
    pixel_resolution = (lambdaref / uv_diag) / args.pixperFWHM


    
    
    rtlog_file = args.rtlog
    rterr_file = args.rterr
    #if args.verbose: printlog("STARTUP PARAMS:" + str((sb,Dec,mjd_init)),output_file=rtlog_file)
    startuperr = False


    if args.verbose: printlog("Ready for data",output_file=rtlog_file)
    
    #memory logging
    if args.debug:
        tracemalloc.start()
        mallocloop = 0
        startmalloc = tracemalloc.take_snapshot()
        f = open(rtmemory_file,"w")
        print("INITIAL MEMORY ALLOCATION",file=f)
        startstats = startmalloc.statistics('lineno')
        for i in range(len(startstats)):
            print(startstats[i],file=f)
        print("-"*20,file=f)
        print("")
        f.close()
    if args.verbose: 
        printlog("Will send data to IP " + str(args.ipaddress),output_file=rtlog_file)

    inject_count=0
    gulps = np.arange(max([args.gulp_offset,0]),min([args.gulp_offset+args.num_gulps,90]),dtype=int)
   
    #initialize last_frame 
    if args.initframes:
        printlog("Initializing previous frames...",output_file=rtlog_file)
        GPlastframe = np.zeros((args.gridsize,args.gridsize,args.num_time_samples,args.num_chans))
        #sl.init_last_frame(args.gridsize,args.gridsize,args.num_time_samples-sl.maxshift,args.num_chans)
        #sl.init_last_frame(args.gridsize,args.gridsize,args.nsamps-sl.maxshift_slow,args.nchans,slow=True)

    #initialize noise stats
    if args.initnoise or args.initnoisezero:
        printlog("Initializing noise statistics...",output_file=rtlog_file)
        GPcurrentnoise = (np.zeros((len(sl.widthtrials),len(sl.DM_trials))),0)

        #noise.init_noise(sl.DM_trials,sl.widthtrials,args.gridsize,args.gridsize,zero=args.initnoisezero)
        #GP_current_noise = noise.noise_update_all(None,args.gridsize,args.gridsize,sl.DM_trials,sl.widthtrials,readonly=True)
        #np.save(noise_dir + "running_vis_mean.npy",None)
        #np.save(noise_dir + "running_vis_mean_burst.npy",None)


    slow=args.slow
    imgdiff=args.imgdiff
    
    for gulp in gulps:
        mjd = mjd_init + (gulp*T/1000/86400)
        img_id_isot = Time(mjd,format='mjd').isot

    
        #look for image files
        if args.usepastimages and len(glob.glob(datadir+"ofbimage_"+img_id_isot+".npy"))>0 and np.load(datadir+"ofbimage_"+img_id_isot+".npy").shape==(args.gridsize,args.gridsize,args.num_time_samples,16):
            printlog("Using past image "+datadir+"ofbimage_"+img_id_isot+".npy",output_file=rtlog_file)
            image_tesseract = np.load(datadir+"ofbimage_"+img_id_isot+".npy")
        else:
            printlog(datadir+"nsfrb_sb*_{b}_ofbimage_gulp{c:02d}_{d}.npy".format(b=args.fnum,c=gulp,d=img_id_isot),output_file=rtlog_file)
            fs = glob.glob(datadir+"nsfrb_sb*_{b}_ofbimage_gulp{c:02d}_{d}.npy".format(b=args.fnum,c=gulp,d=img_id_isot))
            while len(fs)<16:
                time.sleep(0.1)
                fs = glob.glob(datadir+"nsfrb_sb*_{b}_ofbimage_gulp{c:02d}_{d}.npy".format(b=args.fnum,c=gulp,d=img_id_isot))
            print("Found files:",[os.path.basename(fs[i]) for i in range(len(fs))])
        
            if args.debug: tbuffer = tbuffer1=time.time()

            image_tesseract = np.zeros((args.gridsize,args.gridsize,args.num_time_samples,16))
            for i in range(16):
                if datadir+"nsfrb_sb{a:02d}_{b}_ofbimage_gulp{c:02d}_{d}.npy".format(a=i,b=args.fnum,c=gulp,d=img_id_isot) in fs:
                    image_tesseract[:,:,:,i] = np.load(datadir+"nsfrb_sb{a:02d}_{b}_ofbimage_gulp{c:02d}_{d}.npy".format(a=i,b=args.fnum,c=gulp,d=img_id_isot))

        if args.debug: tbuffer = tbuffer1=time.time()

        RA_axis,DEC_axis,tmp = uv_to_pix(mjd_init,image_tesseract.shape[0],Lat=Lat,Lon=Lon,uv_diag=uv_diag,DEC=dec,pixperFWHM=args.pixperFWHM)
        RA_axis_2D,DEC_axis_2D,tmp = uv_to_pix(mjd_init,image_tesseract.shape[0],Lat=Lat,Lon=Lon,uv_diag=uv_diag,DEC=dec,pixperFWHM=args.pixperFWHM,two_dim=True)
        time_axis = np.linspace(0,T,image_tesseract.shape[2])
        TOAs,image_tesseract_searched,image_tesseract_binned,total_noise = sl.run_search_GPU(image_tesseract,SNRthresh=args.SNRthresh,
                                                                            RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,canddict=dict(),
                                                                            usefft=args.usefft,multithreading=args.multithreading,
                                                                            nrows=args.nrows,ncols=args.ncols,
                                                                            output_file=rtlog_file,threadDM=args.threadDM,samenoise=args.samenoise,
                                                                            cuda=args.cuda,
                                                                            space_filter=args.spacefilter,kernel_size=args.kernelsize,
                                                                            exportmaps=args.exportmaps,
                                                                            append_frame=True,DMbatches=args.DMbatches,
                                                                            SNRbatches=args.SNRbatches,usejax=args.usejax,noiseth=args.noiseth,
                                                                            RA_cutoff=get_RA_cutoff(dec,T=tsamp_ms*nsamps,pixsize=np.abs(RA_axis[1]-RA_axis[0])),
                                                                            DM_trials=sl.DM_trials,widthtrials=sl.widthtrials,
                                                                            applySNthresh=False,slow=False,imgdiff=False,attach=dict(),
                                                                            completeness=False,forfeit=False,lockdev=args.lockdev,ofb_lastframe=GPlastframe,
                                                                            ofb_currentnoise=GPcurrentnoise,ofb_lastframeslow=GPlastframe)


        if args.debug:
            printlog("--->SEARCH TIME: "+str(time.time()-tbuffer)+" sec",output_file=rtbench_file)


        #global current_noise
        GPcurrentnoise = (total_noise,GPcurrentnoise[1] +1)
        #GP_current_noise = (noise.noise_update_all(total_noise,args.gridsize,args.gridsize,sl.DM_trials,sl.widthtrials,writeonly=True),GP_current_noise[1] + 1)
        #sl.save_last_frame(image_tesseract,full=True)
        GPlastframe = copy.deepcopy(image_tesseract)

        #save image
        f = open(datadir+"ofbimage_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (not slow and imgdiff) else "") + ".npy","wb")
        np.save(f,image_tesseract)
        f.close()

        f = open(datadir+"ofbimage_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_searched.npy","wb")
        np.save(f,image_tesseract_searched)
        f.close()

        f = open(datadir+"ofbimage_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_TOAs.npy","wb")
        np.save(f,TOAs)
        f.close()

        


        if args.cleanup:
            fs = glob.glob(datadir+"nsfrb_sb*_{b}_ofbimage_gulp{c:02d}_{d}.npy".format(b=args.fnum,c=gulp,d=img_id_isot))
            for f in fs:
                print("rm "+str(f))
                os.system("rm "+str(f))


        #post-processing
        if np.nanmax(image_tesseract_searched)>args.SNRthresh:
            canddict=dict()
            fname = datadir+"ofbimage_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + ".csv"
            final_fname=datadir+"final_candidates_ofbimage_" + img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + ".csv"
            
            #peak S/N candidate -- skip clustering
            finalcandnames,finalcands=cc.sort_cands(fname,image_tesseract_searched,TOAs,args.SNRthresh,RA_axis,DEC_axis,sl.widthtrials,sl.DM_trials,
                            canddict,
                            np.abs(image_tesseract.shape[1]-image_tesseract_searched.shape[1]),
                            np.abs(image_tesseract.shape[0]-image_tesseract_searched.shape[0]),
                            rtlog_file,0,1,False,False,False,False,np.inf)
            candRAidx,candDECidx,candWIDTHidx,candDMidx,candTOA,candSNR=finalcands[0]


            #classification -- use 3D classification of full image
            candpredict, candprob = cc.classify_images_3D(image_tesseract[np.newaxis,:,:,:,:], args.model_weights3D, verbose=args.verbose)
            candpredict = candpredict[0]
            candprob = candprob[0]
            candibox = int(np.ceil(int(sl.widthtrials[int(candWIDTHidx)])*tsamp_ms/baseband_tsamp))
            candmjd = Time(mjd + (candTOA*(tsamp_ms)/1000/86400),format='mjd').mjd
            candisot = Time(candmjd,format='mjd').isot
            candWIDTH=int(sl.widthtrials[int(candWIDTHidx)])
            candDM=sl.DM_trials[int(candDMidx)]
            candRA = RA_axis_2D[int(candDECidx),int(candRAidx)]
            candDEC = DEC_axis_2D[int(candDECidx),int(candRAidx)]

            if candpredict==0:
                printlog("Good candidate found, writing csv and candplot",output_file=rtlog_file)
                #write final candidates to csv
                prefix = "NSFRB"
                with open(table_dir+"nsfrb_lastname.txt","r") as lnamefile:
                    lastname = (lnamefile.read()).strip()
                    if lastname == "None":
                        lastname = None
                lnamefile.close()
                lastname = names.increment_name(candmjd,lastname=lastname)
                candname = prefix+lastname
                printlog("done getting lastname:"+lastname,output_file=rtlog_file)

                with open(final_fname,"w") as csvfile:
                    hdr = ["candname","RA index","DEC index","WIDTH index", "DM index", "TOA", "SNR", "PROB"]
                    cdr = [lastname,candRAidx,candDECidx,candWIDTHidx,candDMidx,candTOA,candSNR,candprob]
                    wr = csv.writer(csvfile,delimiter=',')
                    wr.writerow(hdr)
                    wr.writerow(cdr)
                csvfile.close()
                    
                    

                with open(table_dir+"nsfrb_lastname.txt","w") as lnamefile:
                    if lastname is not None:
                        lnamefile.write(lastname)
                    else:
                        lnamefile.write("None")
                    lnamefile.close()
                lnamefile.close()
                printlog("done naming stuff",output_file=rtlog_file)

                #make diagnostic plot
                printlog("making diagnostic plot...",output_file=rtlog_file)
                canddict['names'] = [prefix+lastname]
                canddict['probs'] = [candprob]
                canddict['predicts'] = [candpredict]
                canddict['ra_idxs'] = [candRAidx]
                canddict['dec_idxs'] = [candDECidx]
                canddict['wid_idxs'] = [candWIDTHidx]
                canddict['dm_idxs'] = [candDMidx]
                canddict['snrs'] = [candSNR]
                canddict['TOAs'] = [candTOA]


                #dedisperse
                
                sourceimg = image_tesseract[int(candDECidx):int(candDECidx)+1,int(candRAidx):int(candRAidx)+1,:,:]
                if (candDM != 0 and not imgdiff):
                    printlog("COMPUTING SHIFTS FOR DM="+str(candDM)+"pc/cc "+ str(sourceimg.shape),output_file=rtlog_file)

                    tshift =np.array(np.abs((4.15)*candDM*((1/np.nanmin(freq_axis)/1e-3)**2 - (1/freq_axis/1e-3)**2))//tsamp_ms,dtype=int)
                    sourceimg_dm = np.zeros_like(sourceimg)
                    for j in range(len(freq_axis)):
                        sourceimg_dm[:,:,:,j] = np.pad(sourceimg[:,:,:,j],((0,0),(0,0),(tshift[j],0)),mode='constant')[:,:,:sourceimg.shape[2]]
                else:
                    sourceimg_dm = sourceimg
                timeseries = [np.nanmean(sourceimg_dm,(0,1,3))]


                candplot=pl.search_plots_new(canddict,image_tesseract,img_id_isot,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                            DM_trials=sl.DM_trials,widthtrials=sl.widthtrials,
                                            output_dir=datadir,
                                            show=False,s100=args.SNRthresh/2,
                                            injection=False,vmax=candSNR,vmin=args.SNRthresh,
                                            searched_image=image_tesseract_searched,timeseries=timeseries,uv_diag=uv_diag,
                                            dec_obs=dec,slow=slow,imgdiff=imgdiff,pcanddict=dict(),output_file=rtlog_file)
                printlog(candplot,output_file=rtlog_file)
                with open(datadir + "/" + candname+ ".json","w") as jf:

                    json.dump({"mjds":candmjd,
                            "isot":candisot,
                            "snr":candSNR,
                            "ibox":candibox,
                            "dm":candDM,
                            "ibeam":-1,
                            "cntb":-1,
                            "cntc":-1,
                            "specnum":-1,
                            "ra":candRA,
                            "dec":candDEC,
                            "trigname":candname,
                            "period":-1
                            },jf)
                jf.close()

                printlog(datadir + "/" + candname+ ".json",output_file=rtlog_file)
                printlog("writecands done",output_file=rtlog_file)

            else:
                printlog("Classifier rejected candidates",output_file=rtlog_file)

        else:
            printlog("No candidates found",output_file=rtlog_file)



    #executor.shutdown()
    return GPcurrentnoise, GPlastframe


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    #parser.add_argument('--sb',type=int,help="sb num",default=0)
    parser.add_argument('--GPdir',type=str,help='directory of Galactic Plane data',default='')
    parser.add_argument('--fnum',type=str,help='filenumber',default='')
    parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=90)
    parser.add_argument('--gulp_offset',type=int,help='Gulp offset to start from, default = 0', default=0)
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=raw_datasize)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--inject',action='store_true',default=False,help='Inject a burst into the gridded visibilities. Unless the --solo_inject flag is set, a noiseless injection will be integrated into the data.')
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=8)
    parser.add_argument('--briggs',action='store_true',help='If set use robust weighted gridding with \'briggs\' weighting')
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging',default=0)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--bmax',type=float,help='Maximum baseline length to include, default=None',default=np.inf)
    parser.add_argument('--wstack',action='store_true',help='If set use w-stacking algorithm with --Nlayers layers')
    parser.add_argument('--wstack_parallel',action='store_true',help='If set uses parallel processing for w-stacking')
    parser.add_argument('--Nlayers',type=int,help='Number of layers for w-stacking',default=18)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--flagSWAVE',action='store_true',help='Flag channels when SWAVE template RFI is detected, which manifests as a 2 Hz sin wave over ~5 minutes of data')
    parser.add_argument('--flagBPASS',action='store_true',help='Flag channels when BPASS template RFI is detected, which is simpl comparison to bandpass mean in visibilities')
    parser.add_argument('--flagFRCBAND',action='store_true',help='Flag channels in FRC miltiary allocation 1435-1525 MHz')
    parser.add_argument('--flagBPASSBURST',action='store_true',help='Flag channel when BPASS template RFI is detected in any timestep, i.e. should detect pulsed narrowband RFI')
    parser.add_argument('--flagcorrs',type=int,nargs='+',default=[],help='List of sb nodes [0,15] to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagants',type=int,nargs='+',default=[],help='List of antennas to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagchans',type=int,nargs='+',default=[],help='List of channels [0,(16*nchans_per_node - 1)] to flag')
    parser.add_argument('--flagbase',type=int,nargs='+',default=[],help='List of baselines [0,4655] to flag')
    parser.add_argument('--nbase',type=int,help='Expected number of baselines',default=4656)
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of processes used for multithreading; only used if --multiimage is set; default=16',default=16)
    parser.add_argument('--multiimage',action='store_true',help='If set, uses multithreading for imaging')
    parser.add_argument('--pixperFWHM',type=float,help='Pixels per FWHM, default 3',default=pixperFWHM)
    #parser.add_argument('--multiimagepol',action='store_true',help='If set with --multiimage flag, runs separate threads for each polarization, otherwise ignored')
    parser.add_argument('--multisend',action='store_true',help='If set, uses multithreading to send data to the process server')
    parser.add_argument('--stagger_multisend',type=float,help='Specifies the time in seconds between sending each subband, default 0 sends all at once',default=0)
    #parser.add_argument('--corrstagger_multisend',type=float,help='Specifies the time in seconds to query etcd for corr status',default=0)
    parser.add_argument('--port',type=int,help='Port number for receiving data from subclient, default = 8080',default=8080)
    parser.add_argument('--multiport',nargs='+',default=list(8810 + np.arange(16)),help='List of port numbers to listen on, default using single port specified in --port',type=int)
    parser.add_argument('-T','--testh23',action='store_true')
    parser.add_argument('--inject_interval',type=int,help='Number of gulps between injections',default=90)
    parser.add_argument('--inject_delay',type=float,help='Number of gulps to delay injection',default=0)
    parser.add_argument('--rttimeout',type=float,help='time to wait for search task to complete before cancelling, default=3 seconds',default=3)
    parser.add_argument('--primarybeam',action='store_true',help='Apply a primary beam correction')
    parser.add_argument('--failsafe',action='store_true',help='Shutdown if real-time limit is exceeded')
    #parser.add_argument('--dec',type=float,help='Pointing declination',default=71.6)
    parser.add_argument('--mjdfile',type=str,help='MJD file',default='/home/ubuntu/tmp/mjd.dat')
    parser.add_argument('--rtlog',type=str,help='Send output to logfile specified, defaults to stdout',default='')
    parser.add_argument('--rterr',type=str,help='Send errors to logfile specified, defaults to stdout',default='')
    parser.add_argument('--debug',action='store_true',help='memory debugging')
    parser.add_argument('--retries',type=int,help='retries',default=1)
    parser.add_argument('--TXmode',type=str,choices=['subimg','subint','base'],default='base',help='TX mode')
    parser.add_argument('--TXnints',type=int,help='Number of sub-integrations for TXmode subint',default=5)
    parser.add_argument('--ipaddress',type=str,help='IP address of process server to send data to',choices=[os.environ["NSFRBIP"],os.environ["NSFRBIP2"]],default=os.environ["NSFRBIP"])
    parser.add_argument('--protocol',choices=['tcp','udp'],default='tcp',help='protocol to use to send data to process server,default=tcp')
    parser.add_argument('--udpchunksize',type=int,help='Data chunksize in bytes,default=25886',default=25886)
    parser.add_argument('--udproundup',action='store_true',help='Round sub-integration size up')
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 10',default=3)
    parser.add_argument('--usefft',action='store_true', help='Implement PSF spatial matched filter as a 2D FFT')
    parser.add_argument('--multithreading',action='store_true',help='Enable multithreading in search')
    parser.add_argument('--nrows',type=int,help='Number of rows to break image into if multithreading, default = 4',default=4)
    parser.add_argument('--ncols',type=int,help='Number of columns to break image into if multithreading, default = 2',default=2)
    parser.add_argument('--threadDM',action='store_true',help='Break DM trials among multiple threads')
    parser.add_argument('--samenoise',action='store_true',help='Assume the noise in each pixel is the same')
    parser.add_argument('--cuda',action='store_true',help='Uses PyTorch to accelerate computation with GPUs. The cuda flag overrides the multithreading option')
    parser.add_argument('--spacefilter',action='store_true', help='Use PSF to spatial matched filter the input image')
    parser.add_argument('--kernelsize',type=int,help='Kernel size for PSF spatial matched filter; default=151',default=151)
    parser.add_argument('--exportmaps',action='store_true',help='Output noise maps for each DM and width trial to the noise directory')
    parser.add_argument('--DMbatches',type=int,help='Number of pixel batches to submit dedispersion to the GPUs with, default = 1',default=1)
    parser.add_argument('--SNRbatches',type=int,help='Number of pixel batches to submit boxcar filtering to the GPUs with, default = 1',default=1)
    parser.add_argument('--usejax',action='store_true',help='Use JAX Just-In-Time compilation for GPU acceleration')
    parser.add_argument('--noiseth',type=float,help='S/N threshold below which samples are included in noise calculation; default=3',default=3)
    parser.add_argument('--initframes',action='store_true',help='Initializes previous frames for dedispersion')
    parser.add_argument('--initnoise',action='store_true',help='Initializes noise statistics from fast vis data for S/N estimates')
    parser.add_argument('--initnoisezero',action='store_true',help='Initializes noise to 0')
    parser.add_argument('--slow',action='store_true',help='Activate slow search pipeline, which bins data by 5 samples and re-searches')
    parser.add_argument('--imgdiff',action='store_true',help='Activate image differencing search pipeline, which bins data by 25 samples and searches 5-minute chunk at DM=0')
    parser.add_argument('--cleanup',action='store_true',help='Remove individual corr images when done')
    parser.add_argument('--model_weights3D',type=str, help='Path to the model weights file for 3D classifying',default=cwd + "/simulations_and_classifications/enhanced3dcnn_weights_final_remote.pth")
    parser.add_argument('--usepastimages',action='store_true',help='Use past images')
    parser.add_argument('--lockdev',type=int,help='Locks all search tasks to a single GPU, 0 or 1',default=-1)
    args = parser.parse_args()

    GPcurrentnoise = (np.zeros((len(sl.widthtrials),len(sl.DM_trials))),0)
    GPlastframe = np.zeros((args.gridsize,args.gridsize,args.num_time_samples,16))

    if len(args.fnum)==0:
        allfs = np.sort(glob.glob(vis_dir + args.GPdir + "/nsfrb_sb00_*.out"))
        print("Searching all data from "+args.GPdir)
        fi=0
        for fs in allfs:
            args.fnum=os.path.basename(fs)[os.path.basename(fs).index("sb00_")+5:-4]
            if fi>0:
                args.initframes = False
                args.initnoise = False
                args.initnoisezero = False
            print(fi,"<<<"+args.fnum+">>>")
            print("-"*100)
            print("")
            

            GPcurrentnoise,GPlastframe = main(args,GPcurrentnoise,GPlastframe)
            fi+=1
    else:
        print("<<<"+args.fnum+">>>")
        main(args,GPcurrentnoise,GPlastframe)


