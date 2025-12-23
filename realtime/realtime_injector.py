import argparse
import json
import random
from inject import injecting
from nsfrb.outputlogging import printlog
import glob
import csv
from matplotlib import pyplot as plt
from nsfrb.simulating import compute_uvw,get_core_coordinates,get_all_coordinates
#from inject import injecting
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

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()

#sys.path.append(cwd+"/nsfrb/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/nsfrb/")
#sys.path.append(cwd+"/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,tsamp
#from nsfrb.imaging import inverse_uniform_image,uniform_image,inverse_revised_uniform_image,revised_uniform_image, uv_to_pix, revised_robust_image,get_ra
from nsfrb.imaging import uv_to_pix
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST
from nsfrb.TXclient import send_data
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images
from tqdm import tqdm
import time
from scipy.stats import norm,multivariate_normal
#import nsfrb.searching as sl
from nsfrb.outputlogging import numpy_to_fits
#from nsfrb import calibration as cal
from nsfrb import pipeline
import os
#vispath = os.environ["NSFRBDATA"] + "dsa110-nsfrb-fast-visibilities" #cwd + "-fast-visibilities"
#imgpath = cwd + "-images"
#inject_file = cwd + "-injections/injections.csv"

from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,flagged_antennas,Lon,Lat,maxrawsamps,flagged_corrs,inject_log_file,table_dir

import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
from dsautils import cnf
import rtreader
"""
This script continuously pulls data from memory mapped from the rtwriter, images, and sends to the process
server in realtime.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","hh16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m


#flagged antennas

#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
"""f = open("/home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat","r")
flagged_antennas = np.array(f.read().split("\n")[:-1],dtype=int)
f.close()
"""
CONF = cnf.Conf(use_etcd=True)
CORR_CONF = CONF.get('corr')
CAL_CONF = CONF.get('cal')
MFS_CONF = CONF.get('fringe')
CORRLIST = list(CORR_CONF['ch0'].keys())
CORRLIST = [CORRLIST[i][6:] for i in range(len(CORRLIST))]
NCORR = len(CORRLIST)
CALTIME = CAL_CONF['caltime_minutes']*u.min
FILELENGTH = MFS_CONF['filelength_minutes']*u.min


# ETCD interface
from multiprocessing import Process, Queue
ETCD = ds.DsaStore()
ETCDKEY = f'/mon/nsfrb/inject'

from nsfrb.searching import DM_trials as default_DMtrials
from nsfrb.searching import widthtrials as default_widthtrials
from nsfrb.searching import maxshift
"""
This service will run on h24 and create injections thqt can be rsynced to each corr node for use in the realtime system.
"""

def update(args,current_inject_time,INJECT_INIT,INJECT_QUEUE,INJECT_TIMES,INJECT_PARAMS):
    
    printlog("INJECT QUEUE:"+str(INJECT_QUEUE),output_file=inject_log_file)
    #check if first injection has been sent
    if not INJECT_INIT:
        INJECT_INIT = True
    
        #push injection parameters to etcd
        try:
            ETCD.put_dict(ETCDKEY,INJECT_QUEUE.pop(0))
            current_inject_time = INJECT_TIMES.pop(0)
        except:
            printlog("no injections available",output_file=inject_log_file)
    elif len(INJECT_PARAMS)>0:
        #otherwise, first check if previous injection has been read or timed out
        current_dict = ETCD.get_dict(ETCDKEY)
        if np.all(current_dict['ack']): 
            current_params = INJECT_PARAMS.pop(0)
            #write to csv
            with open(inject_file,"a") as csvfile:
                wr = csv.writer(csvfile,delimiter=',')
                wr.writerow([current_dict['ISOT'],current_params['DM'],current_params['width'],current_params['SNR']])
            csvfile.close()
            os.system("rm " + inject_dir +  "realtime_staging/" + "injection_" + str(current_dict['ID']) + "_sb*.npy")
            if not np.all(current_dict['injected']):
                printlog("Injection" + current_dict['ISOT'] + " missing channels:" + str(np.arange(args.num_chans)[np.logical_not(np.array(current_dict['injected']))]),output_file=inject_log_file)
            try:
                ETCD.put_dict(ETCDKEY,INJECT_QUEUE.pop(0))
                current_inject_time = INJECT_TIMES.pop(0)
            except:
                printlog("no injections available",output_file=inject_log_file)
        """
        elif (time.time()-current_inject_time >= args.waittime*60):
            printlog("Injection timed out",inject_log_file)
            current_params = INJECT_PARAMS.pop(0)
            #delete injection
            printlog("Removing injection " + str(current_dict['ID']),output_file=inject_log_file)
            os.system("rm " + inject_dir +  "realtime_staging/" + "injection_" + str(current_dict['ID']) + "_sb*.npy")
            try:
                ETCD.put_dict(ETCDKEY,INJECT_QUEUE.pop(0))
                current_inject_time = INJECT_TIMES.pop(0)
            except:
                printlog("no injections available",output_file=inject_log_file)
        """
    return current_inject_time,INJECT_INIT,INJECT_QUEUE,INJECT_TIMES,INJECT_PARAMS

def main(args):

    verbose = args.verbose
    INJECT_QUEUE = []
    INJECT_TIMES = []
    INJECT_PARAMS = []
    INJECT_INIT = False
    current_inject_time = -1
    int_time = time.time()
    int_onoff = True
    while True:
        if args.intermittent and time.time()-int_time >= args.waittime*60:
            int_onoff = not int_onoff
            if int_onoff:
                printlog("Intermittent mode, pausing injection production",output_file=inject_log_file)
            else:
                printlog("Intermittent mode, resuming injection production",output_file=inject_log_file)
            int_time = time.time()
        if args.intermittent and int_onoff:
            current_inject_time,INJECT_INIT,INJECT_QUEUE,INJECT_TIMES,INJECT_PARAMS= update(args,current_inject_time,INJECT_INIT,INJECT_QUEUE,INJECT_TIMES,INJECT_PARAMS)
            continue
        #RA, dec axes
        t_now = Time.now()
        mjd = t_now.mjd
        time_start_isot = t_now.isot
        cleardataflag = args.solo_inject or args.flat_field or args.gauss_field
        injectflatflag = args.point_field or args.gauss_field or args.flat_field
        """
        RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,two_dim=False,manual=False)
        HA_axis = RA_axis - RA_axis[int(args.gridsize//2)]
        cleardataflag = args.solo_inject or args.flat_field or args.gauss_field
        injectflatflag = args.point_field or args.gauss_field or args.flat_field
        HA = 0
        RA = RA_axis[int(args.gridsize//2)]
        Dec = Dec_axis[int(args.gridsize//2)]
        
        #creating injection
        offsetRA,offsetDEC,SNR,width,DM,maxshift = injecting.draw_burst_params(Time.now().isot,RA_axis=RA_axis,DEC_axis=Dec_axis,gridsize=args.gridsize,nsamps=args.num_time_samples,nchans=args.num_chans,tsamp=tsamp,SNRmin=args.snr_min_inject,SNRmax=args.snr_max_inject)
        #offsetRA = offsetDEC = 0

        if args.snr_inject > 0:
            SNR = args.snr_inject
        if args.dm_inject != -1 and args.dm_inject >= 0:
            DM = args.dm_inject
        if args.width_inject > 0:
            width = args.width_inject
        offsetRA = args.offsetRA_inject
        offsetDEC = args.offsetDEC_inject
        #print("PARAMSFROM OFFLINE IMAGER:",offsetRA,offsetDEC,SNR,width,DM,maxshift,tsamp)
        #print("OFFSET HOUR ANGLE:",HA_axis[int(len(HA_axis)//2 + offsetRA)])
        noiseless=False
        if args.inject_noiseless:
            noiseless=True
        inject_img = injecting.generate_inject_image(Time.now().isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,snr=SNR,width=width,loc=0.5,gridsize=args.gridsize,nchans=args.num_chans,nsamps=args.num_time_samples,DM=DM,maxshift=maxshift,offline=args.offline,noiseless=noiseless,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=args.inject_noiseonly,bmin=args.bmin,robust=args.robust if args.briggs else -2)
        if args.flat_field:
            inject_img = np.ones_like(inject_img)
        elif args.gauss_field:
            xx,yy = np.meshgrid(np.linspace(-2,2,args.gridsize),np.linspace(-2,2,args.gridsize))
            inject_img = multivariate_normal(mean=[0,0],cov=0.5).pdf(np.dstack((xx,yy)))
            inject_img = inject_img[:,:,np.newaxis,np.newaxis].repeat(args.num_time_samples,2).repeat(args.num_chans,3)
        elif args.point_field:
            inject_img = np.zeros_like(inject_img)
            inject_img[int(args.gridsize//2)+offsetDEC,int(args.gridsize//2)+offsetRA] = 1
        #report injection in log file
        """
        """
        with open(inject_file,"a") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            wr.writerow([time_start_isot,DM,width,SNR])
        csvfile.close()
        """
        DM = np.random.choice(default_DMtrials)
        width = np.random.choice(default_widthtrials[:-1])
        Dec=args.dec
        SNR=args.snr_inject

        #generate random 10 digit identifier
        printlog("finished injection",output_file=inject_log_file)
        ID = str(random.randint(10**10,10**(11) - 1))
        #for j in range(args.num_chans):
        #    np.save(inject_dir + "realtime_staging/" + "injection_" + str(ID) + "_sb" +str("0" if j<10 else "")+ str(j) + ".npy",inject_img[:,:,:,j])
        
        if args.continuous or args.intermittent:
            #put injection in queue
            INJECT_PARAMS.append({"DM":DM,
                              "width":width,
                              "SNR":SNR})
            INJECT_TIMES.append(time.time())
            INJECT_QUEUE.append({"ID":ID,
                               "fname":"injection_DM"+str(DM)+"_W"+str(width)+"_DEC"+str(args.dec)+"_SB",
                               "dec":Dec,
                               "injected":[False]*args.num_chans,
                               "ack":[False]*args.num_chans,
                               "inject_only":cleardataflag,
                               "inject_flat":injectflatflag})


            #update
            current_inject_time,INJECT_INIT,INJECT_QUEUE,INJECT_TIMES,INJECT_PARAMS = update(args,current_inject_time,INJECT_INIT,INJECT_QUEUE,INJECT_TIMES,INJECT_PARAMS)

        else:
            #push injection parameters to etcd
            ETCD.put_dict(ETCDKEY,{"ID":ID,
                               "dec":Dec,
                               "fname":"injection_DM"+str(DM)+"_W"+str(width)+"_DEC"+str(args.dec)+"_SB",
                               "injected":[False]*args.num_chans,
                               "ack":[False]*args.num_chans,
                               "inject_only":cleardataflag,
                               "inject_flat":injectflatflag})
    
            #sleep...sort of
            t1 = time.time()
            acked = False
            while time.time() - t1 < args.waittime*60:
            
                #check to see its been injected on all corr nodes
                time.sleep(tsamp/1000)#args.waittime*60/90)
                injection_dict = ETCD.get_dict(ETCDKEY)
                if 'ISOT' in injection_dict.keys() and np.any(injection_dict['ack']) and not acked:
                    #write to csv
                    with open(inject_file,"a") as csvfile:
                        wr = csv.writer(csvfile,delimiter=',')
                        wr.writerow([injection_dict['ISOT'],DM,width,SNR])
                    csvfile.close()
                    if not np.all(injection_dict['injected']):
                        printlog("Injection" + injection_dict['ISOT'] + " missing channels:" + str(np.arange(args.num_chans)[np.logical_not(np.array(injection_dict['injected']))]),output_file=inject_log_file)
                    #delete injection
                    #printlog("Removing injection " + str(ID),output_file=inject_log_file)
                    #os.system("rm " + inject_dir +  "realtime_staging/" + "injection_" + str(ID) + "_sb*.npy")
                    #break
                    acked=True

                    # read timestamp files and add to exclude table 
                    printlog("updating exclude tables...",output_file=inject_log_file)
                    s_table_list = glob.glob(table_dir + "/rt_speccal_timestamps_" + injection_dict['ISOT'][:10] +"*.json")
                    if len(s_table_list)>0:
                        for s_table_name in s_table_list:
                            printlog(s_table_name,output_file=inject_log_file)
                            f = open(s_table_name,"r")
                            s_table = json.load(f)
                            f.close()
                            for k in s_table.keys():
                                print("python "+cwd+"/scripts/sensitivity/add_to_extable.py --name "+'"'+str(k)+'"'+" --mjd "+str(Time(injection_dict['ISOT'],format='isot').mjd) + " --reason INJECTION")
                                os.system("python "+cwd+"/scripts/sensitivity/add_to_extable.py --name "+'"'+str(k)+'"'+" --mjd "+str(Time(injection_dict['ISOT'],format='isot').mjd) + " --reason INJECTION")
                                print("python "+cwd+"/scripts/sensitivity/add_to_extable.py --name "+'"'+str(k)+'"'+" --mjd "+str(Time(injection_dict['ISOT'],format='isot').mjd + (tsamp*args.num_time_samples/1000/86400)) + " --reason INJECTION")
                                os.system("python "+cwd+"/scripts/sensitivity/add_to_extable.py --name "+'"'+str(k)+'"'+" --mjd "+str(Time(injection_dict['ISOT'],format='isot').mjd + (tsamp*args.num_time_samples/1000/86400)) + " --reason INJECTION")
                    a_table_list = glob.glob(table_dir + "/rt_astrocal_timestamps_" + injection_dict['ISOT'][:10] +"*.json")
                    if len(a_table_list)>0:
                        for a_table_name in a_table_list:
                            printlog(a_table_name,output_file=inject_log_file)
                            f = open(a_table_name,"r")
                            a_table = json.load(f)
                            f.close()
                            for k in a_table.keys():
                                print("python "+cwd+"/scripts/sensitivity/add_to_extable.py --name "+'"'+str(k)+'"'+" --mjd "+str(Time(injection_dict['ISOT'],format='isot').mjd) + " --reason INJECTION")
                                os.system("python "+cwd+"/scripts/sensitivity/add_to_extable.py --name "+'"'+str(k)+'"'+" --mjd "+str(Time(injection_dict['ISOT'],format='isot').mjd) + " --reason INJECTION")
                                print("python "+cwd+"/scripts/sensitivity/add_to_extable.py --name "+'"'+str(k)+'"'+" --mjd "+str(Time(injection_dict['ISOT'],format='isot').mjd + (tsamp*args.num_time_samples/1000/86400)) + " --reason INJECTION")
                                os.system("python "+cwd+"/scripts/sensitivity/add_to_extable.py --name "+'"'+str(k)+'"'+" --mjd "+str(Time(injection_dict['ISOT'],format='isot').mjd + (tsamp*args.num_time_samples/1000/86400)) + " --reason INJECTION")
                    printlog("Done",output_file=inject_log_file)
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate injections for the realtime pipeline')
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--solo_inject',action='store_true',default=False,help='If set, visibility data will be zeroed and an injection with simulated noise will overwrite the data')
    parser.add_argument('--snr_inject',type=float,help='SNR of injection; default -1 which chooses a random SNR',default=1E7)
    #parser.add_argument('--dm_inject',type=float,help='DM of injection; default -1 which chooses a random DM',default=-1)
    #parser.add_argument('--width_inject',type=int,help='Width of injection in samples; default -1 which chooses a random width',default=-1)
    #parser.add_argument('--offsetRA_inject',type=int,help='Offset RA of injection in samples; default random', default=int(np.random.choice(np.arange(-IMAGE_SIZE//2,IMAGE_SIZE//2))))
    #parser.add_argument('--offsetDEC_inject',type=int,help='Offset DEC of injection in samples; default random', default=int(np.random.choice(np.arange(-IMAGE_SIZE//2,IMAGE_SIZE//2))))
    parser.add_argument('--offline',action='store_true',default=False,help='Initializes previous frame with noise')
    parser.add_argument('--inject_noiseonly',action='store_true',default=False,help='Only inject noise; for use with false positive testing')
    parser.add_argument('--inject_noiseless',action='store_true',default=False,help='Only inject signal')
    parser.add_argument('--num_inject',type=int,help='Number of injections, must be less than number of gulps',default=1)
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--flat_field',action='store_true',help='Illuminate all pixels uniformly')
    parser.add_argument('--gauss_field',action='store_true',help='Illuminate a gaussian source')
    parser.add_argument('--point_field',action='store_true',help='Illuminate a point source')
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--waittime',type=float,help='Time between injections, default 5 minutes',default=5)
    parser.add_argument('--snr_min_inject',type=float,help='Minimum injection S/N, default 1e7',default=1e7)
    parser.add_argument('--snr_max_inject',type=float,help='Maximum injection S/N, default 1e8',default=1e8)
    parser.add_argument('--briggs',action='store_true',help='If set use robust weighted gridding with \'briggs\' weighting')
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging',default=0)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--continuous',action='store_true',help='Continuously make injections')
    parser.add_argument('--intermittent',action='store_true',help='Continuously make injections for --waittime minutes, then stop for --waittime minutes')
    parser.add_argument('--dec',type=float,help='Declination',default=71.6)
    args = parser.parse_args()
    main(args)
