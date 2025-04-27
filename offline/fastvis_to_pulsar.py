import argparse
from nsfrb.imaging import DSAelev_to_ASTROPYalt
from dsautils.coordinates import create_WCS,get_declination,get_elevation
from astropy.coordinates import SkyCoord,EarthLocation
from astropy import units as u
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait
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
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE,fmin,fmax,c,pixsize,bmin,raw_datasize,pixperFWHM,chanbw
from nsfrb.config import tsamp as tsamp_ms
from nsfrb.imaging import inverse_uniform_image,uniform_image,inverse_revised_uniform_image,revised_uniform_image, uv_to_pix, revised_robust_image,get_ra,briggs_weighting,uniform_grid,deredden
from nsfrb.flagging import flag_vis,fct_SWAVE,fct_BPASS,fct_FRCBAND,fct_BPASSBURST
from nsfrb.TXclient import send_data,ipaddress
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

from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,flagged_antennas,Lon,Lat,maxrawsamps,flagged_corrs,psr_dir,Lat,Lon,Height

dsaloc = EarthLocation(lon=Lon*u.deg,lat=Lat*u.deg,height=Height*u.m)


"""
This script reads raw fast visibility data from a file on disk, applies fringe-stopping from a pre-made table,
applies calibration, and images. If specified, the resulting image is transmitted to the process server.
"""

#corr node names and frequencies
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
sbs = ["sb00","sb01","sb02","sb03","sb04","sb05","sb06","sb07","sb08","sb09","sb10","sb11","sb12","sb13","sb14","sb15"]
freqs = np.linspace(fmin,fmax,len(corrs))
wavs = c/(freqs*1e6) #m

#flagged antennas/
def offline_image_task(dat, U_wavs, V_wavs, i_indices_all, j_indices_all, i_conj_indices_all, j_conj_indices_all, bweights_all, gridsize,  pixel_resolution, nchans_per_node, fobs_j, j, briggs=False, robust= 0.0, return_complex=False, inject_img=None, inject_flat=False, wstack=False, W_wavs=None, k_indices_all=None, k_conj_indices_all=None, Nlayers_w=18,pixperFWHM=pixperFWHM):

    outimage = np.nan*np.ones((args.gridsize,args.gridsize,args.num_time_samples))
    for jj in range(nchans_per_node):
        if briggs:
            #print("INPUT SHAPE",dat[:,:,jj,:].mean(2))#dat[:,:,jj,:].transpose((0,2,1)).shape)
            outimage = revised_robust_image(dat[:,:,jj,:].mean(2),#.transpose((0,2,1)),#dat[i:i+1, :, jj, k],
                                            U_wavs[:,jj],
                                            V_wavs[:,jj],
                                            gridsize,
                                            inject_img=None if inject_img is None or np.all(inject_img==0) else inject_img/dat.shape[-1]/nchans_per_node,
                                            robust=robust,
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,
                                            wstack=wstack,
                                            w=None if W_wavs is None else W_wavs[:,jj],
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM,
                                            i_indices=i_indices_all[:,jj],
                                            j_indices=j_indices_all[:,jj],
                                            k_indices=None if not wstack else k_indices_all[:,jj],
                                            i_conj_indices=i_conj_indices_all[:,jj],
                                            j_conj_indices=j_conj_indices_all[:,jj],
                                            k_conj_indices=None if not wstack else k_conj_indices_all[:,jj],clipuv=False,keeptime=True)
        else:
            for i in range(dat.shape[0]):
                #for k in range(dat.shape[-1]):
                if jj==0:#k == 0 and jj == 0:
                    outimage[:,:,i] = revised_uniform_image(dat[i,:,jj,:].transpose((1,0)),#dat[i:i+1, :, jj, k],
                                            U_wavs[:,jj],
                                            V_wavs[:,jj],
                                            gridsize,
                                            inject_img=None if inject_img is None or np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/nchans_per_node,
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,wstack=wstack,
                                            w=W_wavs[:,jj],
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM)
                else:
                    outimage[:,:,i] += revised_uniform_image(dat[i,:,jj,:].transpose((1,0)),#dat[i:i+1, :, jj, k],
                                            U_wavs[:,jj],
                                            V_wavs[:,jj],
                                            gridsize,
                                            inject_img=None if inject_img is None or np.all(inject_img[:,:,i]==0) else inject_img[:,:,i]/dat.shape[-1]/nchans_per_node,
                                            inject_flat=inject_flat,
                                            pixel_resolution=pixel_resolution,wstack=wstack,
                                            w=W_wavs[:,jj],
                                            Nlayers_w=Nlayers_w,
                                            pixperFWHM=pixperFWHM)
    return outimage,j


#function to convert npy to psrfits
from pdat import pdat
def set_primary_header(psrfits_object,prim_dict):

    #prim_dict = dictionary of primary header changes

    PF_obj = psrfits_object
    for key in prim_dict.keys():
        PF_obj.replace_FITS_Record('PRIMARY',key,prim_dict[key])

def set_subint_header(psrfits_object,subint_dict):
    #prim_dict = dictionary of primary header changes
    PF_obj = psrfits_object
    for key in subint_dict.keys():
        PF_obj.replace_FITS_Record('SUBINT',key,subint_dict[key])

def make_subint_BinTable(self):
    subint_draft = self.make_HDU_rec_array(self.nsubint, self.subint_dtype)
    return subint_draft

def numpy_to_psrfits(data,path,fnum,fobs,ra,dec,mjd,sample_size=tsamp_ms,nsblk=25,BMIN=3.5,BMAJ=3.5,suffix=""):
    assert(data.shape[1]==len(fobs))
    assert(data.shape[0]%nsblk==0)
    #based on https://pulsardatatoolbox.readthedocs.io/en/latest/psrfits_write.html

    #create file from template
    fname = path + "/nsfrb_psr_" +str(fnum) + suffix + ".fits"
    psrfits1=pdat.psrfits(fname,from_template="/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/search_template.fits")



    #make data params
    N_Time_Bins = data.shape[0]
    ROWS = int(N_Time_Bins//nsblk)
    Total_time = N_Time_Bins*sample_size
    dt = Total_time/N_Time_Bins
    subband =(fobs[0] - fobs[1])*1000
    BW=(((fobs[0] - fobs[1]) + fobs[0] - fobs[-1]))*1000
    N_freq = int(BW/subband)
    Npols = npol =  1
    print('Total_time',Total_time/1e3)
    print('N_freq',N_freq)

    #set sub-integration params
    np.bool = np.bool_
    psrfits1.set_subint_dims(nsblk=nsblk,nchan=N_freq,nsubint=ROWS,npol=Npols)
    subint_draft = psrfits1.make_HDU_rec_array(psrfits1.nsubint, psrfits1.subint_dtype)
    tsubint = nsblk*dt*1e-3 #in seconds
    offs_sub_init = tsubint/2
    offs_sub = np.zeros((ROWS))
    for jj in range(ROWS):
        offs_sub[jj] = offs_sub_init + (jj * tsubint)

    #get RA and galactic coords
    #ra = imaging.get_ra(mjd,dec)
    coord = SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs')
    gl,gb = coord.galactic.l.value,coord.galactic.b.value
    obstime = Time(mjd,format='mjd')
    elev = get_elevation(obstime).value
    alt = float(DSAelev_to_ASTROPYalt(elev)[0])

    ones = np.ones((ROWS))
    #And assign them using arrays of the appropriate sizes
    subint_draft['TSUBINT'] = tsubint * ones
    subint_draft['OFFS_SUB'] = offs_sub
    subint_draft['LST_SUB'] = ((ra*u.deg).to(u.hourangle).value)*86400 * ones
    subint_draft['RA_SUB'] = ra * ones
    subint_draft['DEC_SUB'] = dec * ones
    subint_draft['GLON_SUB'] = gl * ones
    subint_draft['GLAT_SUB'] = gb * ones
    subint_draft['FD_ANG'] = 0 * ones
    subint_draft['POS_ANG'] = 0 * ones
    subint_draft['PAR_ANG'] = 0 * ones
    subint_draft['TEL_AZ'] = 0 * ones
    subint_draft['TEL_ZEN'] = (90-alt) * ones

    #format image
    final_image_for_fits = data.transpose()[np.newaxis,np.newaxis,:,np.newaxis,:]
    final_image_for_fits = final_image_for_fits.reshape((ROWS,1,N_freq,1,nsblk))
    final_image_for_fits = final_image_for_fits*100/np.nanpercentile(final_image_for_fits,95)
    #convert to unsigned 8-bit int
    final_image_for_fits = 255*(final_image_for_fits - np.nanmin(final_image_for_fits))/(np.nanmax(final_image_for_fits) - np.nanmin(final_image_for_fits))
    final_image_for_fits[np.isnan(final_image_for_fits)] = 0
    final_image_for_fits = final_image_for_fits.astype(np.uint8)
    print(final_image_for_fits.shape)
    for ii in range(subint_draft.size):
        subint_draft[ii]['DATA'] = final_image_for_fits[ii,:,:,:,:]
        subint_draft[ii]['DAT_SCL'] = np.ones(N_freq*npol)
        subint_draft[ii]['DAT_OFFS'] = np.zeros(N_freq*npol)
        subint_draft[ii]['DAT_FREQ'] = fobs*1000 #np.linspace(fobs[-1],fobs[0],N_freq)
        subint_draft[ii]['DAT_WTS'] = np.ones(N_freq)

    #make primary headers
    subint_hdr=psrfits1.draft_hdrs['SUBINT']
    pri_dic= {'OBSERVER':'NSFRB','OBSFREQ':int(np.around(np.mean(fobs),1)*1000),'OBSBW':int(np.around(BW,0)),'OBSNCHAN':N_freq,'PROJID':'nsfrb','CHAN_DM':0,
         'TELESCOP':'DSA110','ANT_X':dsaloc.itrs.x.value,'ANT_Y':dsaloc.itrs.y.value,'ANT_Z':dsaloc.itrs.z.value,
         'BACKEND':'NSFRB','DATE-OBS':Time(mjd,format='mjd').isot,'SRC_NAME':'',
         'STT_CRD1':str(np.around(ra,4)),'STT_CRD2':str('+' if dec>0 else '-') + str(np.around(np.abs(dec),4)),
          'TRK_MODE':'SCANLAT',
          'BMIN':str(np.around(BMIN,0)),
          'BMAJ':str(np.around(BMAJ,0)),
         'STP_CRD1':str(np.around(ra,4)),'STP_CRD2':str('+' if dec>0 else '-') + str(np.around(np.abs(dec),4)),
         'SCANLEN':np.around(float(Total_time)/1000,5),
          'STT_IMJD':int(np.floor(mjd)),'STT_SMJD':int(np.floor((mjd - np.floor(mjd))*86400)),'STT_OFFS':0.99999999988222,
         'STT_LST':int(np.floor(86400*(ra/15)))}
    subint_dic = {'TBIN':str(dt/1000),'CHAN_BW':-subband}
    psrfits1.make_FITS_card(subint_hdr,'TBIN',subint_dic['TBIN'])

    set_primary_header(psrfits1,pri_dic)
    set_subint_header(psrfits1,subint_dic)
    psrfits1.HDU_drafts['SUBINT'] = subint_draft
    print("PRIMARY HEADER:")
    print(psrfits1.draft_hdrs['PRIMARY'])
    print("SUBINT HEADER:")
    print(psrfits1.draft_hdrs['SUBINT'])
    psrfits1.write_psrfits()
    psrfits1.close()
    print("Writing data to " + fname)
    return fname



#flagged_antennas = np.arange(101,115,dtype=int) #[21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
def main(args):

    verbose = args.verbose
    #send in sub-gulps
    
    num_gulps = 90
    #num_chans = int(NUM_CHANNELS//AVERAGING_FACTOR)

    #parameters from etcd
    #test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)

    #if verbose: print("TIMESTAMP:",tsamp)
    #get timestamp
    if len(args.timestamp) != 0:
        timestamp = args.timestamp

    if args.multiimage:
        print("Using multi-threaded imaging with ",args.maxProcesses,"threads")
        executor = ThreadPoolExecutor(args.maxProcesses)
    else: executor = None

    dirty_img = np.nan*np.ones((args.gridsize,args.gridsize,args.num_time_samples*num_gulps,args.num_chans))
    """
    def image_future_callback(future):
        print("Callback ",future.result()[1])#,future.result()[2])
        dirty_img[:,:,:,future.result()[1]] = future.result()[0][:,:,:,0] #np.nansum(np.concatenate([dirty_img[:,:,:,future.result()[1],np.newaxis],future.result()[0][:,:,:,np.newaxis]],3),axis=3)
        return
    """

    if args.numpyfile:
        #parameters from etcd
        test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
        ff = 1.53-np.arange(8192)*0.25/8192
        fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
        fobs = np.reshape(fobs,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1)
        #dat = dat_all[gulp*args.num_time_samples:(gulp+1)*args.num_time_samples,:,:,:]


        dirty_img = np.load(psr_dir + args.numpyfile)
        gulp = 0

        if len(args.filedir) == 0:
            fname = args.path + "/lxd110"+ corrs[0] + "/" + ("nsfrb_" + sbs[0] if args.sb else corrs[0]) + args.filelabel + ".out"
        else:
            fname =  args.filedir + "/" + ("nsfrb_" + "00" if sbs[0] else corrs[0]) + args.filelabel + ".out"
        sbnum,tstamp_mjd,Dec = pipeline.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples,gulp=0,nchan=int(args.nchans_per_node),headersize=16,get_header=True)
        if len(args.timestamp) == 0:
            timestamp = Time(tstamp_mjd,format='mjd').isot
        dat = np.nan*np.ones((args.num_time_samples,4656,args.nchans_per_node,2),dtype=complex)

        #use MJD to get pointing
        mjd = Time(timestamp,format='isot').mjd + gulp*args.num_time_samples*tsamp/86400
        time_start_isot = Time(mjd,format='mjd').isot
        #LST = Time(mjd,format='mjd').sidereal_time("mean",longitude=Lon).to(u.hourangle).value
        print("DEC from file:",Dec)



        pt_dec = Dec*np.pi/180.
        if verbose: print("Pointing dec (deg):",pt_dec*180/np.pi)
        bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)

        #flagging andd baseline cut
        fcts = []
        dat, bname, blen, UVW, antenna_order = flag_vis(dat, bname, blen, UVW, antenna_order, list(flagged_antennas) + list(args.flagants), bmin, list(flagged_corrs) + list(args.flagcorrs), flag_channel_templates = fcts)

        U = UVW[0,:,1]
        V = UVW[0,:,0]
        W = UVW[0,:,2]
        uv_diag=np.max(np.sqrt(U**2 + V**2))
        pixel_resolution = (0.20 / uv_diag) / args.pixperFWHM

        #use pixel resolution to get uv max and re-flag
        bmax = (2.998e8/np.max(fobs)/1e9)/pixel_resolution/2
        print("Max baseline:",bmax,"meters")
        dat, bname, blen, UVW, antenna_order = flag_vis(dat, bname, blen, UVW, antenna_order, [], bmin, [], flag_channel_templates =[] , bmax=bmax)

        if verbose: print(antenna_order,len(antenna_order))#x_m.shape,y_m.shape,z_m.shape)
        if verbose: print(UVW.shape,U.shape,V.shape,W.shape)
        if verbose: print(UVW)

        print("Print bad channels:",np.isnan(dat.mean((0,1,3))))

        #pt_RA = LST*15*np.pi/180
        if verbose: print("Time:",time_start_isot)
        #if verbose: print("LST (hr):",LST
        if Dec is None:

            RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,pixperFWHM=args.pixperFWHM)
            #HA_axis = (LST*15) - RA_axis
            HA_axis = RA_axis[int(len(RA_axis)//2)] - RA_axis
            print(HA_axis)
            #HA_axis = RA_axis - RA_axis[int(len(RA_axis)//2)] #want to image the central RA, so the hour angle should be 0 here, right?
            RA = RA_axis[int(len(RA_axis)//2)]
            HA = HA_axis[int(len(HA_axis)//2)]
            Dec = Dec_axis[int(len(Dec_axis)//2)]
        else:
            #RA = get_ra(mjd,Dec) #LST*15
            #HA = 0
            RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,DEC=Dec,pixperFWHM=args.pixperFWHM)
            #HA_axis = (LST*15) - RA_axis
            HA_axis = RA_axis[int(len(RA_axis)//2)] - RA_axis
            RA = RA_axis[int(len(RA_axis)//2)]
            HA = HA_axis[int(len(HA_axis)//2)]
            print(HA_axis[len(HA_axis)//2-10:len(HA_axis)//2+10])
        if verbose: print("Coordinates (deg):",RA,Dec)
        if verbose: print("Hour angle (deg):",HA)

        ra_grid_2D,dec_grid_2D,tmp = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,DEC=Dec,two_dim=True,pixperFWHM=args.pixperFWHM)
        BMIN = np.abs(np.max(ra_grid_2D) - np.min(ra_grid_2D))
        BMAJ = np.abs(np.max(dec_grid_2D) - np.min(dec_grid_2D))


        #convert each pixel to psrfits
        for i in range(dirty_img.shape[0]):
            for j in range(dirty_img.shape[1]):
                rapix = ra_grid_2D[i,j]
                decpix = dec_grid_2D[i,j]
                pos = SkyCoord(rapix*u.deg,decpix*u.deg,frame='icrs')
                poslabel = str('J{RH:02d}{RM:02d}{RS:02d}'.format(RH=int(pos.ra.hms.h),
                                                               RM=int(pos.ra.hms.m),
                                                               RS=int(pos.ra.hms.s)) +
                                        str("+" if pos.dec>=0 else "-") +
                                            '{DD:02d}{DM:02d}{DS:02d}'.format(DD=int(pos.dec.dms.d),
                                                               DM=int(pos.dec.dms.m),
                                                               DS=int(pos.dec.dms.s)))
                numpy_to_psrfits(dirty_img[i,j],psr_dir,int(args.filelabel[1:]),fobs.reshape((len(corrs),args.nchans_per_node)).mean(1),rapix,decpix,mjd,sample_size=tsamp_ms,nsblk=args.num_time_samples*num_gulps,BMIN=BMIN,BMAJ=BMAJ,suffix="_"+poslabel)

        if args.multiimage:
            executor.shutdown()
        return

    for gulp in range(num_gulps):#range(args.gulp_offset - (1 if args.gulp_offset>0 and args.search else 0),args.gulp_offset + num_gulps):
        

        #parameters from etcd
        test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
        ff = 1.53-np.arange(8192)*0.25/8192
        fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
        fobs = np.reshape(fobs,(len(corrs)*args.nchans_per_node,int(NUM_CHANNELS/2/args.nchans_per_node))).mean(axis=1)
        #dat = dat_all[gulp*args.num_time_samples:(gulp+1)*args.num_time_samples,:,:,:]

        dat = None
        Dec = None
        for i in range(len(corrs)):
            corr = corrs[i]
            sb = sbs[i]

            if len(args.filedir) == 0:
                fname = args.path + "/lxd110"+ corr + "/" + ("nsfrb_" + sb if args.sb else corr) + args.filelabel + ".out"
            else:
                fname =  args.filedir + "/" + ("nsfrb_" + sb if args.sb else corr) + args.filelabel + ".out"
            print(fname)
            #fname = args.path + "/lxd110"+ corr + "/" + corr + args.filelabel + ".out"
            #fname = args.path + "/3C286_vis/" + corr + args.filelabel + ".out"
            try:
                #tmp = cal.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples)
                #print("tmp",tmp)

                dat_corr,sbnum,tstamp_mjd,Dec = pipeline.read_raw_vis(fname,datasize=args.datasize,nsamps=args.num_time_samples,gulp=gulp,nchan=int(args.nchans_per_node),headersize=16)
                if len(args.timestamp) == 0: 
                    timestamp = Time(tstamp_mjd,format='mjd').isot
            
                #dat_corr = np.nanmean(dat_corr,axis=2,keepdims=True)
                if verbose: print(dat_corr.shape)
                if dat is None:
                    dat = np.nan*np.ones(dat_corr.shape,dtype=dat_corr.dtype).repeat(len(corrs),axis=2)
                #print(dat_all.shape,dat_corr.shape)
                dat[:,:,i*args.nchans_per_node:(i+1)*args.nchans_per_node,:] = dat_corr
                #print("tmp2",dat_all[:,:,i,:],dat_corr)
            except Exception as exc:
                if verbose: print("No data for " + corr)
                if verbose: print(exc)
        

        print("Are any values nan?:",np.any(np.isnan(dat))) 
        #print(list(np.isnan(dat.mean((0,1,3)))))
        if verbose: print("Gulp size:",dat.shape)

        
        #use MJD to get pointing
        mjd = Time(timestamp,format='isot').mjd + gulp*args.num_time_samples*tsamp/86400
        time_start_isot = Time(mjd,format='mjd').isot
        #LST = Time(mjd,format='mjd').sidereal_time("mean",longitude=Lon).to(u.hourangle).value
        print("DEC from file:",Dec)



        pt_dec = Dec*np.pi/180.
        if verbose: print("Pointing dec (deg):",pt_dec*180/np.pi)
        bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)

        #flagging andd baseline cut
        fcts = []
        if args.flagSWAVE:
            fcts.append(fct_SWAVE)
        if args.flagBPASS:
            fcts.append(fct_BPASS)
        if args.flagFRCBAND:
            fcts.append(fct_FRCBAND)
        if args.flagBPASSBURST:
            fcts.append(fct_BPASSBURST)
        dat, bname, blen, UVW, antenna_order = flag_vis(dat, bname, blen, UVW, antenna_order, list(flagged_antennas) + list(args.flagants), bmin, list(flagged_corrs) + list(args.flagcorrs), flag_channel_templates = fcts)
            
        U = UVW[0,:,1]
        V = UVW[0,:,0]
        W = UVW[0,:,2]
        uv_diag=np.max(np.sqrt(U**2 + V**2))
        pixel_resolution = (0.20 / uv_diag) / args.pixperFWHM

        #use pixel resolution to get uv max and re-flag
        bmax = (2.998e8/np.max(fobs)/1e9)/pixel_resolution/2
        print("Max baseline:",bmax,"meters")
        dat, bname, blen, UVW, antenna_order = flag_vis(dat, bname, blen, UVW, antenna_order, [], bmin, [], flag_channel_templates =[] , bmax=bmax)

        if verbose: print(antenna_order,len(antenna_order))#x_m.shape,y_m.shape,z_m.shape)
        if verbose: print(UVW.shape,U.shape,V.shape,W.shape)
        if verbose: print(UVW)

        print("Print bad channels:",np.isnan(dat.mean((0,1,3))))



        #pt_RA = LST*15*np.pi/180
        if verbose: print("Time:",time_start_isot)
        #if verbose: print("LST (hr):",LST
        if Dec is None:

            RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,pixperFWHM=args.pixperFWHM)
            #HA_axis = (LST*15) - RA_axis
            HA_axis = RA_axis[int(len(RA_axis)//2)] - RA_axis
            print(HA_axis)
            #HA_axis = RA_axis - RA_axis[int(len(RA_axis)//2)] #want to image the central RA, so the hour angle should be 0 here, right?
            RA = RA_axis[int(len(RA_axis)//2)]
            HA = HA_axis[int(len(HA_axis)//2)]
            Dec = Dec_axis[int(len(Dec_axis)//2)]
        else:
            #RA = get_ra(mjd,Dec) #LST*15
            #HA = 0
            RA_axis,Dec_axis,elev = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,DEC=Dec,pixperFWHM=args.pixperFWHM)
            #HA_axis = (LST*15) - RA_axis
            HA_axis = RA_axis[int(len(RA_axis)//2)] - RA_axis
            RA = RA_axis[int(len(RA_axis)//2)]
            HA = HA_axis[int(len(HA_axis)//2)]
            print(HA_axis[len(HA_axis)//2-10:len(HA_axis)//2+10])
        if verbose: print("Coordinates (deg):",RA,Dec)
        if verbose: print("Hour angle (deg):",HA)

        ra_grid_2D,dec_grid_2D,tmp = uv_to_pix(mjd,args.gridsize,flagged_antennas=flagged_antennas,uv_diag=uv_diag,DEC=Dec,two_dim=True,pixperFWHM=args.pixperFWHM)
        BMIN = np.abs(np.max(ra_grid_2D) - np.min(ra_grid_2D))
        BMAJ = np.abs(np.max(dec_grid_2D) - np.min(dec_grid_2D))


        dat[np.isnan(dat)]= 0 
        
        #imaging
        print("Start imaging")
        if args.wstack: print("W-stacking with ",args.Nlayers," layers")
           
        timage = time.time()
        if args.multiimage:
            task_list = []
            for j in range(args.num_chans):
                tgrid = time.time()
                print("gridding in advance...")
                #make U,V,Ws in advance
                U_wavs = np.zeros((len(U),args.nchans_per_node))
                V_wavs = np.zeros((len(V),args.nchans_per_node))
                W_wavs = np.zeros((len(W),args.nchans_per_node))
                i_indices_all = np.zeros(U_wavs.shape,dtype=int)
                j_indices_all = np.zeros(V_wavs.shape,dtype=int)
                k_indices_all = np.zeros(W_wavs.shape,dtype=int)
                i_conj_indices_all = np.zeros(U_wavs.shape,dtype=int)
                j_conj_indices_all = np.zeros(V_wavs.shape,dtype=int)
                k_conj_indices_all = np.zeros(W_wavs.shape,dtype=int)
                bweights_all = np.zeros(U_wavs.shape)
                for jj in range(args.nchans_per_node):
                    chanidx = (args.nchans_per_node*j)+jj
                    U_wavs[:,jj] = U/(2.998e8/fobs[chanidx]/1e9)
                    V_wavs[:,jj] = V/(2.998e8/fobs[chanidx]/1e9)
                    if args.wstack:
                        W_wavs[:,jj] = W/(2.998e8/fobs[chanidx]/1e9)
                    if args.briggs:
                        if args.wstack:
                            i_indices_all[:,jj],j_indices_all[:,jj],k_indices_all[:,jj],i_conj_indices_all[:,jj],j_conj_indices_all[:,jj],k_conj_indices_all[:,jj] = uniform_grid(U_wavs[:,jj], V_wavs[:,jj], args.gridsize, pixel_resolution, args.pixperFWHM, w=W_wavs[:,jj], wstack=args.wstack)
                        else:
                            i_indices_all[:,jj],j_indices_all[:,jj],i_conj_indices_all[:,jj],j_conj_indices_all[:,jj] = uniform_grid(U_wavs[:,jj], V_wavs[:,jj], args.gridsize, pixel_resolution, args.pixperFWHM, w=W_wavs[:,jj], wstack=args.wstack)
                        bweights_all[:,jj] = briggs_weighting(U_wavs[:,jj], V_wavs[:,jj], args.gridsize, robust=args.robust,pixel_resolution=pixel_resolution)
                #ftime = open(timelogfile,"a")
                #ftime.write("[grid] " + str(time.time()-tgrid)+"\n")
                #ftime.close()
                    

                print("submitting task:",j)
                task_list.append(executor.submit(offline_image_task,dat[:,:,j*args.nchans_per_node:(j+1)*args.nchans_per_node,:],
                                                    U_wavs,
                                                    V_wavs,
                                                    i_indices_all,
                                                    j_indices_all,
                                                    i_conj_indices_all,
                                                    j_conj_indices_all,
                                                    bweights_all,
                                                    args.gridsize,
                                                    pixel_resolution,
                                                    args.nchans_per_node,
                                                    fobs[j*args.nchans_per_node:(j+1)*args.nchans_per_node],
                                                    j,
                                                    args.briggs,
                                                    args.robust,
                                                    False,
                                                    None,
                                                    False,
                                                    args.wstack,
                                                    W_wavs,
                                                    k_indices_all,
                                                    k_conj_indices_all,
                                                    args.Nlayers,
                                                    args.pixperFWHM))
            wait(task_list)
            for t in task_list:
                dirty_img[:,:,gulp*args.num_time_samples:(gulp+1)*args.num_time_samples,t.result()[1]] = t.result()[0]
            #ftime = open(timelogfile,"a")
            #ftime.write("[image] " + str(time.time()-timage)+"\n")
            #ftime.close()
        else:
            for j in range(args.num_chans):
                for jj in range(args.nchans_per_node):
                    chanidx = (args.nchans_per_node*j)+jj
                    U_wav = U/(2.998e8/fobs[chanidx]/1e9)
                    V_wav = V/(2.998e8/fobs[chanidx]/1e9)
                    W_wav = None if not args.wstack else W/(2.998e8/fobs[chanidx]/1e9)
                    #uniform_grid(U_wav, V_wav, args.gridsize, pixel_resolution, args.pixperFWHM, w=W_wav, wstack=args.wstack)
                    if args.briggs:
                        if args.wstack:
                            i_indices,j_indices,k_indices,i_conj_indices,j_conj_indices,k_conj_indices = uniform_grid(U_wav, V_wav, args.gridsize, pixel_resolution, args.pixperFWHM, w=W_wav, wstack=args.wstack)
                        else:
                            i_indices,j_indices,i_conj_indices,j_conj_indices = uniform_grid(U_wav, V_wav, args.gridsize, pixel_resolution, args.pixperFWHM, w=W_wav, wstack=args.wstack)
                        #print("indices:",i_indices,j_indices,i_conj_indices,j_conj_indices)
                        bweights = briggs_weighting(U_wav, V_wav, args.gridsize, robust=args.robust,pixel_resolution=pixel_resolution)

                    for i in range(dat.shape[0]):
                        for k in range(dat.shape[-1]):
                            if args.briggs:
                                if k == 0 and jj == 0:
                                    dirty_img[:,:,gulp*args.num_time_samples + i,j] = revised_robust_image(dat[i:i+1, :, chanidx, k],
                                            U_wav,
                                            V_wav,
                                            args.gridsize,
                                            inject_img=None,
                                            robust=args.robust,
                                            inject_flat=False,
                                            pixel_resolution=pixel_resolution,
                                            wstack=args.wstack,
                                            w=W_wav,
                                            Nlayers_w=args.Nlayers,
                                            pixperFWHM=args.pixperFWHM,
                                            i_indices=i_indices,
                                            j_indices=j_indices,
                                            k_indices=None if not args.wstack else k_indices,
                                            i_conj_indices=i_conj_indices,
                                            j_conj_indices=j_conj_indices,
                                            k_conj_indices=None if not args.wstack else k_conj_indices)
                                else:
                                    dirty_img[:,:,gulp*args.num_time_samples + i,j] += revised_robust_image(dat[i:i+1, :, chanidx, k],
                                            U_wav,
                                            V_wav,
                                            args.gridsize,
                                            inject_img=None,
                                            robust=args.robust,
                                            inject_flat=False,
                                            pixel_resolution=pixel_resolution,
                                            wstack=args.wstack,
                                            w=W_wav,
                                            Nlayers_w=args.Nlayers,
                                            pixperFWHM=args.pixperFWHM,
                                            i_indices=i_indices,
                                            j_indices=j_indices,
                                            k_indices=None if not args.wstack else k_indices,
                                            i_conj_indices=i_conj_indices,
                                            j_conj_indices=j_conj_indices,
                                            k_conj_indices=None if not args.wstack else k_conj_indices)
                            else:
                                if k == 0 and jj == 0:
                                    dirty_img[:,:,gulp*args.num_time_samples + i,j] = revised_uniform_image(dat[i:i+1, :, chanidx, k],
                                            U_wav,
                                            V_wav,
                                            args.gridsize,
                                            inject_img=None,
                                            inject_flat=False,
                                            pixel_resolution=pixel_resolution,wstack=args.wstack,
                                            w=W_wav,
                                            Nlayers_w=args.Nlayers,
                                            pixperFWHM=args.pixperFWHM)
                                else:
                                    dirty_img[:,:,gulp*args.num_time_samples + i,j] += revised_uniform_image(dat[i:i+1, :, chanidx, k],
                                            U_wav,
                                            V_wav,
                                            args.gridsize,
                                            inject_img=None,
                                            inject_flat=False,
                                            pixel_resolution=pixel_resolution,wstack=args.wstack,
                                            w=W_wav,
                                            Nlayers_w=args.Nlayers,
                                            pixperFWHM=args.pixperFWHM)
            
                                        
        print("Imaging complete:",time.time()-timage,"s")            
        print(dirty_img)    
       
       
       
    #write the full image to file
    np.save(psr_dir + "nsfrb_psr" + args.filelabel + ".npy",dirty_img)

    #convert each pixel to psrfits
    for i in range(dirty_img.shape[0]):
        for j in range(dirty_img.shape[1]):
            rapix = ra_grid_2D[i,j]
            decpix = dec_grid_2D[i,j]
            pos = SkyCoord(rapix*u.deg,decpix*u.deg,frame='icrs')
            poslabel = str('J{RH:02d}{RM:02d}{RS:02d}'.format(RH=int(pos.ra.hms.h),
                                                               RM=int(pos.ra.hms.m),
                                                               RS=int(pos.ra.hms.s)) +
                                        str("+" if pos.dec>=0 else "-") +
                                            '{DD:02d}{DM:02d}{DS:02d}'.format(DD=int(pos.dec.dms.d),
                                                               DM=int(pos.dec.dms.m),
                                                               DS=int(pos.dec.dms.s)))
            numpy_to_psrfits(dirty_img[i,j],psr_dir,int(args.filelabel[1:]),fobs,rapix,decpix,mjd,sample_size=tsamp_ms,nsblk=args.num_time_samples,BMIN=BMIN,BMAJ=BMAJ,suffix="_"+poslabel)

    if args.multiimage:
        executor.shutdown()
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    parser.add_argument('filelabel')           # positional argument
    parser.add_argument('--numpyfile',type=str,help='Numpy file',default='')
    parser.add_argument('--timestamp',type=str,help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49); if not given, timestamp is retrieved from sb00 file with os.path.getctime() or from time of rsync',default='')
    parser.add_argument('--filedir',type=str,help='Path to fast visibilities; if not given, the /dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/lxd110h**/ paths are used',default='')
    #parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=-1)
    #parser.add_argument('--gulp_offset',type=int,help='Gulp offset to start from, default = 0', default=0) #always use all 5 minutes
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    #parser.add_argument('--fringestop', action='store_true', default=False, help='Fringe stop manually')
    #parser.add_argument('--fringetable',type=str,help='Fringe stop manually with specified table in the dsa110-nsfrb-fast-visibilities dir',default='')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=raw_datasize)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vis_dir[:-1])
    parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    #parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    #parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file') #always save, that's the point
    parser.add_argument('--offline',action='store_true',default=False,help='Initializes previous frame with noise')
    parser.add_argument('--sb',action='store_true',default=False,help='Use nsfrb_sbxx names')
    parser.add_argument('--num_chans',type=int,help='Number of channels',default=int(NUM_CHANNELS//AVERAGING_FACTOR))
    parser.add_argument('--nchans_per_node',type=int,help='Number of channels per corr node prior to imaging',default=8)
    parser.add_argument('--briggs',action='store_true',help='If set use robust weighted gridding with \'briggs\' weighting')
    parser.add_argument('--robust',type=float,help='Briggs factor for robust imaging',default=0)
    parser.add_argument('--sleeptime',type=float,help='Time to sleep between processing gulps (seconds)',default=30)
    parser.add_argument('--bmin',type=float,help='Minimum baseline length to include, default=20 meters',default=bmin)
    parser.add_argument('--wstack',action='store_true',help='If set use w-stacking algorithm with --Nlayers layers')
    parser.add_argument('--Nlayers',type=int,help='Number of layers for w-stacking',default=18)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--flagSWAVE',action='store_true',help='Flag channels when SWAVE template RFI is detected, which manifests as a 2 Hz sin wave over ~5 minutes of data')
    parser.add_argument('--flagBPASS',action='store_true',help='Flag channels when BPASS template RFI is detected, which is simpl comparison to bandpass mean in visibilities')
    parser.add_argument('--flagFRCBAND',action='store_true',help='Flag channels in FRC miltiary allocation 1435-1525 MHz')
    parser.add_argument('--flagBPASSBURST',action='store_true',help='Flag channel when BPASS template RFI is detected in any timestep, i.e. should detect pulsed narrowband RFI')
    parser.add_argument('--flagcorrs',type=int,nargs='+',default=[],help='List of sb nodes [0-15] to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--flagants',type=int,nargs='+',default=[],help='List of antennas to flag, in addition to whichever ones are in nsfrb.config')
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of processes used for multithreading; only used if --multiimage is set; default=16',default=16)
    parser.add_argument('--multiimage',action='store_true',help='If set, uses multithreading for imaging')
    parser.add_argument('--pixperFWHM',type=float,help='Pixels per FWHM, default 3',default=pixperFWHM)
    args = parser.parse_args()
    main(args)



