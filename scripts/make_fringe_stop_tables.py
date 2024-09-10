import numpy as np
import pickle as pkl
import os
import sys
import h5py
import nsfrb.calibration as cal
import argparse
from nsfrb import config
from astropy.time import Time

cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
vis_dir = cwd + '-fast-visibilities/'
def main(args):

    #define time axis
    phase_time_axis = np.arange(0,24*3600,args.phase_interval)#np.arange(0,(1/60)*3600,args.phase_interval)
    fringe_time_axis = np.arange(0,args.chunk_interval,args.fringe_interval)
    if args.verbose: print("num phases:",len(phase_time_axis))
    if args.verbose: print("num fringes:",len(fringe_time_axis))

    #define dec axis
    dec_axis = np.arange(-90,90,args.dec_interval) + args.dec_interval/2

    #open hdf5
    tb = h5py.File(args.template)
    time_col = np.zeros(tb['Header']['time_array'].shape)
    tb['Header']['time_array'].read_direct(time_col)# Get the entire TIME column, saved as JULIAN DAY
    time_col = ((time_col - 2400000.5) - np.floor(time_col - 2400000.5))*86400 # convert to seconds from start of day
    time_col = time_col - np.min(time_col) #convert to offset
    if args.verbose: print("time",np.unique(time_col))


    # Find the minimum and maximum times
    begin_time = np.min(time_col)
    end_time = np.max(time_col)
    tInt = tb['Header']['integration_time'][0]

    #for each phase interval, get the pointing
    caltable = dict()
    for dec_center in dec_axis:
        caltable[dec_center] = dict()
        j = 0
        caltable_i = np.nan*np.ones((int(args.nbase),len(fringe_time_axis),len(phase_time_axis)),dtype=complex)
        for phase in phase_time_axis:
            i = 0
            ra_center = phase*15/3600
            for fringe in fringe_time_axis:
                time_start = begin_time + tInt*(fringe//tInt) #tInt * num_time_samples * gulp
                time_end = begin_time + tInt*((fringe//tInt) + 1) #tInt * num_time_samples * (gulp + 1)
                if args.verbose: print("timestart/end:",time_start,time_end,args.fringe_interval)
                idx_start = np.argmin(np.abs(time_col-time_start))
                idx_end = np.argmin(np.abs(time_col-time_end))
                if args.verbose: print(idx_start,idx_end)
                #if idx_end != idx_start:
                    
                uvw_selected = tb['Header']['uvw_array'][idx_start:idx_end,:].transpose()
                if args.verbose: print(uvw_selected.shape,uvw_selected[0, :].reshape((len(uvw_selected[0, :])//args.nbase,args.nbase)).shape)
                u = uvw_selected[0, :].reshape((len(uvw_selected[0, :])//args.nbase,args.nbase)).mean(0)
                v = uvw_selected[1, :].reshape((len(uvw_selected[1, :])//args.nbase,args.nbase)).mean(0)
                w = np.zeros(int(args.nbase))#uvw_selected[2, :].reshape((len(uvw_selected[2, :])//args.nbase,args.nbase)).mean(0)#np.zeros(len(u))
                if args.verbose: print("u",u,u.shape)
                if args.verbose: print("v",v,v.shape)
                if args.verbose: print("w",w,w.shape)
                if args.verbose: print(u.shape,v.shape,args.nbase)
                
                ra_point = ra_center + ((time_end-time_start)*np.cos(dec_center*np.pi/180)*15/3600)
                dec_point = dec_center
                #caltable[dec_center][ra_point] = 
                caltable_i[:,i,j] = cal.make_phase_table(u,v,w,ra_center,dec_center,ra_point,dec_point,verbose=args.verbose)
                if args.verbose: print(ra_center,dec_center,ra_point,dec_point)
                if args.verbose: print(cal.make_phase_table(u,v,w,ra_center,dec_center,ra_point,dec_point))
                i += 1
            j += 1
        caltable[dec_center] = caltable_i
    #save to pkl file
    f = open(vis_dir + "fringetable_" + str(args.phase_interval) + "_" + str(args.fringe_interval) + ".pkl","wb")
    pkl.dump(caltable,f)
    f.close()
    
    return caltable


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template',type=str,help='Path of template visibility file',default="/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-fast-visibilities/test_files/2024-06-12T21:30:51_sb00.hdf5")
    parser.add_argument('--phase_interval',type=float,help='Time interval between computing the phase_center (s0),default=3.25s',default=3.25)
    parser.add_argument('--chunk_interval',type=float,help='Time interval for each chunk of samples,default=3.25s',default=3.25)
    parser.add_argument('--fringe_interval',type=float,help='Time interval over which pointing is assumed fixed (s), default=0.130s',default=0.130)
    parser.add_argument('--dec_interval',type=float,help='Declination interval for which phase correction is assumed constant,default=10 degrees',default=10)
    parser.add_argument('--lat',type=float,help='Latitude of observatory',default=37.23)
    parser.add_argument('--lon',type=float,help='Longitude of observatory',default=-118.2851)
    parser.add_argument('--nbase',type=int,help='Number of baselines',default=int(96*97/2))
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    main(args)
