import numpy as np
import struct
from dsamfs import utils as pu
import sys


"""
Adapted from /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/gen_nsfrb_fstable.py to run nsfrb fringestopping in real-time
"""


offline_caltable_path = "/home/ubuntu/data/calTable.out"
def make_fstable(my_pt_dec,iNode,bfweights_path = "/home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out",vis_path = "/home/ubuntu/data/visModel.npz",fobs=None,caltable_path="",antenna_order=None,refmjd=None,outrigger_delays=None,bname=None,blen=None,uvw=None):


    # set up params
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
    pt_dec = my_pt_dec*np.pi/180.
    if fobs is None:
        ff = 1.53-np.arange(8192)*0.25/8192
        fobs = ff[1024+(iNode)*384:1024+(iNode+1)*384]

    # calc uvw
    bname, blen, uvw = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
        
    # make vis model
    vis_model = pu.load_visibility_model(vis_path,blen, 25, fobs, pt_dec, tsamp, antenna_order, outrigger_delays, bname, refmjd)
    vis_model = vis_model[0,:,:,:,0] # now [time, baseline, channel]

    # get weights
    f = open(bfweights_path,'rb')
    data = f.read()
    vv = np.asarray(struct.unpack("<18624f",data)).astype(np.float32)
    vals = vv[2*96:]
    vals = vals.reshape((96,48,2,2))
    w = 1.*vals[:,:,:,0] + 1j*vals[:,:,:,1]

    # make cal table from weights
    cal_table = np.zeros((4656, 48, 8, 2),dtype=np.complex64)
    bi = 0
    for i in np.arange(96):
        for j in np.arange(i+1):  
            for k in np.arange(8):
                cal_table[bi,:,k,:] = w[i,:,:]*np.conjugate(w[j,:,:])
            bi += 1
    cal_table = cal_table.reshape((4656, 384, 2)) # now [baseline, channel, pol]

    # make output table
    output_table = np.zeros((25, 4656, 384, 2),dtype=np.complex64)
    
    for iT in np.arange(25):
        for i in np.arange(2):
            output_table[iT,:,:,i] = cal_table[:,:,i] / vis_model[iT,:,:]

    output_table /= np.abs(output_table)
    output_table = output_table.astype(np.complex64).view(np.float32)

    # write out
    if len(caltable_path) > 0:
        with open(caltable_path,"wb") as fout:
            fout.write(bytes(output_table))
        fout.close()
        print(f"written {caltable_path}")
    return output_table



### --> create function to re-fringestop for slow and image differenceing search (only use with offline system b/c it requires re-imaging...
from nsfrb.config import table_dir
def refstop_SLOW(bin_slow,Dec,fobs_GHz=None,iNode=0,fname="",vis_path=table_dir+"/tmp_visModel.npz"):

    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
    pt_dec = Dec*np.pi/180
    if fobs_GHz is None:
        ff = 1.53-np.arange(8192)*0.25/8192
        fobs = ff[1024+(iNode)*384:1024+(iNode+1)*384]
    else:
        fobs = fobs_GHz

    # calc uvw
    bname, blen, uvw = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)

    # make ORIGINAL vis model
    vis_model = pu.load_visibility_model(vis_path,blen, 25, fobs, pt_dec, tsamp, antenna_order, outrigger_delays, bname, refmjd)
    vis_model = vis_model[0,:,:,:,0] # now [time, baseline, channel]

    # make NEW vis model
    vis_model_slow = pu.load_visibility_model(vis_path,blen, 25*bin_slow, fobs, pt_dec, tsamp, antenna_order, outrigger_delays, bname, refmjd)
    vis_model_slow = vis_model_slow[0,:,:,:,0] # now [time, baseline, channel]

    #combine
    refstop_table = np.zeros_like(vis_model_slow)
    for i in range(bin_slow):
        refstop_table[i*25:(i+1)*25,:,:] = vis_model/vis_model_slow[i*25:(i+1)*25,:,:]
    refstop_table /= np.abs(refstop_table)

    if len(fname)>0:
        np.save(fname,refstop_table)
    return refstop_table
