from nsfrb.flagging import flag_vis
from dsamfs import utils as pu
import argparse
import numpy as np
import struct
from nsfrb import config




def main(args):

    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None)
    if ~np.isnan(args.pt_dec):
        print("Using input pointing dec = ",args.pt_dec*180/np.pi,"degrees")
        pt_dec = args.pt_dec
    bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)

    #write antennas to .bin
    ant1 = []
    ant2 = []
    for i in range(len(bname)):
        ant1.append(list(bname)[i][:list(bname)[i].index("-")])
        ant2.append(list(bname)[i][list(bname)[i].index("-")+1:])



    """
    nsamps=25
    nchans_per_node = 8
    bmin=20

    tmp, bname, blen, UVW, antenna_order,keep = flag_vis(np.zeros((nsamps,UVW.shape[1],nchans_per_node,2)), bname, blen, UVW, antenna_order, config.flagged_antennas, bmin, [], flag_channel_templates = [], flagged_chans=[], returnidxs=True)
    """
    print("Final UVW Shape:"+str(UVW.shape))
    UVW = UVW.astype(np.float64)
    blen = np.sqrt(UVW[0,:,0]**2 + UVW[0,:,1]**2).astype(np.float64)
    with open(config.table_dir + "U.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(struct.pack("<d",UVW[0,i,0]))
    with open(config.table_dir + "V.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(struct.pack("<d",UVW[0,i,1]))
    with open(config.table_dir + "W.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(struct.pack("<d",UVW[0,i,2]))
    with open(config.table_dir + "BLEN.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(struct.pack("<d",blen[i]))
    with open(config.table_dir + "ANT1.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(np.uint8(ant1[i]).tobytes())
    with open(config.table_dir + "ANT2.bin","wb") as f:
        for i in range(UVW.shape[1]):
            f.write(np.uint8(ant2[i]).tobytes())
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Update UVW coordinates for cuda imager')
    parser.add_argument("--pt_dec",type=float,help="pointing declination in radians. If not specified, uses value from current configuration",default=np.nan) 
    args = parser.parse_args()
    main(args)
