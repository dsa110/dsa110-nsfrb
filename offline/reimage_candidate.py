from nsfrb import plotting
import argparse
from astropy.time import Time
from nsfrb.config import *



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Offline re-imaging with outriggers')
    parser.add_argument('T_interval',type=float)           # positional argument
    parser.add_argument('isot',type=str)
    parser.add_argument('--full_array',action='store_true',help='Include outriggers in imaging')
    parser.add_argument('--image_size',type=int,help='Size of image',default=1001)
    parser.add_argument('--gif',action='store_true',help='Create a gif')
    parser.add_argument('--visfile_dir',type=str,help='Directory where visibilities are stored',default=vis_dir)
    parser.add_argument('--gulpsize',type=int,help='Number of samples in gulp',default=nsamps)
    parser.add_argument('--nchan',type=int,help='Number of channels in each visibility file',default=2)
    parser.add_argument('--headersize',type=int,help='Size of header in bytes',default=16)
    parser.add_argument('--binsize',type=int,help='Number of samples in a bin',default=5)
    parser.add_argument('--bmin',type=int,help='Minimum baseline',default=20)
    parser.add_argument('--sbimg',type=int,help='Sub-band to image',default=None)
    parser.add_argument('--output_dir',type=str,help='Directory to output png or gif',default=vis_dir)
    parser.add_argument('--viewsize',type=float,help='Size of the image in degrees',default=2)
    args = parser.parse_args()

    print(args)

    plotting.make_image_from_vis(args.T_interval,Time(args.isot,format='isot').mjd,full_array=args.full_array,image_size=args.image_size,gif=args.gif,visfile_dir=args.visfile_dir,gulpsize=args.gulpsize,nchan=args.nchan,headersize=args.headersize,binsize=args.binsize,bmin=args.bmin,sbimg=args.sbimg,output_dir=args.output_dir,viewsize=args.viewsize)
