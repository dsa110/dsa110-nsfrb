import csv
import struct
from matplotlib import pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, ICRS,SkyCoord
from astropy.time import Time
import astropy.units as u
from influxdb import DataFrameClient
import os
import sys

cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
from nsfrb.config import *
from nsfrb.noise import noise_update,noise_dir,noise_update_all
from nsfrb import jax_funcs

#output_dir = cwd + "/tmpoutput/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/"
coordfile = cwd + "/DSA110_Station_Coordinates.csv" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/DSA110_Station_Coordinates.csv"
output_file = cwd + "-logfiles/search_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt"
cand_dir = cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
processfile = cwd + "-logfiles/process_log.txt"
frame_dir = cwd + "-frames/"
psf_dir = cwd + "-PSF/"
f=open(output_file,"w")
f.close()



influx = DataFrameClient('influxdbservice.pro.pvt', 8086, 'root', 'root', 'dsa110')
def get_phase_center(mjd_obs,uv_diag=UVMAX,Lat=37.23,Lon=-118.2851,timerangems=100,maxtries=5,output_file=output_file):
    """
    This function queries etcd to get the coorinates at the current pointing.
    This will be used to set the phase center/reference point for the next 3.25 s 
    chunk of data.
    """

    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout


    #(1) ovro location
    loc = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg) #default is ovro

    #(2) observation time
    tobs = Time(mjd_obs,format='mjd')
    tms = int(tobs.unix*1000) #ms

    #(3) query antenna elevation at obs time
    result = dict()
    tries = 0
    while len(result) == 0 and tries < maxtries:
        query = f'SELECT time,ant_el FROM "antmon" WHERE time >= {tms-timerangems}ms and time < {tms+timerangems}ms'
        result = influx.query(query)
        timerangems *= 10
        tries += 1
    if len(result) == 0:
        print("Failed to retrieve elevation, using RA,DEC = 0,0",file=fout)
        icrs_pos = ICRS(ra=0*u.deg,dec=0*u.deg)
    else:
        bestidx = np.argmin(np.abs(tobs.mjd - Time(np.array(result['antmon'].index),format='datetime').mjd))
        elev = result['antmon']['ant_el'].values[bestidx]
        alt = 180-elev
        alt = elev - 90
        antpos = AltAz(obstime=tobs,location=loc,az=0*u.deg,alt=alt*u.deg)

        #(4) convert to ICRS frame
        icrs_pos = antpos.transform_to(ICRS())

    print("Retrieved Coordinates: " + str(tobs.isot) + ", RA="+str(icrs_pos.ra.value) + "deg, DEC="+str(icrs_pos.dec.value) + "deg",file=fout)
    if output_file != "":
        fout.close()
    return icrs_pos.ra.value,icrs_pos.dec.value


def make_phase_table(U,V,W,ra_center,dec_center,ra_point,dec_point,verbose=False):
    """
    This function computes the phase terms for each baseline to be phase referenced
    to the meridian.
    """
    #get direction cosines
    coord = SkyCoord(ra=ra_point*u.deg,dec=dec_point*u.deg,frame='icrs')
    coord0 = SkyCoord(ra=ra_center*u.deg,dec=dec_center*u.deg,frame='icrs')
    l,m,n = coord.cartesian.x.value/coord.distance.value,coord.cartesian.y.value/coord.distance.value,coord.cartesian.z.value/coord.distance.value
    l0,m0,n0 = coord0.cartesian.x.value/coord0.distance.value,coord0.cartesian.y.value/coord0.distance.value,coord0.cartesian.z.value/coord0.distance.value
    if verbose:
        print(l,m,l0,m0)
        print('uterm',U*(l-l0))
        print('vterm',V*(m-m0))
        print('wterm',W*(n-n0))
    return np.exp(-1j*2*np.pi*(U*(l-l0) + V*(m-m0) + W*(n-n0)))
    """if np.all(w==0):
        return np.exp(-1j*2*np.pi*(u*(l-l0) +
                                v*(m-m0)))
    else:
        return np.exp(-1j*2*np.pi*(u*(l-l0) + 
                                v*(m-m0) + 
                                w*(((1 - l**2 - m**2)**0.5-1) - 
                                    ((1 - l0**2 - m0**2)**0.5-1))))"""

def read_raw_vis(fname,datasize=4,nbase=4656,nchan=384,npol=2,nsamps=-1,gulp=0,headersize=8):
    """
    Read raw visibility data from given file.
    fname: file name
    datasize: size of data in bytes
    headersize: size of header in bytes (1/2 headersize is sub-band number, 1/2 headersize is mjd as float)
    """
    if datasize==4:
        dtype = np.float32
        dtypecomplex = np.complex64
    elif datasize==2:
        dtype = np.float16
        dtypecomplex = np.complex32
    elif datasize==8:
        dtype = np.float64
        dtypecomplex = np.complex128
    elif datasize==16:
        dtype = np.float128
        dtypecomplex = np.complex256
    else:
        print("Invalid datasize")
        return None


    f = open(fname,"rb")

    #first read header
    if headersize != 0:
        sbnum = int.from_bytes(f.read(headersize//2),sys.byteorder,signed=False)
        mjd = struct.unpack(('>' if sys.byteorder=='big' else '<') + 'f', f.read(headersize//2))[0]
    
    if nsamps == -1:
        raw_data = np.frombuffer(f.read(),dtype=dtype) #default reads all time samples
    else:
        print("Reading from byte",gulp*nsamps*nbase*nchan*npol*2*datasize)
        f.seek(gulp*nsamps*nbase*nchan*npol*2*datasize)
        raw_data = np.frombuffer(f.read(nsamps*nbase*nchan*npol*2*datasize),dtype=dtype)
    f.close()

    ntimes = int(len(raw_data)/nbase/nchan/npol/2)

    dat = raw_data.reshape((ntimes,nbase,nchan,npol,2))

    #real and imaginary
    dat_complex = np.zeros(dat.shape[:-1],dtype=dtypecomplex)
    dat_complex[:,:,:,:] = dat[:,:,:,:,0] + 1j*dat[:,:,:,:,1]
    if headersize == 0:
        return dat_complex
    else:
        return dat_complex,sbnum,mjd

#21 22 23 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 
EX_ANTENNAS = ['DSA-021',
               'DSA-022',
               'DSA-023',
               'DSA-052',
               'DSA-053',
               'DSA-054',
               'DSA-055',
               'DSA-056',
               'DSA-057',
               'DSA-058',
               'DSA-059',
               'DSA-060',
               'DSA-061',
               'DSA-062',
               'DSA-063',
               'DSA-064',
               'DSA-065',
               'DSA-066',
               'DSA-067',
               'DSA-117']
"""
f = open("/home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat","r")
EX_ANTENNASdat = f.read().split("\n")[:-1]
f.close()
EX_ANTENNAS = ['DSA-' + str('0'+EX_ANTENNASdat[i] if len(EX_ANTENNASdat[i])==2 else EX_ANTENNASdat[i]) for i in range(len(EX_ANTENNASdat))]
print(EX_ANTENNAS)
"""
def antpos_to_uv(coordfile=coordfile,frequency=1.4e9,plot=False,core_only=False):
    #read antenna positions
    ANTENNALONS = []
    ANTENNALATS = []
    ANTENNAELEVS = []
    ANTENNANAMES = []
    with open(coordfile,'r') as csvfile:
        rdr = csv.reader(csvfile,delimiter=',')
        i = 0
        for row in rdr:
            #print(row)
            if row[1][:3] == 'DSA' and (row[1] not in EX_ANTENNAS) and row[1] != 'DSA-110 Station Coordinates':
                ANTENNALATS.append(float(row[2]))
                ANTENNALONS.append(float(row[3]))
                ANTENNANAMES.append(row[1])
                if row[4] == '':
                    ANTENNAELEVS.append(np.nan)
                else:
                    ANTENNAELEVS.append(float(row[4]))
    csvfile.close()

    ANTENNALATS = np.array(ANTENNALATS)
    ANTENNALONS = np.array(ANTENNALONS)
    ANTENNAELEVS = np.array(ANTENNAELEVS)
    ANTENNANAMES = np.array(ANTENNANAMES)

    if core_only:
        core_lat_lims = [37.2325,37.2375]
        core_lon_lims = [-0.0875-1.182e2,-0.0800-1.182e2]
        core_condition = np.logical_and(np.logical_and(ANTENNALATS>core_lat_lims[0],ANTENNALATS<core_lat_lims[1]),
                                                np.logical_and(ANTENNALONS>core_lon_lims[0],ANTENNALONS<core_lon_lims[1]))

    if plot:
        plt.figure()
        plt.plot(ANTENNALONS,ANTENNALATS,'o')

        if core_only:
            plt.plot(ANTENNALONS[core_condition],
                    ANTENNALATS[core_condition],'o')


        plt.savefig("DSAANTENNAPOSITIONS.png")
        plt.close()
    print("total antennas:",len(ANTENNALATS))
    if core_only:
        ANTENNALONS = ANTENNALONS[core_condition]
        ANTENNALATS = ANTENNALATS[core_condition]
        ANTENNAELEVS = ANTENNAELEVS[core_condition]
        ANTENNANAMES = ANTENNANAMES[core_condition]
        print("using core:",len(ANTENNALATS),ANTENNANAMES)
    print(ANTENNANAMES)
    # get uvw coordinates
    antpos = EarthLocation(lat=ANTENNALATS*u.deg, lon=ANTENNALONS*u.deg,height=ANTENNAELEVS*u.m)
    c = 3e8 #m/s
    wav = c/frequency #m
    U = []
    V = []
    W = []
    for i in range(len(antpos)):
        for j in range(i+1,len(antpos)):
            U.append((antpos[i].x.value-antpos[j].x.value)/wav)
            V.append((antpos[i].y.value-antpos[j].y.value)/wav)
            W.append((antpos[i].z.value-antpos[j].z.value)/wav)

    return U,V,W


def apply_phasecal(data,fringetable):
    """
    Applies fringe stopping calibration given the visibilities and fringe table dict
    """
    

