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


def make_phase_table(u,v,w,ra_center,dec_center,ra_point,dec_point,verbose=False):
    """
    This function computes the phase terms for each baseline to be phase referenced
    to the meridian.
    """
    l = np.cos(ra_center*np.pi/180)
    m = np.cos((90+dec_center)*np.pi/180)
    l0 = np.cos(ra_point*np.pi/180)
    m0 = np.cos((90+dec_point)*np.pi/180)
    if verbose:
        print(l,m,l0,m0)
        print('uterm',u*(l-l0))
        print('vterm',v*(m-m0))
        print('wterm',w*(((1 - l**2 - m**2)**0.5-1) -
                                    ((1 - l0**2 - m0**2)**0.5-1)))
    if np.all(w==0):
        return np.exp(-1j*2*np.pi*(u*(l-l0) +
                                v*(m-m0)))
    else:
        return np.exp(-1j*2*np.pi*(u*(l-l0) + 
                                v*(m-m0) + 
                                w*(((1 - l**2 - m**2)**0.5-1) - 
                                    ((1 - l0**2 - m0**2)**0.5-1))))

