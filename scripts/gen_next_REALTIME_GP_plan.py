#get mjd for a given day
import argparse
from nsfrb import config
import json
from nsfrb.planning import generate_plane_timetile
from astropy.time import Time
from astropy.coordinates import SkyCoord,AltAz,Galactic,ICRS,EarthLocation
from nsfrb.config import Lon,Lat,Height
from nsfrb.imaging import DSAelev_to_ASTROPYalt
from astropy import units as u
from nsfrb.config import plan_dir
import csv
import glob
from nsfrb.config import Lat
from nsfrb.planning import find_source_pass
from nsfrb.imaging import get_ra
from nsfrb.planning import find_source_pass,find_plane,find_fast_vis_label
from matplotlib import pyplot as plt
import numpy as np
import os
from matplotlib import pyplot as plt
import numpy as np
from nsfrb.imaging import DSAelev_to_ASTROPYalt
from matplotlib import colormaps
from nsfrb.config import vis_dir

from nsfrb.config import Lat
import copy
from dsautils.coordinates import create_WCS,get_declination,get_elevation
from nsfrb.planning import find_plane
from astropy.coordinates import EarthLocation, AltAz, ICRS,SkyCoord,Galactic
"""
This script generates the next set of galactic plane elevation commands based on which have been
completed before. Plans will be automatically marked as completed, but you can manually reset them by editing the
json file "completed" field at h24:/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-plans/REALTIME*json
"""


def main(args):
    plane = SkyCoord(l=np.linspace(0,360,1000)*u.deg,b=0*u.deg,frame='galactic')

    with open(config.plan_dir+"/REALTIME_GP_SURVEY_DECTRACKS.json","r") as jfile:
        psets = json.load(jfile)

    mjd_now = Time.now().mjd
    elev_now = get_elevation(Time(mjd_now,format='mjd')).value #+ (psets[i][0][0] - (69.04-90 + config.Lat))
    print("current MJD:",mjd_now)
    print("current elevation:",elev_now)

    #check if above/below the plane first
    loc = EarthLocation(lat=config.Lat*u.deg,lon=config.Lon*u.deg,height=config.Height*u.m) #default is ovro
    alt,az = DSAelev_to_ASTROPYalt(elev_now,0)
    antpos = AltAz(obstime=Time(mjd_now,format='mjd'),location=loc,az=az*u.deg,alt=alt*u.deg)
    gpos = antpos.transform_to(Galactic())
    icrspos = antpos.transform_to(ICRS())
    print("current pointing:",icrspos)
    print("Looking for plan...")
    for k_ in psets.keys():
        
        if (gpos.b.value>0 and psets[k_][0][1]) or (gpos.b.value<0 and not psets[k_][0][1]) or len(psets[k_])<args.minslews or len(psets[k_])>args.maxslews:
            continue

        #check if already completed
        jglobs = glob.glob(config.plan_dir+"GP_observing_plan_*json")
        print(jglobs)
        jfound = False
        for jglob in jglobs:
            with open(jglob,"r") as jfile:
                metadata = json.load(jfile)
            if int(k_) in metadata['dectrack_ids'] and metadata['completed'][metadata['dectrack_ids'].index(int(k_))]==1:
                print(metadata)
                jfound = True
                break
        if jfound:
            continue
    
        pset = copy.deepcopy(psets[k_])
        print(pset)
    
    
        #pair it with another
        pset_ids = [int(k_)]
        if args.both:
            print("looking for plan to search other side of plane...")
            for kk in psets.keys():
                pset2 = psets[kk]
                #print(np.array(pairkeys).flatten())
                if k_ != kk and (pset[-1][1]!=pset2[0][1]) and np.abs(pset[-1][0]-pset2[0][0])<5 and len(pset)+len(pset2)>=args.minslews and len(pset)+len(pset2)<=args.maxslews:# and (len(pset1)+len(pset2) > 2):# ((len(pset1)>1 and len(pset2)==1) or (len(pset1)==1 and len(pset2)>1)):
                    print(int(k_),int(kk))
                    pset += copy.deepcopy(pset2)
                    pset_ids.append(int(kk))
                    break
    
            print(pset,k_,kk)

        plt.figure()
        plt.plot(plane.icrs.ra.value,plane.icrs.dec.value,color='black')
        plt.plot(icrspos.ra.value,icrspos.dec.value,"*",color='purple')
    
        mjd_steps = []
        elev_steps = []
        ra_steps = []
        dec_steps = []
        for j in range(len(pset)):
        
            startra=pset[j][-2]
            startdec=(elev_now-90 + config.Lat) #psets[i][0][0]
            startpos = SkyCoord(ra=startra*u.deg,dec=startdec*u.deg,frame='icrs')
            startstr = str('{RH:02d}h{RM:02d}m{RS:02d}s'.format(RH=int(startpos.ra.hms.h),
                                                                       RM=int(startpos.ra.hms.m),
                                                                       RS=int(startpos.ra.hms.s)) +
                                   str("+" if startpos.dec>=0 else "-") +
                                   '{DD:02d}d{DM:02d}m{DS:02d}s'.format(DD=int(startpos.dec.dms.d),
                                                                       DM=int(startpos.dec.dms.m),
                                                                       DS=int(startpos.dec.dms.s)))
            gb_offset = np.abs(SkyCoord(ra=startra*u.deg,dec=startdec*u.deg,frame='icrs').galactic.b.value)
        
        
        
        
            #find the files within the timestamp
            timeax = Time(mjd_now + np.linspace(0,16/24,1000),format='mjd')
            DSA = EarthLocation(lat=config.Lat*u.deg,lon=config.Lon*u.deg,height=config.Height*u.m)
            hourvis = SkyCoord(startstr,frame='icrs',location=DSA,obstime=timeax)
        
            #narrow to best minute
            antpos = hourvis.transform_to(AltAz)
            timeax = Time(timeax[np.argmax(antpos.alt.value)].mjd + np.linspace(-1,1,24)/24,format='mjd')
            minvis = SkyCoord(startstr,location=DSA,obstime=timeax)
        
            #narrow to best second
            antpos = minvis.transform_to(AltAz)
            timeax = Time(timeax[np.argmax(antpos.alt.value)].mjd + np.linspace(-1/60,1/60,24)/24,format='mjd')
            secvis = SkyCoord(startstr,location=DSA,obstime=timeax)
        
            antpos = secvis.transform_to(AltAz)
            besttime = timeax[np.argmax(antpos.alt.value)]
        
        
            #subtract slew time
            finaltime = besttime - ((((pset[j][0] - (elev_now-90 + config.Lat))/args.el_slew_rate)/86400) + args.sys_time_offset/86400)
        
            print(mjd_now,finaltime)
    
    
            loc = EarthLocation(lat=config.Lat*u.deg,lon=config.Lon*u.deg,height=config.Height*u.m) #default is ovro
            alt,az = DSAelev_to_ASTROPYalt(elev_now,0)
            antpos = AltAz(obstime=Time(mjd_now,format='mjd'),location=loc,az=az*u.deg,alt=alt*u.deg)
            icrspos = antpos.transform_to(ICRS())
            plt.plot(icrspos.ra.value,icrspos.dec.value,'o',color='red')
            
            loc = EarthLocation(lat=config.Lat*u.deg,lon=config.Lon*u.deg,height=config.Height*u.m) #default is ovro
            alt,az = DSAelev_to_ASTROPYalt(pset[j][0]+90 -config.Lat,0)
            antpos = AltAz(obstime=Time(besttime,format='mjd'),location=loc,az=az*u.deg,alt=alt*u.deg)
            icrspos = antpos.transform_to(ICRS())
            plt.plot(icrspos.ra.value,icrspos.dec.value,'o',color='blue')
        
        
            loc = EarthLocation(lat=config.Lat*u.deg,lon=config.Lon*u.deg,height=config.Height*u.m) #default is ovro
            alt,az = DSAelev_to_ASTROPYalt(pset[j][0]+90 -config.Lat,0)
            antpos = AltAz(obstime=Time(finaltime,format='mjd'),location=loc,az=az*u.deg,alt=alt*u.deg)
            icrspos = antpos.transform_to(ICRS())
            plt.plot(icrspos.ra.value,icrspos.dec.value,'o',color='green')
        
        
            mjd_steps.append(finaltime.mjd)
            elev_steps.append(pset[j][0] + 90 - config.Lat)
            dec_steps.append(pset[j][0])
            ra_steps.append(icrspos.ra.value)
            elev_now = pset[j][0] + 90 - config.Lat
            mjd_now = besttime.mjd# + (pset[j][2]/24/60)
    
        fulltrack_ra = np.array([[pset[k][-2],pset[k][-1]] for k in range(len(pset))]).flatten() #+ [pset[k][-1] for k in range(len(pset))]
        fulltrack_dec = np.array([[pset[k][0],pset[k][0]] for k in range(len(pset))]).flatten() #+ [pset[k][0] for k in range(len(pset))]
        #print(fulltrack_ra,fulltrack_dec)
    
        plt.plot(fulltrack_ra,fulltrack_dec,'.-',color='black')
    
        plt.savefig(config.plan_dir +"GP_observing_plan_"+ str(Time(mjd_steps[0],format='mjd').isot) + ".pdf")

        #output to file
        fname=config.plan_dir + "GP_observing_plan_" + str(Time(mjd_steps[0],format='mjd').isot) + ".csv"
        print("Writing observing plan to ",fname)
        with open(fname,"w") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            for i in range(len(mjd_steps)):
                wr.writerow([mjd_steps[i],elev_steps[i]])
    
        print("Copying to h23")
        os.system("scp "+ fname + " lxd110h23.pro.pvt:/home/ubuntu/proj/dsa110-shell/dsa110-observer/")


        fname=plan_dir + "GP_observing_plan_" + str(Time(mjd_steps[0],format='mjd').isot) + ".json"
        print("Writing plan metadata to ",fname)
        metadata = dict()
        metadata['start_mjd'] = mjd_steps[0]
        metadata['start_isot'] = Time(mjd_steps[0],format='mjd').isot
        metadata['plan_file'] = plan_dir + "GP_observing_plan_" + str(Time(mjd_steps[0],format='mjd').isot) + ".csv"
        metadata['start_elev'] = elev_steps[0]
        metadata['start_dec'] = elev_steps[0]-90 + config.Lat
        metadata['start_ra'] = ra_steps[0]
        metadata['full_obs_time_hr'] = (pset[-1][2]/60) + (besttime.mjd - mjd_steps[0])*24
        metadata['full_obs_range_deg'] = np.nanmax(ra_steps)-np.nanmin(ra_steps)
        metadata['stop_mjd'] = mjd_steps[-1]
        metadata['stop_isot'] = Time(mjd_steps[-1],format='mjd').isot
        metadata['plan_format'] = "MJD," + str("ELEV")
        metadata['dectrack_ids'] = list(pset_ids)
        metadata['completed'] = [0]*len(pset_ids)
        metadata['realtime'] = 1
        f =open(fname,'w')
        json.dump(metadata,f)
        f.close()

   
        return 0

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--el_slew_rate",type=float,help="Approximate antenna slew rate in deg/s, default 0.5368867455531618 deg/s",default=0.5368867455531618)
    parser.add_argument("--sys_time_offset",type=float,help="Hard-coded delay time before start of observing in seconds, default 720 s (12 min)",default=12*60)
    parser.add_argument("--both",action='store_true',help='If possible, include plans for both sides of the Galactic Plane')
    parser.add_argument("--minslews",type=int,help="Ensure that plan has minimum number of slews, default=1",default=1)
    parser.add_argument("--maxslews",type=int,help="Ensure that plan has maximum number of slews, default=10",default=10)
    args = parser.parse_args()
    main(args)
