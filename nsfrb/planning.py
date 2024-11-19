import numpy as np
import csv
from matplotlib import pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord,EarthLocation,AltAz,ICRS,Galactic
import astropy.units as u
from scipy.stats import norm,uniform
import copy
from scipy.interpolate import interp1d

from nsfrb.config import plan_dir

"""
This module contains functions for observation planning, including making DSA-110 observing scripts giving desired elevation at subsequent timesteps.
"""


def GP_curve(longitude,lat_offset=0):
    """
    This function takes the Galactic longitude and returns RA and DEC lying on the Galactic Plane. If lat_offset is specified, the points are offset by a constant Galactic Latitude
    """
    cc = SkyCoord(b=lat_offset*u.deg,l=longitude*u.deg,frame='galactic')
    return cc.icrs.ra.value,cc.icrs.dec.value


az_offset=1.23001
Lat=37.23
Lon=-118.2851
def DSAelev_to_ASTROPYalt(elev,az=az_offset):
    """
    DSA110 uses elevation from 0 to 180 with azimuth fixed at 1.23 deg
    Astropy uses altitude from -90 to 90 (<0 = below the horizon) and azimuth 0 to 360
    This function converts in between them.
    
    elev: DSA-110 specified elevation
    az_offset: offset from perfect az=0
    """

    elev = np.array(elev)

    #if elevation > 90, need to shift to 0-90 range
    alt = copy.deepcopy(elev)
    alt[elev>90] = 180 - alt[elev>90]

    #if elevation > 90, need to rotate az by 180 deg
    az = np.array(az*np.ones_like(alt))
    az[elev>90] = 180 + az[elev>90]
    return alt,az




def find_plane(mjd,elev,el_slew_rate=0.5368867455531618,resolution=3,Lat=37.23,Lon=-118.2851,az_offset=az_offset,maxtime=(180/0.5368867455531618)/3600,sys_time_offset=10*60,verbose=False,gb_offset=0): #slew rate 0.1 deg/s from https://www.antesky.com/project/4-5m-cku-dual-bands-tvro-antenna/
    """
    This function takes the current time an elevation of the DSA-110 and finds the fastest path to the Galactic Plane, returning the required elevation and time of crossing.
    """

    #make a high-res interpolation for plane
    raGP,decGP = GP_curve(np.linspace(0,360,100000),gb_offset)
    interpGP = interp1d(raGP,decGP,fill_value='extrapolate')
    interpGP_inv1 = interp1d(decGP[raGP>180],raGP[raGP>180],fill_value='extrapolate')
    interpGP_inv2 = interp1d(decGP[raGP<=180],raGP[raGP<=180],fill_value='extrapolate')
    def interpGPwrapped(raGP):
        return interpGP(raGP%360)
    def interpGP_invwrapped(decGP):
        return interpGP_inv1(((decGP+90)%180) - 90),interpGP_inv2(((decGP+90)%180) - 90)


    mjd += sys_time_offset/86400 #sys_time_offset is time for signal to reach all antennas. Vikram says command must be given twice, maybe 10 seconds?

    #need to do this with fixed azimuth...
    #so first convert current pointing to RA, DEC
    loc = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg) #default is ovro
    #alt = (90 - elev if elev <= 90 else 180 -(90-elev))
    #az = (az_offset if elev > 90 else 180-az_offset)
    alt,az = DSAelev_to_ASTROPYalt(elev,az_offset)#elev-90


    antpos = AltAz(obstime=Time(mjd,format='mjd'),location=loc,az=az*u.deg,alt=alt*u.deg)
    icrspos = antpos.transform_to(ICRS())
    ra = icrspos.ra.value
    dec = icrspos.dec.value




    #get longitude/latitude
    cc = SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs')
    gl = cc.galactic.l.value
    gb = cc.galactic.b.value

    #make time axis (seconds)
    #print(int(maxtime/(resolution/15)))
    t_steps = np.linspace(0,maxtime,int(maxtime/(resolution/15)))*3600
    mjd_steps = mjd + (t_steps/86400)
    #print(mjd_steps)

    #for given point, get the point on GP that could be reached at each timestep, if possible
    if dec < interpGPwrapped(ra):#(dec > interpGPwrapped(ra) and elev <90) or (dec <= interpGPwrapped(ra) and elev >=90):#gb > 0:
        if verbose: print("DECREASING ELEV")
        elev_steps = np.clip(elev - el_slew_rate*t_steps,0,180)

    else:
        if verbose: print("INCREASING ELEV")
        elev_steps = np.clip(elev + el_slew_rate*t_steps,0,180)

    alt_steps,az_steps = DSAelev_to_ASTROPYalt(elev_steps,az_offset)#elev_steps - 90
    #az_steps = az_offset*np.ones_like(t_steps)
    #print(elev_steps)

    antpos_steps = AltAz(obstime=Time(mjd_steps,format='mjd'),location=loc,az=az_steps*u.deg,alt=alt_steps*u.deg)
    icrspos_steps = antpos_steps.transform_to(ICRS())
    ra_steps = ra_plane = icrspos_steps.ra.value
    dec_steps = icrspos_steps.dec.value




    dec_plane = interpGPwrapped(ra_steps)


    #ra_plane = (ra + t_steps*15)%360
    #dec_plane = interpGPwrapped(ra_plane)

    #find where reachable declinations cross the plane, i.e. first zero crossing

    try:
        best_step = np.where((dec_plane-dec_steps)/np.abs(dec_plane-dec_steps) != (dec_plane-dec_steps)[0]/np.abs(dec_plane-dec_steps)[0])[0][0]#np.argmin(np.abs(dec_plane-dec_steps))
        best_t = t_steps[best_step]



    except IndexError as exc:
        #print("Can't reach plane by slewing, wait for crossing")


        #find the possible RAs at the current dec
        dec_int = dec_steps[-1]
        ra_int1,ra_int2 = interpGP_invwrapped(dec_int)

        #find time required to reach the plane
        #antpos_observer = Observer(location=loc)#,az=az_steps*u.deg,alt=alt_steps*u.deg)
        #GPpasstimes = antpos_observer.target_meridian_transit_time(Time(mjd_steps[-1],format='mjd'),
        #                                                           SkyCoord(ra=np.array([ra_int1,ra_int2])*u.deg,dec=dec_int*u.deg,frame='icrs'),
        #                                                          which='next')
        GPpasstimes_mjd = [mjd_steps[-1]+(ra_int1-ra_steps[-1] if ra_int1>ra_steps[-1] else ra_int1+(360-ra_steps[-1]))/15/24,
                          mjd_steps[-1]+(ra_int2-ra_steps[-1] if ra_int2>ra_steps[-1] else ra_int2+(360-ra_steps[-1]))/15/24]
        if verbose: print(mjd,GPpasstimes_mjd)

        #choose the one that passes sooner
        #ra_int = (ra_int1 if GPpasstimes[0]<GPpasstimes[1] else ra_int2)
        mjd_int = np.min(GPpasstimes_mjd)

        #get true crossing position given time, alt, and az
        antpos_int = AltAz(obstime=Time(mjd_int,format='mjd'),location=loc,az=az_steps[-1]*u.deg,alt=alt_steps[-1]*u.deg)
        icrspos_int = antpos_int.transform_to(ICRS())
        ra_int = icrspos_int.ra.value
        dec_int = icrspos_int.dec.value


        if verbose: print(ra_int,dec_int)

        dec_steps = np.concatenate([dec_steps,[dec_int]])
        ra_steps = np.concatenate([ra_steps,[ra_int]])
        ra_plane = np.concatenate([ra_plane,[ra_int]])
        dec_plane = interpGPwrapped(ra_plane)
        mjd_steps = np.concatenate([mjd_steps,[(ra_int-ra_steps[-2])/15/24 + mjd_steps[-1]]])
        az_steps = np.concatenate([az_steps,[az_steps[-1]]])
        alt_steps = np.concatenate([alt_steps,[alt_steps[-1]]])
        elev_steps = np.concatenate([elev_steps,[elev_steps[-1]]])
        t_steps = np.concatenate([t_steps,[(ra_int-ra_steps[-2])*3600/15]])
        best_step = len(dec_steps)-1
        best_t = t_steps[best_step]


    cc = SkyCoord(ra=ra_steps*u.deg,dec=dec_steps*u.deg,frame='icrs')
    gl_steps = cc.galactic.l.value
    gb_steps = cc.galactic.b.value

    #return the time, ra, and dec of intersection
    mjd_int = mjd_steps[best_step]
    ra_int = ra_steps[best_step]
    dec_int = dec_steps[best_step]
    az_int = az_steps[best_step]
    elev_int = elev_steps[best_step]
    alt_int = alt_steps[best_step]


    return mjd_int,ra_int,dec_int,az_int,elev_int,ra_steps,dec_steps,dec_plane,elev_steps,az_steps,alt_steps,mjd_steps,gb_steps,gl_steps#ra_plane,dec_plane,dec_steps,best_step

def track_plane(mjd,elev,el_slew_rate=0.5368867455531618,resolution=1,subresolution=10000,Lat=37.23,Lon=-118.2851,az_offset=az_offset,sys_time_offset=10,verbose=False,gb_offset=0): #slew rate 0.1 deg/s from https://www.antesky.com/project/4-5m-cku-dual-bands-tvro-antenna/
    """
    This function takes the time and elevation at which the DSA-110 points to the Galactic plane and returns an observing plan for tracking the plane until no-longer visible
    """

    #make a high-res interpolation for plane
    raGP,decGP = GP_curve(np.linspace(0,360,100000),gb_offset)
    interpGP = interp1d(raGP,decGP,fill_value='extrapolate')
    interpGP_inv1 = interp1d(decGP[raGP>180],raGP[raGP>180],fill_value='extrapolate')
    interpGP_inv2 = interp1d(decGP[raGP<=180],raGP[raGP<=180],fill_value='extrapolate')
    def interpGPwrapped(raGP):
        return interpGP(raGP%360)
    def interpGP_invwrapped(decGP):
        return interpGP_inv1(((decGP+90)%180) - 90),interpGP_inv2(((decGP+90)%180) - 90)
    
    
    #set number of steps to minimum needed to cover the plane
    nsteps = 2*int(360/resolution)
    
    #set max slew time, time to go from 0 to 180
    max_slew_time = 90/el_slew_rate #s
    if verbose: print("Maximum slew time:",max_slew_time,"s")
    loc = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg) #default is ovro
    #need to do this with fixed azimuth...
    #so first convert current pointing to RA, DEC
    #alt = (90 - elev if elev <= 90 else 180 -(90-elev))
    #az = (az_offset if elev > 90 else 180-az_offset)
    alt,az = DSAelev_to_ASTROPYalt(elev,az_offset)#elev-90
    #az = az_offset
    antpos = AltAz(obstime=Time(mjd,format='mjd'),location=loc,az=az*u.deg,alt=alt*u.deg)
    icrspos = antpos.transform_to(ICRS())
    ra = icrspos.ra.value
    dec = icrspos.dec.value
    #print(dec)
    
    #get longitude/latitude
    cc = SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs')
    gl = cc.galactic.l.value
    gb = cc.galactic.b.value
    
    
    #if spacing=='deg': #-->resolution in degrees
    
    gl_steps = [gl]
    gb_steps = [gb]#*np.ones(nsteps)
    ra_steps = [ra]
    dec_steps = [dec]
    
    az_steps = [az_offset]#*np.ones(nsteps)
    elev_steps = [elev]
    alt_steps = [alt]
    mjd_steps = [mjd]
    t_slews = [0]
    t_waits = [0]
    t_steps = [0]
    flipped_steps = [False]
    #for each timestep we can see when the plane point passes the meridian
    #antpos_observer = Observer(location=loc)#,az=az_steps*u.deg,alt=alt_steps*u.deg)
    flipped=False
    skips = [False]
    lastskips=0
    for i in range(1,nsteps):
        if verbose: print(lastskips)
        
        #step in galactic longitude evenly
        gl_steps.append((gl_steps[i-1-lastskips] + resolution)%360)#(gl + np.arange(nsteps)*resolution)%360
        gb_steps.append(gb)
        cc = SkyCoord(l=gl_steps[i]*u.deg,b=gb_steps[i]*u.deg,frame='galactic')
        ra_steps.append(cc.icrs.ra.value)
        dec_steps.append(cc.icrs.dec.value)

        
        #print(ra_steps[i-1],ra_steps[i])
        #print(dec_steps[i-1],dec_steps[i])
        #print(gl_steps[i-1],gl_steps[i])
        #print("")
        
        #GPpasstime = antpos_observer.target_meridian_transit_time(Time(mjd_steps[i-1],format='mjd'),
        #                                                       SkyCoord(ra=ra_steps[i]*u.deg,dec=dec_steps[i]*u.deg,frame='icrs'),
        #                                                      which='next')
        GPpasstime_mjd = mjd_steps[i-1-lastskips] + (resolution/15/24)/np.cos(dec_steps[i-1-lastskips]*np.pi/180)
        #print(mjd,GPpasstime.mjd)
        
        #sample the time up to this mjd and the elevation in either direction
        #print((GPpasstime.mjd-mjd_steps[i-1])*24)
        
        #if already at 180 or 0, need to wrap elevation
        tsamps = np.concatenate([np.linspace(0,max_slew_time,subresolution),
                                 -np.linspace(0,max_slew_time,subresolution)]) #s
        """if (np.around(elev_steps[i-1-lastskips])==180):# and np.abs(180-elev_steps[i-1])<resolution) or (len(elev_steps)>3 and elev_steps[i-2]==elev_steps[i-1]):
            flipped = True
            if verbose: print("flipped 180-->0")
            flipped_steps.append(True)
        elif (np.around(elev_steps[i-1-lastskips])==0):# and np.abs(0-elev_steps[i-1])<resolution) or (len(elev_steps)>3 and elev_steps[i-2]==elev_steps[i-1]):
            flipped = True
            if verbose: print("flipped 0-->180")
            flipped_steps.append(True)
        else:
            flipped_steps.append(False)"""
        flipped_steps.append(False)
        elsamps = (elev_steps[i-1-lastskips] + el_slew_rate*tsamps)%180
        altsamps,azsamps = DSAelev_to_ASTROPYalt(elsamps,az_offset)#elsamps - 90

        #find which one brings us closest to the plane point
        #print(elev_steps[i-1])
        #print(elsamps)
        antpos_i = AltAz(obstime=Time(mjd_steps[i-1-lastskips] + (tsamps/86400),format='mjd'),location=loc,az=azsamps*u.deg,alt=altsamps*u.deg)
        icrs_i = antpos_i.transform_to(ICRS())
        coord_i = SkyCoord(ra=icrs_i.ra,dec=icrs_i.dec,frame='icrs')
        #plt.subplot(3,1,1)
        
        bestidx = np.argmin(SkyCoord(ra=ra_steps[i-1]*u.deg,dec=dec_steps[i]*u.deg,frame='icrs').separation(coord_i))
        #bestidx = np.argmin(SkyCoord(l=gl_steps[i]*u.deg,b=gb_steps[i]*u.deg,frame='galactic').separation(coord_i))
        #bestidx = np.argmin(np.abs(dec_steps[i]-coord_i.dec.value))
        if verbose: print("Elevation change:",elev_steps[i-1-lastskips],"to",elsamps[bestidx],"in",np.abs(tsamps[bestidx]),'s')
        #add the systematic time offset to get the total slew time
        t_slew_i = np.abs(tsamps[bestidx]) + sys_time_offset #s
        
        #find the time to wait for crossing
        tsamps = np.linspace(0,100*max_slew_time,subresolution) #s
        
        elsamps_wait = elsamps[bestidx]*np.ones_like(tsamps)
        altsamps_wait,azsamps_wait = DSAelev_to_ASTROPYalt(elsamps_wait,az_offset)#elsamps_wait - 90
        antpos_wait_i = AltAz(obstime=Time(mjd_steps[i-1-lastskips] + (t_slew_i/86400) +  (tsamps/86400),format='mjd'),location=loc,az=azsamps_wait*u.deg,alt=altsamps_wait*u.deg)
        icrs_wait_i = antpos_wait_i.transform_to(ICRS())
        coord_wait_i = SkyCoord(ra=icrs_wait_i.ra,dec=icrs_wait_i.dec,frame='icrs')
        bestidx_wait = np.argmin(SkyCoord(ra=ra_steps[i]*u.deg,dec=dec_steps[i]*u.deg,frame='icrs').separation(coord_wait_i))
        t_wait_i = tsamps[bestidx_wait]
        
        #plt.plot(coord_wait_i.ra.value/15,coord_wait_i.dec.value)
        
        
        
        if verbose: print(np.abs(coord_i[bestidx].galactic.b.value) ,flipped)
        
        
        
       
            
        
        
        #print(ra_steps[i],dec_steps[i],coord_i[bestidx],coord_i)
        
        #final results
        mjd_steps.append(mjd_steps[i-1-lastskips] + (t_slew_i + t_wait_i)/86400) #np.abs(tsamps[bestidx])/86400)
        t_slews.append(t_slew_i)
        t_waits.append(t_wait_i)
        t_steps.append(t_slew_i+t_wait_i)
        elev_steps.append(elsamps[bestidx])
        alt_steps.append(DSAelev_to_ASTROPYalt(elsamps[bestidx],az_offset)[0])#elsamps[bestidx]-90)
        az_steps.append(DSAelev_to_ASTROPYalt(elsamps[bestidx],az_offset)[1])
        ra_steps[i] = icrs_wait_i[bestidx_wait].ra.value
        dec_steps[i] = icrs_wait_i[bestidx_wait].dec.value
        gl_steps[i] = coord_wait_i[bestidx_wait].galactic.l.value
        gb_steps[i] = coord_wait_i[bestidx_wait].galactic.b.value
        #print(ra_steps[i-1],ra_steps[i],ra_steps[i])
        #print("")
        #confirm that point can be reached at this time
        if t_wait_i < 0:
            if verbose: print("slew takes too long")
            skips.append(True)
            lastskips += 1
        else:
            lastskips = 0
        
        if (np.around(elev_steps[i]) == np.around(elev_steps[i-1]) == 0) or (np.around(elev_steps[i]) == np.around(elev_steps[i-1]) == 180):# and flipped:
            if verbose: print("breaking")
            break
        
    gl_steps = np.array(gl_steps)
    gb_steps = np.array(gb_steps)
    ra_steps = np.array(ra_steps)
    dec_steps = np.array(dec_steps)
    elev_steps = np.array(elev_steps)
    alt_steps = np.array(alt_steps)
    az_steps = np.array(az_steps)
    mjd_steps = np.array(mjd_steps)
    flipped_steps = np.array(flipped_steps)
    t_slews = np.array(t_slews)
    t_waits = np.array(t_waits)
    t_steps = np.array(t_steps)
    
    
    #get the total observing time
    obs_time = (mjd_steps[-1]-mjd)*24 #hrs
    if verbose: print("Total observing time:",obs_time," hours")
    
    #get total plane coverage
    obs_deg = (gl_steps[-1]-gl_steps[0] if gl_steps[-1]>gl_steps[0] else gl_steps[-1] + (360-gl_steps[0]))
    if verbose: print("Total plane coverage:",obs_deg," degrees")
    return mjd_steps,ra_steps,dec_steps,az_steps,elev_steps,gl_steps,gb_steps,flipped_steps,obs_time,obs_deg,t_waits,t_slews#slew_time,nsteps

dec_limit = -52
gl_limits = np.array([275,330])
def make_gl_grid(gl_res,gl_limits=gl_limits):
    """
    This function takes the desired angular spacing of GP tracking pointings andd returns a regular grid of Galactic longitudes.
    """
    
    return np.concatenate([np.arange(0,gl_limits[0],gl_res),np.arange(gl_limits[1],360.0,gl_res)])


def lock_to_grid(mjd_steps,elev_steps,gl_grid,Lat=37.23,Lon=-118.2851,az_offset=az_offset):
    #given mjds and elevations, want to lock to gl grid by interpolating
    """
    Given an observing plan of mjd steps and elevations, this function locks the observing points as near as possible to a regular grid of observing points. Computed with e.g. make_gl_grid().
    """

    #get coordinates
    loc = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg) #default is ovro
    alt_steps,az_steps = DSAelev_to_ASTROPYalt(elev_steps,az_offset)#elev_steps-90
    antpos = AltAz(obstime=Time(mjd_steps,format='mjd'),location=loc,az=az_steps*u.deg,alt=alt_steps*u.deg)
    galpos = antpos.transform_to(Galactic())
    gl_steps = galpos.l.value
    gb_steps = galpos.b.value

    #interpolate
    interp_mjd = interp1d(list(gl_steps),list(mjd_steps),fill_value=(mjd_steps[0],mjd_steps[-1]),bounds_error=False)
    interp_elev = interp1d(list(gl_steps),list(elev_steps),fill_value=(elev_steps[0],elev_steps[-1]),bounds_error=False)


    new_mjd_steps = np.array([np.nan if np.sum(gl_grid>=gl_steps[i])==0 else interp_mjd(gl_grid[gl_grid>=gl_steps[i]][np.argmin(np.abs(gl_steps[i]-gl_grid[gl_grid>=gl_steps[i]]))]) for i in range(len(gl_steps))])
    new_elev_steps = np.array(np.clip([np.nan if np.sum(gl_grid>=gl_steps[i])==0 else interp_elev(gl_grid[gl_grid>=gl_steps[i]][np.argmin(np.abs(gl_steps[i]-gl_grid[gl_grid>=gl_steps[i]]))]) for i in range(len(gl_steps))],0,180))
    new_mjd_steps = new_mjd_steps[~np.isnan(new_mjd_steps)]
    new_elev_steps = new_elev_steps[~np.isnan(new_elev_steps)]

    #get new ra,dec,etc
    loc = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg) #default is ovro
    new_alt_steps,new_az_steps = DSAelev_to_ASTROPYalt(new_elev_steps,az_offset)#new_elev_steps-90
    antpos = AltAz(obstime=Time(new_mjd_steps,format='mjd'),location=loc,az=new_az_steps*u.deg,alt=new_alt_steps*u.deg)
    galpos = antpos.transform_to(Galactic())
    new_gl_steps = galpos.l.value
    new_gb_steps = galpos.b.value
    icrspos = antpos.transform_to(ICRS())
    new_ra_steps = icrspos.ra.value
    new_dec_steps = icrspos.dec.value

    #get the total observing time
    new_obs_time = (np.max(new_mjd_steps)-np.min(new_mjd_steps[0]))*24 #hrs
    print("Locked Total observing time:",new_obs_time," hours")

    #get total plane coverage
    new_obs_deg = (new_gl_steps[-1]-new_gl_steps[0] if new_gl_steps[-1]>new_gl_steps[0] else new_gl_steps[-1] + (360-new_gl_steps[0]))

    new_obs_deg = (new_gl_steps[-1]-np.min(new_gl_steps)) + (new_gl_steps[np.argmin(new_gl_steps[-1])-1]-new_gl_steps[0])
    print("Locked Total plane coverage:",new_obs_deg," degrees")



    return new_mjd_steps,new_elev_steps,new_alt_steps,new_ra_steps,new_dec_steps,new_gl_steps,new_gb_steps,new_obs_time,new_obs_deg


def generate_plane(mjd,elev,el_slew_rate=0.5368867455531618,resolution=1,subresolution=1000,Lat=37.23,Lon=-118.2851,az_offset=az_offset,sys_time_offset=0,savefile=True,plot=False,show=False,gb_offset=0,gl_grid=None,plan_dir=plan_dir):
    """
    This function generates an observing plan to track the Galactic Plane given the current mjd and elevation, as a list of mjds and elevations. The plan is output as numpy arrays and written to a csv.
    """
    if plot:
        #make a high-res interpolation for plane
        raGP,decGP = GP_curve(np.linspace(0,360,100000),gb_offset)
        interpGP = interp1d(raGP,decGP,fill_value='extrapolate')
        interpGP_inv1 = interp1d(decGP[raGP>180],raGP[raGP>180],fill_value='extrapolate')
        interpGP_inv2 = interp1d(decGP[raGP<=180],raGP[raGP<=180],fill_value='extrapolate')
        def interpGPwrapped(raGP):
            return interpGP(raGP%360)
        def interpGP_invwrapped(decGP):
            return interpGP_inv1(((decGP+90)%180) - 90),interpGP_inv2(((decGP+90)%180) - 90)


        plt.figure(figsize=(18,30))

        plt.subplot(3,1,1)
        plt.plot(np.linspace(0,360,1000)/15,interpGP(np.linspace(0,360,1000)))
        plt.xlabel("RA")
        plt.ylabel("DEC")

        plt.subplot(3,1,2)
        plt.axhline(0)
        plt.xlabel("GL")
        plt.ylabel("GB")
        plt.xlim(0,360)
        plt.ylim(-90,90)

        plt.subplot(3,1,3)
        plt.ylim(0,180)
        plt.xlabel("Time (MJD)")
        plt.ylabel("Elevation")
        
        loc = EarthLocation(lat=Lat*u.deg,lon=Lon*u.deg) #default is ovro
        alt,az = DSAelev_to_ASTROPYalt(elev,az_offset)#elev-90
        #az = az_offset
        antpos = AltAz(obstime=Time(mjd,format='mjd'),location=loc,az=az*u.deg,alt=alt*u.deg)
        icrspos = antpos.transform_to(ICRS())
        ra = icrspos.ra.value
        dec = icrspos.dec.value
        
        



    #first get to the plane
    mjd_int,ra_int,dec_int,az_int,elev_int,ra_steps,dec_steps,dec_plane,elev_steps,az_steps,alt_steps,mjd_steps,gb_steps,gl_steps = find_plane(mjd,elev,el_slew_rate=el_slew_rate,resolution=resolution*15/3600,Lat=Lat,Lon=Lon,az_offset=az_offset,sys_time_offset=sys_time_offset,gb_offset=gb_offset)
    print("Start UTC:",Time(mjd_int,format='mjd').isot)
    print("Start Elevation:",elev_int,'degrees')
    print("GP Offset:",gb_offset,'degrees')
    
    if plot:
        plt.subplot(3,1,1)
        c=plt.plot(ra/15,dec,'o',markersize=20,alpha=1)
        plt.plot(ra_steps[0]/15,dec_steps[0],'o',markersize=10,alpha=1,color=c[0].get_color())
        plt.plot(ra_steps/15,dec_steps,'.-',color=c[0].get_color())
        
    
    #then iterate with equal steps in gl
    mjd_steps,ra_steps,dec_steps,az_steps,elev_steps,gl_steps,gb_steps,flipped_steps,obs_time,obs_deg,t_waits,t_slews = track_plane(mjd_int,elev_int,el_slew_rate=el_slew_rate,resolution=resolution,subresolution=subresolution,Lat=Lat,Lon=Lon,az_offset=az_offset,sys_time_offset=sys_time_offset,gb_offset=gb_offset)
    
    #lock to the grid points if needed
    if gl_grid is not None:
        mjd_steps,elev_steps,alt_steps,ra_steps,dec_steps,gl_steps,gb_steps,obs_time,obs_deg = lock_to_grid(mjd_steps,elev_steps,gl_grid,az_offset=az_offset)
        #flipped_steps = flipped_steps[:-1]
        #az_steps = az_steps[:-1]
    print("Total Number of Observations:",len(mjd_steps))
    print("Total Observing Time:",obs_time,'hours')
    print("Total Galactic Longitude coverage:",obs_deg,'deg')
    
    
    
    if plot:
        
        plt.subplot(3,1,1)
        c=plt.plot(ra_steps/15,dec_steps,'o',markersize=10,alpha=0.5)
        #plt.plot(ra_steps[flipped_steps]/15,dec_steps[flipped_steps],'x',markersize=20,alpha=0.5,color='red')
        plt.plot(ra_int/15,dec_int,'v',markersize=20,alpha=0.5,color=c[0].get_color())
        
        plt.subplot(3,1,2)
        if gl_grid is not None:
            plt.plot(gl_grid,gb_offset*np.ones_like(gl_grid),'s',alpha=0.5,markersize=20,markeredgecolor='black')
        plt.plot(gl_steps,gb_steps,'o')
        
        
        plt.subplot(3,1,3)
        plt.plot(mjd_steps,elev_steps,'o')
        fnameplot=plan_dir + "GP_observing_plan_" + str(Time(mjd,format='mjd').isot) + ".png"
        if savefile:
            print("Saving summary plot to ",fnameplot)
            plt.savefig(fnameplot)
        if show: plt.show()
        else: plt.close()
            
    #output to file
    if savefile:
        fname=plan_dir + "GP_observing_plan_" + str(Time(mjd,format='mjd').isot) + ".csv"
        print("Writing observing plan to ",fname)
        with open(fname,"w") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            for i in range(len(mjd_steps)):
                wr.writerow([mjd_steps[i],elev_steps[i]])
                
    
    return mjd_steps,elev_steps