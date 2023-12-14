import numpy as np
from matplotlib import pyplot as plt
import random
import copy
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import truncnorm
from scipy.signal import peak_widths
from scipy.stats import norm

#from gen_dmtrials_copy import gen_dm


from scipy.ndimage import convolve
from scipy.signal import convolve2d

fsize=45
fsize2=35
plt.rcParams.update({
                    'font.size': fsize,
                    'font.family': 'sans-serif',
                    'axes.labelsize': fsize,
                    'axes.titlesize': fsize,
                    'xtick.labelsize': fsize,
                    'ytick.labelsize': fsize,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 1,
                    'lines.markersize': 5,
                    'legend.fontsize': fsize2,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})

"""
This file contains Python functions for the offline slow transient (NSFRB) search. 
Myles Sherman
"""

"""
Directory for output data
"""
output_dir = "/media/ubuntu/ssd/sherman/NSFRB_search_output/"
output_dir = "./NSFRB_search_output/"


"""
Search parameters
"""
tsamp = 130 #ms
T = 3250 #ms
nsamps = int(T/tsamp)

fmax  = 1530 #MHz
fmin = 1280 #MHz
c = 3e8 #m/s
fc = 1400 #MHz
lambdac = (c/(fc*1e6)) #m
nchans = 16 #16 coarse channels
chanbw = (fmax-fmin)/nchans #MHz
telescope_diameter = 4.65 #m


#resolution parameters
pixsize = 0.002962513099862611#(48/3600)*np.pi/180 #rad
gridsize = 32#256
RA_point = 0 #rad
DEC_point = 0 #rad

#create axes
RA_axis = np.linspace(RA_point-pixsize*gridsize//2,RA_point+pixsize*gridsize//2,gridsize)
DEC_axis = np.linspace(DEC_point-pixsize*gridsize//2,DEC_point+pixsize*gridsize//2,gridsize)
time_axis = np.linspace(0,T,nsamps) #ms
freq_axis = np.linspace(fmin,fmax,nchans) #MHz

#width trials
nwidths=4
widthtrials = np.logspace(0,3,nwidths,base=2,dtype=int)

#DM trials
def gen_dm(dm1,dm2,tol,nu,nchan,tsamp,B):
    #tol = 1.25 # S/N loss tolerance
    #nu = 1.405 # center frequency (GHz)
    #nchan = 1024 # number of channels
    #tsamp = 262.144 # sampling time (microseconds)
    #B = 250./nchan # bandwidth per channel (MHz)

    ndms = 1
    dm_prev = dm1
    dm = 0.
    dms = []
    while dm<dm2:

        n2 = nchan**2.
        alp = 1./(16.+n2)
        bet = tsamp**2.
        dm = n2*alp*dm_prev + np.sqrt(16.*alp*(tol**2.-n2*alp)*dm_prev**2.+16.*alp*bet*(tol**2.-1.)*(nu**3./8.3/B)**2.)
        dm_prev = dm
        ndms += 1
        #print(dm)
        dms.append(dm)

    #print('DM trials:',ndms)
    return dms
minDM = 171
maxDM = 4000
DM_trials = np.array(gen_dm(minDM,maxDM,1.5,fc*1e-3,nchans,tsamp,chanbw))#[5:17]
DM_trials = np.concatenate([[0],DM_trials])
nDMtrials = len(DM_trials)

#snr threshold
SNRthresh = 6


#antenna positions
import csv
ANTENNALONS = []
ANTENNALATS = []
ANTENNAELEVS = []
with open('../DSA110_Station_Coordinates.csv','r') as csvfile:
    rdr = csv.reader(csvfile,delimiter=',')
    i = 0
    for row in rdr:
        print(row)
        if row[1][:3] == 'DSA' and row[1] != 'DSA-110 Station Coordinates':
            ANTENNALATS.append(float(row[2]))
            ANTENNALONS.append(float(row[3]))
            if row[4] == '':
                ANTENNAELEVS.append(np.nan)
            else:
                ANTENNAELEVS.append(float(row[4]))
csvfile.close()

ANTENNALATS = np.array(ANTENNALATS)
ANTENNALONS = np.array(ANTENNALONS)
ANTENNAELEVS = np.array(ANTENNAELEVS)

"""Search functions"""

#Create psf of desired resolution using antenna positions
#Note this is a rather crude implementation, it assumes 2D, but
#Nikita can get better PSFs
def make_2D_PSF(gridsize=gridsize,ANTENNALONS=ANTENNALONS,ANTENNALATS=ANTENNALATS,plot=False):
    if plot:
        plt.figure(figsize=(12,12))
        plt.scatter(ANTENNALATS,ANTENNALONS,c=ANTENNAELEVS)
        plt.plot((np.mean(ANTENNALATS)),(np.mean(ANTENNALONS)),'x',color='red')
        plt.show()

    centerlat = np.mean(ANTENNALATS)
    centerlon = np.mean(ANTENNALONS)

    Rearth = 6378
    ANTENNAx = Rearth*(ANTENNALONS-centerlon)*np.pi/180
    ANTENNAy = Rearth*(ANTENNALATS-centerlat)*np.pi/180

    if plot:
        plt.figure(figsize=(12,12))
        plt.scatter(ANTENNAx,ANTENNAy,c=ANTENNAELEVS)
        plt.plot(0,0,'x',color='red')
        plt.show()

    U_base = []#np.zeros(Nbase)
    V_base = []#np.zeros(Nbase)
    k = 0
    for i in range(len(ANTENNAx)):
        for j in range(len(ANTENNAy)):
            if i < j:
                U_base.append((ANTENNAx[j] - ANTENNAx[i])*(1e3)/lambdac)
                V_base.append((ANTENNAy[j] - ANTENNAy[i])*(1e3)/lambdac)
                #k += 1
    U_base = np.array(U_base)
    V_base = np.array(V_base)
    baselengths = np.sqrt(U_base**2 +V_base**2)

    if plot:
        plt.figure(figsize=(12,12))
        plt.scatter(U_base,V_base,c=baselengths)
        plt.show()

    #gridsize = 256
    U_grid = np.linspace(-10000,10000,gridsize)
    V_grid = np.linspace(-10000,10000,gridsize)

    Vij_grid = np.zeros((gridsize,gridsize),dtype=float)

    Ulocs = []
    Vlocs = []
    Vislocs = []



    for i in range(len(U_base)):
        U = U_base[i]
        V = V_base[i] 
        uidx = np.argmin(np.abs(U_grid-U))
        vidx = np.argmin(np.abs(V_grid-V))
        #print(U,V,U_grid[uidx],V_grid[vidx])

        Vij_grid[uidx,vidx] += 1 #Vij_base[i]

        Ulocs.append(uidx)
        Vlocs.append(vidx)
        Vislocs.append(1)#(Vij_base[i])


    Ulocs = np.array(Ulocs,dtype=int)
    Vlocs = np.array(Vlocs,dtype=int)
    Vislocs = np.array(Vislocs)

    if plot:
        plt.figure(figsize=(12,12))
        plt.imshow(Vij_grid,aspect='auto',vmin=0,vmax=1)
        plt.colorbar()
        plt.show()
        
    ANTENNAPSF = np.fft.fft2(Vij_grid)
    ANTENNAPSF = np.roll(np.abs(ANTENNAPSF),gridsize//2,axis=(0,1))
    if plot:
        plt.figure(figsize=(12,12))
        plt.imshow(ANTENNAPSF,aspect='auto')
        plt.colorbar()
        plt.show()

    return ANTENNAPSF


#snr calculation code
def snr_vs_RA_DEC(image_tesseract,boxcar,gridsize,plot=False,width=1,TMPCOORDS=[0,0]):
    #plot=False
    
    #step 1: average over frequency
    image_tesseract_freq = np.nanmean(image_tesseract,axis=3)#.mean(3)
    print("off pulse noise: " + str(np.mean(np.std(image_tesseract[:,:,10:],axis=2))*np.sqrt(np.sum(boxcar**2))/np.sum(boxcar)))
    if plot:
        plt.figure(figsize=(12,6))
    #plt.plot(boxcar)
    image_tesseract_time = np.zeros(image_tesseract_freq.shape[:2])
    for i in range(image_tesseract_time.shape[0]):
        for j in range(image_tesseract_time.shape[1]):


            #noiseest = np.sqrt(np.sum(boxcar**2))/np.sum(boxcar)
            #print(noiseest)
            signal_time = np.convolve(image_tesseract_freq[i,j,:],boxcar,'same')/np.sum(boxcar)
            noise = np.std(np.concatenate([signal_time[:np.argmax(signal_time)-width*3],signal_time[np.argmax(signal_time)+width*3:]]))
            signal = np.nanmax(signal_time)
            if i == TMPCOORDS[0] and j == TMPCOORDS[1]:
                print("sig: ", signal)
                print("noise: ",noise)
            image_tesseract_time[i,j] = signal/noise#np.nanmax(np.convolve(image_tesseract_freq[i,j,:],boxcar,'same')/np.sum(boxcar))/noiseest#/np.sum(boxcar))
            #print(boxcar)
            if plot:
                plt.plot(np.convolve(image_tesseract_freq[i,j,:],boxcar,'same'))
                
    if plot:
        plt.show()
    #step 3: smooth with PSF kernel
    #image_tesseract_snr = np.sqrt(convolve2d(image_tesseract_time,PSF,mode='same',boundary='fill',fillvalue=0))#/initnoise)
    image_tesseract_snr = image_tesseract_time
    if plot:
        
        fig, ax = plt.subplots(figsize=(12,12),subplot_kw={"projection": "3d"})
        RA_axis_2D, DEC_axis_2D = np.meshgrid(np.arange(gridsize), np.arange(gridsize))
        surf = ax.plot_surface(RA_axis_2D, DEC_axis_2D, image_tesseract_snr, cmap='jet',
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
    return image_tesseract_snr


#run search pipeline with desired DM, width trial range; output candidates to a csv? pkl? txt?
#takes 4D cube (RA,DEC,TIME,FREQUENCY)
def run_search(image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,freq_axis=freq_axis,DM_trials=DM_trials,widthtrials=widthtrials,tsamp=tsamp,SNRthresh=SNRthresh,plot=False,off=10,widthmode="gaussian",PSF=None,offpnoise=0.3,verbose=False):
    
    
    time_axis_long = np.arange(image_tesseract.shape[2])*tsamp
    
    #get axis sizes
    gridsize = len(RA_axis)
    nsamps = len(time_axis_long)
    nchans = len(freq_axis)
    
    if plot:
        plt.figure(figsize=(12,12))
        plt.imshow(image_tesseract.mean((2,3)))
        plt.colorbar(label="S/N")
        plt.xlabel("RA")
        plt.ylabel("DEC")
        plt.savefig(output_dir + "input_image.png")
        plt.close() 
    #de-disperse all beams
    nDMtrials = len(DM_trials)
    print("Starting dedispersion with " + str(nDMtrials) + " trials")
    image_tesseract_dedisp = np.zeros((gridsize,gridsize,nsamps,nDMtrials))

    for i in range(gridsize):
        for j in range(gridsize):
            for d in range(nDMtrials):
                tdelays = DM_trials[d]*4.15*(((fmin*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
                tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
                tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
                tdelays_frac = tdelays/tsamp - tdelays_idx_low
            
                for k in range(nchans):
                    #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac)
                    arrlow =  np.pad(image_tesseract[i,j,:,k],((0,tdelays_idx_low[k])),mode="constant",constant_values=0)[tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                    arrhi =  np.pad(image_tesseract[i,j,:,k],((0,tdelays_idx_hi[k])),mode="constant",constant_values=0)[tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

                    image_tesseract_dedisp[i,j,:,d] += arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])
                
                
    #boxcar search over pulse width
    nwidths=len(widthtrials)
    print("Boxcar searching over width with " + str(nwidths) + " trials")
    image_tesseract_binned = np.zeros((gridsize,gridsize,nwidths,nDMtrials))

    for i in range(nwidths):
        if plot:
            plt.figure()
        if widthmode == "boxcar":
            boxcar = np.zeros(nsamps)
            if widthtrials[i] == 1:
                minidx = int(nsamps//2)
                maxidx = minidx + 1
            else:
                minidx = np.max([int(np.floor(nsamps//2 - widthtrials[i]//2)),0])
                maxidx = np.min([int(minidx+widthtrials[i]),nsamps])
                boxcar = np.zeros(nsamps)
            boxcar[minidx:maxidx] = 1
        elif widthmode == "gaussian":
            boxcar = norm.pdf(np.arange(nsamps),loc=nsamps/2,scale=widthtrials[i]/2)
            #boxcar = boxcar*len(boxcar)/np.sum(boxcar)
        """
        print("Off-pulse noise estimate: " + str(np.sqrt(np.sum(boxcar**3))/np.sum(boxcar)))
        for j1 in range(gridsize):
            for j2 in range(gridsize):
                
                
                for k in range(nDMtrials):
                    noiseest = np.sqrt(np.sum(boxcar**2))/np.sum(boxcar)
                    sigest = np.max(np.convolve(image_tesseract_dedisp[j1,j2,:,k],boxcar,mode="same"))/np.sum(boxcar)#/np.sum(boxcar))
                    image_tesseract_binned[j1,j2,i,k] = sigest/noiseest#np.max(convolve(image_tesseract_dedisp[j1,j2,:,k],boxcar,mode="constant"))/np.sqrt(np.sum(boxcar))
        """
        for k in range(nDMtrials):
            image_tesseract_binned[:,:,i,k] = snr_vs_RA_DEC(image_tesseract_dedisp[:,:,:,k:k+1],boxcar=boxcar,gridsize=gridsize,width=widthtrials[i])
        #print(image_tesseract_binned.shape)
        #print(image_tesseract_binned)
        
        if plot:
            plt.show()

    #smooth image w/ PSF, scale to maximum
    if ~np.all(PSF == None):
        #initnoise = np.std(convolve2d(norm.rvs(size=(gridsize,gridsize),loc=0,scale=offpnoise),PSF,mode='same',boundary='fill',fillvalue=0))
        for i in range(nwidths):
            for j in range(nDMtrials):
                image_tesseract_binned[:,:,i,j] = np.sqrt(convolve2d(image_tesseract_binned[:,:,i,j],PSF,mode='same',boundary='fill',fillvalue=0))#/initnoise)
    else:
        print("skip smoothing")
    
    
    #threshold search for candidates
    print("Threshold searching for candidates with SNR > " + str(SNRthresh))
    
    #find indices
    flat_cands = np.arange(len(image_tesseract_binned.flatten()))[image_tesseract_binned.flatten()>SNRthresh]
    unravel_cands = []
    for i in range(len(flat_cands)):
        unravel_cands.append(np.unravel_index(flat_cands[i],image_tesseract_binned.shape))
    unique_spots = []
    for i in range(len(unravel_cands)):
        if unravel_cands[i] not in unique_spots:
            unique_spots.append(unravel_cands[i])
    
            
    #write cands to a txt file and csv file
    fcsv = open(output_dir + "all_cluster_cands.csv","w")
    ftxt = open(output_dir + "init_cluster_cands.txt","w")
    fcsv2 = open(output_dir + "init_cluster_cands.csv","w")
    csvwriter = csv.writer(fcsv)
    csvwriter2 = csv.writer(fcsv2)
    cands = []
    cluster_cands = []
    k = 0
    for spot in unique_spots:
        print(spot)
        print(image_tesseract_binned[spot[0],spot[1],spot[2],spot[3]])
        #find DM and width indices
        dms_all = []
        wids_all = []
        snrs_all = []
        if verbose:
            print("Candidate " + str(k) +":")
            print("RA: " + str(np.around(RA_axis[spot[0]]*180/np.pi,2)) + " deg")
            print("DEC: " + str(np.around(DEC_axis[spot[1]]*180/np.pi,2)) + " deg")
        ftxt.write("Candidate " + str(k) +":\n")
        ftxt.write("RA: " + str(np.around(RA_axis[spot[0]]*180/np.pi,2)) + " deg\n")
        ftxt.write("DEC: " + str(np.around(DEC_axis[spot[1]]*180/np.pi,2)) + " deg\n")

        for i in range(len(unravel_cands)):
            if unravel_cands[i][0] == spot[0] and unravel_cands[i][1] == spot[1]:
                dms_all.append(unravel_cands[i][3])
                wids_all.append(unravel_cands[i][2])
                snrs_all.append(image_tesseract_binned[spot[0],spot[1],unravel_cands[i][2],unravel_cands[i][3]])
                cands.append((spot[0],spot[1],unravel_cands[i][2],unravel_cands[i][3],image_tesseract_binned[spot[0],spot[1],unravel_cands[i][2],unravel_cands[i][3]]))
                csvwriter.writerow((spot[0],spot[1],unravel_cands[i][2],unravel_cands[i][3],image_tesseract_binned[spot[0],spot[1],unravel_cands[i][2],unravel_cands[i][3]]))
        dms_all = np.array(dms_all)
        wids_all = np.array(wids_all)
        snrs_all = np.array(snrs_all)

        #get width, dm with highest snr
        bestdmidx = dms_all[np.argmax(snrs_all)]
        bestwididx = wids_all[np.argmax(snrs_all)]
        bestsnr = np.max(snrs_all)
        cluster_cands.append((spot[0],spot[1],bestdmidx,bestwididx,bestsnr))
        csvwriter2.writerow((spot[0],spot[1],bestdmidx,bestwididx,bestsnr))
        if verbose:
            print("BEST SNR: " + str(np.around(bestsnr,2)) + "")
            print("BEST WIDTH: " + str(np.around(widthtrials[bestwididx]*tsamp,2)) + " ms")
            print("BEST DM: " + str(np.around(DM_trials[bestdmidx],2)) + " pc/cc")
            print("")
        ftxt.write("BEST SNR: " + str(np.around(bestsnr,2)) + "\n")
        ftxt.write("BEST WIDTH: " + str(np.around(widthtrials[bestwididx]*tsamp,2)) + " ms\n")
        ftxt.write("BEST DM: " + str(np.around(DM_trials[bestdmidx],2)) + " pc/cc\n")
        ftxt.write("\n")
        if plot:
            plt.figure(figsize=(24,12))
            plt.title("Candidate " + str(k) + ": (RA_offset=" +str(np.around(RA_axis[spot[0]]*180/np.pi,2)) + ",DEC_offset=" + str(np.around(DEC_axis[spot[1]]*180/np.pi,2))+")" )
            plt.scatter(tsamp*widthtrials[wids_all],DM_trials[dms_all],c=image_tesseract_binned[spot[0],spot[1],wids_all,dms_all],s=100)
            plt.plot(widthtrials[bestwididx]*tsamp,DM_trials[bestdmidx],'o',markerfacecolor="None",markeredgecolor="red",markersize=20)
            plt.colorbar(label="S/N")
            plt.xlabel("Width (ms)")
            plt.ylabel("DM (pc/cc)")
            #plt.yscale("log")
            plt.xscale("log")
            plt.xlim(0.1*tsamp,T*10)
            #plt.ylim(0.1,np.max(DM_trials)*10)
            plt.grid()
            plt.savefig(output_dir + "initial_clustering_plot_" + str(spot[0]) + "-" + str(spot[1]) + ".png")#plt.show()
            plt.close()
        k+=1
    if plot:
        # plot locations of found candidates
        image_spatial = image_tesseract.max(axis=(2,3))
        plt.figure(figsize=(12,12))
        plt.imshow(image_spatial)
        plt.xlabel("RA index")
        plt.xlabel("DEC index")
        plt.colorbar()
        plt.savefig(output_dir + "all_clusters.png")#plt.show()
        plt.close()
    ftxt.close()
    fcsv.close()
    fcsv2.close()
    print(str(len(cands)) + " total candidates")
    print(str(len(cluster_cands)) + " initial clusters")
    return cands,cluster_cands,image_tesseract_binned

#get cands and clusters from csv file
def read_cands(fname):
    cands = []
    with open(output_dir+fname,"r") as csvfile:
        rdr = csv.reader(csvfile,delimiter=',')
        for row in rdr:#.read_row():
            cands.append(row)
    csvfile.close()
    return cands

#make diagnostic plots
def search_plots(cands_gaussian,cluster_cands_gaussian,DM_trials=DM_trials,widthtrials=widthtrials,tsamp=tsamp,injected_cands_gaussian=None):

    if ~np.all(injected_cands_gaussian==None):
        injected_ras = []
        injected_decs = []
        injected_wids = []
        injected_dms = []
        injected_snrs = []
        for i in range(len(inject_cands_gaussian)):
            injected_ras.append(inject_cands_gaussian[i][0])
            injected_decs.append(inject_cands_gaussian[i][1])
            injected_wids.append(inject_cands_gaussian[i][3])
            injected_dms.append(inject_cands_gaussian[i][2])
            injected_snrs.append(inject_cands_gaussian[i][4])
        injected_ras = np.array(injected_ras)
        injected_decs = np.array(injected_decs)
        injected_wids = np.array(injected_wids)
        injected_dms = np.array(injected_dms)
        injected_snrs = np.array(injected_snrs)

    cand_ras = []
    cand_decs = []
    cand_wids = []
    cand_dms = []
    cand_snrs = []
    for i in range(len(cands_gaussian)):
        if ~np.isinf(cands_gaussian[i][4]):
            cand_ras.append(cands_gaussian[i][0])
            cand_decs.append(cands_gaussian[i][1])
            cand_wids.append(cands_gaussian[i][2])
            cand_dms.append(cands_gaussian[i][3])
            cand_snrs.append(cands_gaussian[i][4])
    cand_ras = np.array(cand_ras)
    cand_decs = np.array(cand_decs)
    cand_wids = np.array(cand_wids)
    cand_dms = np.array(cand_dms)
    cand_snrs = np.array(cand_snrs)

    cluster_cand_ras = []
    cluster_cand_decs = []
    cluster_cand_wids = []
    cluster_cand_dms = []
    cluster_cand_snrs = []
    for i in range(len(cluster_cands_gaussian)):
        if ~np.isinf(cluster_cands_gaussian[i][4]):
            cluster_cand_ras.append(cluster_cands_gaussian[i][0])
            cluster_cand_decs.append(cluster_cands_gaussian[i][1])
            cluster_cand_wids.append(cluster_cands_gaussian[i][3])
            cluster_cand_dms.append(cluster_cands_gaussian[i][2])
            cluster_cand_snrs.append(cluster_cands_gaussian[i][4])
    cluster_cand_ras = np.array(cluster_cand_ras)
    cluster_cand_decs = np.array(cluster_cand_decs)
    cluster_cand_wids = np.array(cluster_cand_wids)
    cluster_cand_dms = np.array(cluster_cand_dms)
    cluster_cand_snrs = np.array(cluster_cand_snrs)


    plt.figure(figsize=(12,12))
    if ~np.all(injected_cands_gaussian==None):
        plt.plot(injected_dms,injected_wids*tsamp,'o',markersize=25,alpha=0.5,color='blue',label='injected')
    plt.plot(DM_trials[cand_dms],widthtrials[cand_wids]*tsamp,'o',markersize=15,alpha=0.5,color='red',label='candidates')
    plt.plot(DM_trials[cluster_cand_dms],widthtrials[cluster_cand_wids]*tsamp,'o',markersize=5,alpha=0.5,color='green',label='clusters')
    plt.ylabel("Width (ms)")
    plt.xlabel("DM (pc/cc)")
    plt.legend(loc="upper left")
    for w in widthtrials:
        plt.axhline(tsamp*w)
    for dm in DM_trials:
        plt.axvline(dm)
    plt.savefig(output_dir + "diagnostic_DM_width.png")
    plt.close()

    plt.figure(figsize=(12,12))
    if ~np.all(injected_cands_gaussian==None):
        plt.scatter(RA_axis[injected_ras],DEC_axis[injected_decs],marker='o',s=400,alpha=injected_snrs/np.max(injected_snrs),c='blue',label='injected')
    plt.scatter(RA_axis[cand_ras],DEC_axis[cand_decs],marker='o',s=150,alpha=cand_snrs/np.max(cand_snrs),c='red',label='candidates')
    plt.scatter(RA_axis[cluster_cand_ras],DEC_axis[cluster_cand_decs],marker='o',s=50,alpha=cluster_cand_snrs/np.max(cluster_cand_snrs),c='green',label='clusters')
    plt.ylabel("RA (deg)")
    plt.xlabel("DEC (deg)")
    plt.legend(loc="upper left")
    for ra in RA_axis:
        plt.axhline(ra)
    for dec in DEC_axis:
        plt.axvline(dec)
    plt.savefig(output_dir + "diagnostic_RA_DEC.png")
    plt.close()
    return

import pickle as pkl
#function to update the currently stored noisemap
def update_noisemap(image_tesseract,noisemap_file="/dataz/dsa110/imaging/NSFRB_storage/NSFRB_noisemaps/noisemap_256pix.npy",noisemap_info_file="/dataz/dsa110/imaging/NSFRB_storage/NSFRB_noisemaps/noisemap_256pix_info.pkl"):
    #get previous noise map
    noisemap = np.load(noisemap_file)

    #get info from noise map
    f = open(noisemap_info_file,'rb')
    noiseinfo = pkl.load(f)
    f.close()

    #get standard deviation of new image
    imagenoisemap = image_tesseract.std(2)
    imagenoisemap = np.expand_dims(imagenoisemap,axis=2)
    
    if noiseinfo['iterations'] == 0:#np.all(noisemap == -1):
        #if all -1, replace noise map
        newnoisemap = imagenoisemap
    else:
        #otherwise, average together standard deviations
        newnoisemap = (noisemap*noiseinfo['iterations'] + imagenoisemap)/(noiseinfo['iterations'] + 1)

    #increase iterations
    noiseinfo['iterations'] += 1

    #write to file
    np.save(noisemap_file,newnoisemap)
    f = open(noisemap_info_file,'wb')
    pkl.dump(noiseinfo,f)
    f.close()
    return newnoisemap


#normalize using noisemap
def normalize_image(image_tesseract,noisemap_file="/dataz/dsa110/imaging/NSFRB_storage/NSFRB_noisemaps/noisemap_256pix.npy",noisemap_info_file="/dataz/dsa110/imaging/NSFRB_storage/NSFRB_noisemaps/noisemap_256pix_info.pkl"):

    #get previous noise map
    noisemap = np.load(noisemap_file)

    #get info from noise map
    f = open(noisemap_info_file,'rb')
    noiseinfo = pkl.load(f)
    f.close()

    if noiseinfo['iterations']==0:
        print("Noise map not initialized")
        return image_tesseract
    else:
        return image_tesseract/noisemap


#code to cutout subimages
def get_subimage(image_tesseract,ra_idx,dec_idx,dm=-1,freq_axis=freq_axis,tsamp=130,subimgpix=11,save=False,prefix="candidate_stamp",plot=False):
    gridsize = image_tesseract.shape[0]
    fname = "/dataz/dsa110/imaging/NSFRB_storage/NSFRB_candidates/" + prefix + "_" + str(ra_idx) + "_" + str(dec_idx)
    if subimgpix%2 == 0:
        print("subimgpix must be odd")
        return None



    #dedisperse if given a dm
    if dm != -1:
        fname = fname + "_dedisp" + str(dm) + ".npy"
        image_tesseract_dm = copy.deepcopy(image_tesseract)
        for i in range(gridsize):
            for j in range(gridsize):
                tdelays = dm*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
                tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
                tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
                tdelays_frac = tdelays/tsamp - tdelays_idx_low

                for k in range(nchans):
                    #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac)
                    arrlow =  np.pad(image_tesseract[i,j,:,k],((0,tdelays_idx_low[k])),mode="constant",constant_values=0)[tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                    arrhi =  np.pad(image_tesseract[i,j,:,k],((0,tdelays_idx_hi[k])),mode="constant",constant_values=0)[tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

                    image_tesseract_dm[i,j,:,k] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])

    else:
        fname = fname + ".npy"
        image_tesseract_dm = copy.deepcopy(image_tesseract)

    #pad w/ nans
    image_tesseract_dm = np.pad(image_tesseract_dm,((gridsize,gridsize),
                                                   (gridsize,gridsize),
                                                   (0,0),
                                                   (0,0)),
                                                   mode='constant',
                                                   constant_values=np.nan)

    #cut out subimage
    minraidx = gridsize + ra_idx - subimgpix//2#np.max([ra_idx - subimgpix//2,0])
    maxraidx = gridsize + ra_idx + subimgpix//2#np.min([ra_idx + subimgpix//2 + 1,gridsize-1])
    mindecidx = gridsize + dec_idx - subimgpix//2#np.max([dec_idx - subimgpix//2,0])
    maxdecidx = gridsize + dec_idx + subimgpix//2#np.min([dec_idx + subimgpix//2 + 1,gridsize-1])

    #print(minraidx_cut,maxraidx_cut,mindecidx_cut,maxdecidx_cut)
    print(minraidx,maxraidx,mindecidx,maxdecidx)

    image_cutout = image_tesseract_dm[minraidx:maxraidx,mindecidx:maxdecidx,:,:]

    if save:
        np.save(fname,image_cutout)

    if plot:
        plt.figure(figsize=(12,12))
        plt.imshow(image_cutout.mean((2,3)),aspect='auto')
        plt.show()
    return image_cutout





#hdbscan clustering function; clusters in DM, width, RA, DEC space
"""
import hdbscan
def hdbscan_cluster(cands,min_cluster_size=50,gridsize=gridsize,nDMtrials=nDMtrials,nwidths=nwidths,plot=False):

    print(str(len(cands)) + " candidates")

    #make list for each param
    raidxs = []
    decidxs = []
    dmidxs = []
    widthidxs = []
    snridxs = []
    for i in range(len(cands)):
        raidxs.append(cands[i][0])
        decidxs.append(cands[i][1])
        dmidxs.append(cands[i][3])
        widthidxs.append(cands[i][2])
        snridxs.append(cands[i][4])
    raidxs = np.array(raidxs)
    decidxs = np.array(decidxs)
    dmidxs = np.array(dmidxs)
    widthidxs = np.array(widthidxs)
    snridxs = np.array(snridxs)

    test_data=np.array([raidxs,decidxs,dmidxs,widthidxs]).transpose()


    #create clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)

    #cluster data
    clusterer.fit(test_data)


    #print number of noise points
    noisepoints = np.sum(clusterer.labels_==-1)
    print(str(noisepoints) + " noise points")

    nclasses = len(np.unique(clusterer.labels_))
    classnames = np.unique(clusterer.labels_)
    classes = clusterer.labels_
    if -1 in clusterer.labels_:
        nclasses -= 1

    print(str(nclasses) + " unique classes")


    #get centroids
    fcsv = open(output_dir + "hdbscan_cluster_cands.csv","w")
    csvwriter = csv.writer(fcsv)
    centroid_ras = []
    centroid_decs = []
    centroid_dms = []
    centroid_widths = []
    centroid_snrs = []
    for k in classnames:
        if k != -1:
            centroid_ras.append(int(np.sum((snridxs*raidxs)[classes==k])/np.sum(snridxs[classes==k])))
            centroid_decs.append(int(np.sum((snridxs*decidxs)[classes==k])/np.sum(snridxs[classes==k])))
            centroid_dms.append(int(np.sum((snridxs*dmidxs)[classes==k])/np.sum(snridxs[classes==k])))
            centroid_widths.append(int(np.sum((snridxs*widthidxs)[classes==k])/np.sum(snridxs[classes==k])))
            centroid_snrs.append(np.sum((snridxs*snridxs)[classes==k])/np.sum(snridxs[classes==k]))
            csvwriter.writerow([centroid_ras[-1],centroid_decs[-1],centroid_widths[-1],centroid_dms[-1],centroid_snrs[-1]])            
    fcsv.close()
    centroid_ras = np.array(centroid_ras)
    centroid_decs = np.array(centroid_decs)
    centroid_dms = np.array(centroid_dms)
    centroid_widths = np.array(centroid_widths)
    centroid_snrs = np.array(centroid_snrs)


    if plot:
        plt.figure(figsize=(24,24))
        ax = plt.subplot(projection="3d")
        for k in classnames:
            if k != -1:
                c=ax.scatter(raidxs[classes==k],decidxs[classes==k],dmidxs[classes==k],s=100*(2**widthidxs[classes==k]),marker='o',alpha=0.1)
                ax.scatter(centroid_ras[k],centroid_decs[k],centroid_dms[k],s=100*(2**centroid_widths[k]),marker='v',color=c.get_facecolor(),alpha=1)
        if noisepoints > 0:
            c=ax.scatter(raidxs[classes==-1],decidxs[classes==-1],dmidxs[classes==-1],s=100*(2**widthidxs[classes==-1]),marker='o',alpha=0.1,color='grey')

        plt.savefig(outdir + "hdbscan_cluster_plot.png")
        plt.close()


    return classes,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs
"""
