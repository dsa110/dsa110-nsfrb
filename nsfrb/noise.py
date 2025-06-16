import numpy as np
import pickle as pkl
import sys
import os
import glob


from nsfrb.config import *
"""
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
from nsfrb.config import *


#noise directory
noise_dir = cwd + "-noise/" 
output_file = cwd + "-logfiles/search_log.txt"
"""

def noise_update(noise,gridsize_RA,gridsize_DEC,DM,width,noise_dir=noise_dir,output_file=output_file):
    """
    This function retrieves and updates the running mean standard deviation 
    noise for a given DM and pulse width.
    """

    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #find noise pkl file
    fname = noise_dir + "noise_" + str(gridsize_RA) + "x" + str(gridsize_DEC) +".pkl"
    try:
        f = open(fname,"rb")
        noise_dict = pkl.load(f)
        f.close()
    except:
        print("Creating noise file " + fname + "...",file=fout)
        noise_dict = dict()

    #update entry for DM, width trial
    if (DM not in noise_dict.keys()) or (width not in noise_dict[DM].keys()):
        if DM not in noise_dict.keys():
            noise_dict[DM] = dict()    
        noise_dict[DM][width] = [1, noise]
    
    else:
        prevN, prevnoise = noise_dict[DM][width]
        nextN = prevN + 1
        nextnoise = (prevnoise*prevN + noise)/nextN
        noise_dict[DM][width] = [nextN, nextnoise]

    f = open(fname,"wb")
    pkl.dump(noise_dict,f)
    f.close()

    if output_file != "":
        fout.close()
    return noise_dict[DM][width]


def get_noise_dict(gridsize_RA,gridsize_DEC):
    #find noise pkl file
    fname = noise_dir + "noise_" + str(gridsize_RA) + "x" + str(gridsize_DEC) +".pkl"
    try:
        f = open(fname,"rb")
        noise_dict = pkl.load(f)
        f.close()
    except:
        print("Initializing to Empty Noise Dict",file=fout)#Creating noise file " + fname + "...",file=fout)
        noise_dict = dict()
    return noise_dict

def noise_update_all(noise,gridsize_RA,gridsize_DEC,DM_trials,widthtrials,noise_dir=noise_dir,output_file=output_file,writeonly=False,readonly=False,suff=""):
    """
    This function retrieves and updates the running mean standard deviation 
    noise for a given DM and pulse width.
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #find noise pkl file
    print(suff)
    fname = noise_dir + "noise_" + str(gridsize_RA) + "x" + str(gridsize_DEC) + str(suff) +".pkl"
    try:
        f = open(fname,"rb")
        noise_dict = pkl.load(f)
        f.close()
    except:
        print("Creating noise file " + fname + "...",file=fout)
        noise_dict = dict()
        
        if readonly:
            return np.zeros((len(widthtrials),len(DM_trials))),0
    #update entry for DM, width trial

    print("INPUT NOISE MEDIAN:" + str(noise),file=fout)
    noise_final = np.zeros((len(widthtrials),len(DM_trials)))
    for i in range(len(DM_trials)):
        DM = DM_trials[i]
        for j in range(len(widthtrials)):
            
            width = widthtrials[j]
            if not readonly and noise is not None and np.isnan(noise[j,i]):
                print("NOISE UPDATE IS NAN",file=fout)
            elif (DM not in noise_dict.keys()) or (width not in noise_dict[DM].keys()):
                if DM not in noise_dict.keys():
                    noise_dict[DM] = dict()
                if not readonly:
                    noise_dict[DM][width] = [1, noise[j,i]]
                else:
                    noise_dict[DM][width] = [0,0]#[1, np.nan]
            elif not writeonly and not readonly:
                prevN, prevnoise = noise_dict[DM][width]
                nextN = prevN + 1
                nextnoise = (prevnoise*prevN + noise[j,i])/nextN
                noise_dict[DM][width] = [nextN, nextnoise]
            elif writeonly: #writeonly set to true if noise has already been updated, so just increment the number and write the new noise
                prevN, prevnoise = noise_dict[DM][width]
                nextN = prevN + 1
                noise_dict[DM][width] = [nextN, noise[j,i]]
            elif readonly:
                prevN, prevnoise = noise_dict[DM][width]
            noise_final[j,i] = noise_dict[DM][width][1]
    print("OUTPUT_NOISE MEDIAN:" + str(noise_final),file=fout)
    f = open(fname,"wb")
    pkl.dump(noise_dict,f)
    f.close()

    if output_file != "":
        fout.close()
    if readonly:
        return noise_final, prevN
    return noise_final 

def init_noise(DM_trials,widthtrials,gridsize_RA,gridsize_DEC,noise_dir=noise_dir,img=False,suff="",zero=False):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    #remove non-initialization files
    noisefiles = glob.glob(noise_dir + "/*pkl")
    for n in noisefiles:
        if '_sim' not in n:
            os.system("rm " + n)
    
    #initialize
    if zero:
        print("initializing noise to zero",file=fout)
        all_noise = dict()
        for i in range(len(DM_trials)):
            all_noise[DM_trials[i]] = dict()
            for j in range(len(widthtrials)):
                all_noise[DM_trials[i]][widthtrials[j]] = [0,0.0]
        fname = noise_dir + "noise_" + str(gridsize_RA) + "x" + str(gridsize_DEC) + str(suff) +".pkl"
        noisefile = open(fname,"wb")
        pkl.dump(all_noise,noisefile)
        noisefile.close()

    elif img:
        noisefiles = glob.glob(noise_dir + "/*pkl")
        for n in noisefiles:
            if '_sim' in n:
                print("initializing with " + n,file=fout)
                os.system("cp " + n + " " + n[:n.index("_sim")] + ".pkl")
    else:
        #initialize based on fit relation
        vis_noise = np.mean(np.load(noise_dir + "raw_vis_noise_real.npy"))
        all_noise = dict()
        for i in range(len(DM_trials)):
            all_noise[DM_trials[i]] = dict()
            for j in range(len(widthtrials)):
                all_noise[DM_trials[i]][widthtrials[j]] = [1,vis_to_img_slope*vis_noise*np.sqrt(widthtrials[j])]
        fname = noise_dir + "noise_" + str(gridsize_RA) + "x" + str(gridsize_DEC) + str(suff) +".pkl"
        noisefile = open(fname,"wb")
        pkl.dump(all_noise,noisefile)
        noisefile.close()
        
    if output_file != "":
        fout.close()
    return


