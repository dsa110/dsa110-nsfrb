import numpy as np
import pickle as pkl
import sys
import os
import glob

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()
sys.path.append(cwd + "/")
from nsfrb.config import *


#noise directory
noise_dir = cwd + "-noise/" 
output_file = cwd + "-logfiles/search_log.txt"


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


def noise_update_all(noise,gridsize_RA,gridsize_DEC,DM_trials,widthtrials,noise_dir=noise_dir,output_file=output_file):
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

    print("INPUT NOISE MEDIAN:" + str(noise),file=fout)
    noise_final = np.zeros((len(widthtrials),len(DM_trials)))
    for i in range(len(DM_trials)):
        DM = DM_trials[i]
        for j in range(len(widthtrials)):
            width = widthtrials[j]
            if (DM not in noise_dict.keys()) or (width not in noise_dict[DM].keys()):
                if DM not in noise_dict.keys():
                    noise_dict[DM] = dict()
                noise_dict[DM][width] = [1, noise[j,i]]

            else:
                prevN, prevnoise = noise_dict[DM][width]
                nextN = prevN + 1
                nextnoise = (prevnoise*prevN + noise[j,i])/nextN
                noise_dict[DM][width] = [nextN, nextnoise]
            noise_final[j,i] = noise_dict[DM][width][1]
    print("OUTPUT_NOISE MEDIAN:" + str(noise_final),file=fout)
    f = open(fname,"wb")
    pkl.dump(noise_dict,f)
    f.close()

    if output_file != "":
        fout.close()
    return noise_final 

def init_noise(noise_dir=noise_dir):
    if len(glob.glob(noise_dir + "/*pkl")) > 0:
        os.system("rm " + noise_dir + "/*pkl")
    return
