import numpy as np
import pickle as pkl
import sys


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



