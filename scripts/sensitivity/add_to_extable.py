import json
import numpy as np
from nsfrb.config import table_dir
import argparse

def main(args):


    f = open(table_dir+"/NSFRB_excludecal.json","r")
    tab = json.load(f)
    f.close()


    print(args)
    if 'NVSS' in args.name:
        if args.remove:
            idxs = np.arange(len(tab['NVSS_exclude']),dtype=int)[np.array(tab['NVSS_exclude'])==args.name]
            while len(idxs)>0:
                for idx in idxs:
                     print(tab['NVSS_exclude'].pop(idx))
                     print(tab['NVSS_reason'].pop(idx))
                     print(tab['NVSS_MJD'].pop(idx))
                idxs = np.arange(len(tab['NVSS_exclude']),dtype=int)[np.array(tab['NVSS_exclude'])==args.name]
            print("Done, removed " + args.name)
        else:
            tab['NVSS_exclude'].append(args.name)
            tab['NVSS_reason'].append(args.reason)
            tab['NVSS_MJD'].append(args.mjd)
    elif 'RFC' in args.name:
        if args.remove:
            idxs = np.arange(len(tab['RFC_exclude']),dtype=int)[np.array(tab['RFC_exclude'])==args.name]
            while len(idxs)>0:
                for idx in idxs:
                     print(tab['RFC_exclude'].pop(idx))
                     print(tab['RFC_reason'].pop(idx))
                     print(tab['RFC_MJD'].pop(idx))
                idxs = np.arange(len(tab['RFC_exclude']),dtype=int)[np.array(tab['RFC_exclude'])==args.name]
            print("Done, removed " + args.name)
        else:
            tab['RFC_exclude'].append(args.name)
            tab['RFC_reason'].append(args.reason)
            tab['RFC_MJD'].append(args.mjd)
    else:
        print("source must be from NVSS or RFC catalog")
        return 1

    
    f = open(table_dir+"/NSFRB_excludecal.json","w")
    json.dump(tab,f)
    f.close()

    return 0


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Add an NVSS or RFC source to exclusion table")
    parser.add_argument('name')
    parser.add_argument('--reason',type=str,help='Reason to exclude the source (e.g. \'RFI\',\'Bright source not detected\')',default="")
    parser.add_argument('--mjd',type=float,help='MJD of specific pass to exclude, not required if excluding all passes',default=-1)
    parser.add_argument('--remove',action='store_true',help='Remove source')
    args = parser.parse_args()
    main(args)

