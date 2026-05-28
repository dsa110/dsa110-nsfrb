import json
import os
from astropy.time import Time
import numpy as np
from nsfrb.config import table_dir
import argparse

def main(args):


    f = open(table_dir+"/NSFRB_excludecal.json","r")
    tab = json.load(f)
    f.close()
    for k in tab.keys():
        print(k,len(tab[k]))
    #return

    if len(args.clean)>0:
        os.system("cp "+table_dir+"/NSFRB_excludecal.json "+table_dir+"/NSFRB_excludecal_backup"+Time.now().isot+".json")
        mintime = Time(args.clean,format='isot')
        idxnvss = np.logical_or(tab['NVSS_MJD']==-1,tab['NVSS_MJD']>mintime.mjd)
        idxrfc = np.logical_or(tab['RFC_MJD']==-1,tab['RFC_MJD']>mintime.mjd)
        tab['NVSS_exclude'] = list(np.array(tab['NVSS_exclude'])[idxnvss])
        tab['NVSS_reason'] = list(np.array(tab['NVSS_reason'])[idxnvss])
        tab['NVSS_MJD'] = list(np.array(tab['NVSS_MJD'])[idxnvss])
        tab['RFC_exclude'] = list(np.array(tab['RFC_exclude'])[idxrfc])
        tab['RFC_reason'] = list(np.array(tab['RFC_reason'])[idxrfc])
        tab['RFC_MJD'] = list(np.array(tab['RFC_MJD'])[idxrfc])
        
        print(tab)
        print("done cleaning exclude cal before ",args.clean)

        f = open(table_dir+"/NSFRB_excludecal.json","w")
        json.dump(tab,f)
        f.close()

        f = open(table_dir+"/NSFRB_astrocal.json","r")
        tab = json.load(f)
        f.close()
        os.system("cp "+table_dir+"/NSFRB_astrocal.json "+table_dir+"/NSFRB_astrocal_backup"+Time.now().isot+".json")
        for k in tab.keys():
            if type(tab[k])==dict and len(tab[k].keys())>0:
                print(k,tab[k].keys())
                for jj in tab[k].keys():
                    delkeys = []
                    for j in tab[k][jj].keys():
                        print(j)
                        t1 = Time(str(j)[-27:-4],format='isot')
                        print(t1)
                        if t1<mintime:
                            print(k,jj,j,t1)
                            delkeys.append(j)
                    for j in delkeys:
                        del tab[k][jj][j]
        print("done cleaning astrocal before ",args.clean)

        f = open(table_dir+"/NSFRB_astrocal.json","w")
        json.dump(tab,f)
        f.close()

        f = open(table_dir+"/NSFRB_speccal.json","r")
        tab = json.load(f)
        f.close()
        #backup
        os.system("cp "+table_dir+"/NSFRB_speccal.json "+table_dir+"/NSFRB_speccal_backup"+Time.now().isot+".json")
        for k in tab.keys():
            if type(tab[k])==dict and len(tab[k].keys())>0:
                print(k,tab[k].keys())
                for jj in tab[k].keys():
                    delkeys = []
                    for j in tab[k][jj].keys():
                        print(j)
                        t1 = Time(str(j)[-27:-4],format='isot')
                        print(t1)
                        if t1<mintime:
                            print(k,jj,j,t1)
                            delkeys.append(j)
                    for j in delkeys:
                        del tab[k][jj][j]
        print("done cleaning speccal before ",args.clean)

        f = open(table_dir+"/NSFRB_speccal.json","w")
        json.dump(tab,f)
        f.close()

        return
    print(args)
    if 'NVSS' in args.name:
        if args.remove:
            idxs = np.arange(len(tab['NVSS_exclude']),dtype=int)[np.array(tab['NVSS_exclude'])==args.name]
            while len(idxs)>0:
                for idx in idxs[:1]:
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
                for idx in idxs[:1]:
                     print(tab['RFC_exclude'].pop(idx))
                     print(tab['RFC_reason'].pop(idx))
                     print(tab['RFC_MJD'].pop(idx))
                idxs = np.arange(len(tab['RFC_exclude']),dtype=int)[np.array(tab['RFC_exclude'])==args.name]
                print(idxs)
            print("Done, removed " + args.name)
        else:
            tab['RFC_exclude'].append(args.name)
            tab['RFC_reason'].append(args.reason)
            tab['RFC_MJD'].append(args.mjd)
    elif args.mjd != -1:

        if not args.remove:
            print("Excluding all observations from "+Time(args.mjd,format='mjd').isot)
            tab['NVSS_exclude'].append("ALL")
            tab['NVSS_reason'].append(args.reason)
            tab['NVSS_MJD'].append(args.mjd)
            tab['RFC_exclude'].append("ALL")
            tab['RFC_reason'].append(args.reason)
            tab['RFC_MJD'].append(args.mjd)
        else:
            print("Removing "+Time(args.mjd,format='mjd').isot+" from exclude table")
            idxs = np.arange(len(tab['NVSS_exclude']),dtype=int)[np.logical_and(np.array(tab['NVSS_exclude'])=="ALL",np.array(tab['NVSS_MJD'])==args.mjd)]
            while len(idxs)>0:
                for idx in idxs[:1]:
                     print(tab['NVSS_exclude'].pop(idx))
                     print(tab['NVSS_reason'].pop(idx))
                     print(tab['NVSS_MJD'].pop(idx))
                idxs = np.arange(len(tab['NVSS_exclude']),dtype=int)[np.array(tab['NVSS_exclude'])==args.name]
            print("Done, removed " + args.name)

            idxs = np.arange(len(tab['RFC_exclude']),dtype=int)[np.logical_and(np.array(tab['RFC_exclude'])=="ALL",np.array(tab['RFC_MJD'])==args.mjd)]
            while len(idxs)>0:
                for idx in idxs[:1]:
                     print(tab['RFC_exclude'].pop(idx))
                     print(tab['RFC_reason'].pop(idx))
                     print(tab['RFC_MJD'].pop(idx))
                idxs = np.arange(len(tab['RFC_exclude']),dtype=int)[np.array(tab['RFC_exclude'])==args.name]
            print("Done, removed " + args.name)

    else:
        print("source must be from NVSS or RFC catalog")
        return 1

    
    f = open(table_dir+"/NSFRB_excludecal.json","w")
    json.dump(tab,f)
    f.close()

    return 0


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Add an NVSS or RFC source to exclusion table")
    parser.add_argument('--name',type=str,help='Source name',default="")
    parser.add_argument('--reason',type=str,help='Reason to exclude the source (e.g. \'RFI\',\'Bright source not detected\')',default="")
    parser.add_argument('--mjd',type=float,help='MJD of specific pass to exclude, not required if excluding all passes',default=-1)
    parser.add_argument('--remove',action='store_true',help='Remove source')
    parser.add_argument('--clean',type=str,help='Removes from exclude, astrocal, speccal tables all data before specified ISOT',default='')
    args = parser.parse_args()
    main(args)

