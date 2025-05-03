from pathlib import Path
import copy
from astropy.time import Time
import glob
import argparse
import csv
import datetime
import time
import os
import shutil
import numpy as np
from nsfrb.pipeline import read_raw_vis
"""
This script waits for visibilities to pass their 1-day expiration, then deletes them
"""


operations_dir = Path(os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/")
raw_cand_dir = Path(os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/raw_cands/")
cand_dir = Path(os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/final_cands/")
subdirs_to_clear_init = [
    ("lxd110h03","*.out"),
    ("lxd110h04","*.out"),
    ("lxd110h05","*.out"),
    ("lxd110h06","*.out"),
    ("lxd110h07","*.out"),
    ("lxd110h08","*.out"),
    ("lxd110h10","*.out"),
    ("lxd110h11","*.out"),
    ("lxd110h12","*.out"),
    ("lxd110h14","*.out"),
    ("lxd110h15","*.out"),
    ("lxd110h16","*.out"),
    ("lxd110h18","*.out"),
    ("lxd110h19","*.out"),
    ("lxd110h21","*.out"),
    ("lxd110h22","*.out")
    ]

vis_file = os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/vis_files.csv"
T4dir = os.environ['NSFRBT4'] + "candidates/"

def main(args):
    

    """
    if args.populate:
        with open(vis_file,"w") as csvfile:
            for subdir, pattern in subdirs_to_clear:
                for file in (operations_dir / subdir).glob(pattern):
                    wr = csv.writer(csvfile,delimiter=',')
                    wr.writerow([os.path.basename(str(file)),int(0),""])
        print("Populated csv, returning")
        return 0
    """
    subdirs_to_clear = copy.deepcopy(subdirs_to_clear_init)
    if args.populate:
        with open(vis_file,"w") as csvfile:
            for subdir, pattern in subdirs_to_clear:
                files = np.sort(glob.glob(os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/" + subdir + "/" + pattern))
                for f in files:
                    wr = csv.writer(csvfile,delimiter=',')
                    wr.writerow([os.path.basename(str(f)),int(0),""])
                    #print(os.path.basename(str(f)))
            
        print("Populated csv, returning")
        return 0


    while True:
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.waittime)
        print(
            f"Removing operation files last modified prior to "
            f"{cutoff.strftime('%Y-%m-%dT%H:%M:%S')} UTC")

        #clear candidate and injection stuff
        print("Clearing raw cands...")
        for file in (raw_cand_dir).glob("*"):
            #print(os.path.basename(str(file)),type(file))


            modtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            # modtime is timezone naive, so we set it to utc
            # lxc managed containers are all using utc
            modtime = modtime.replace(tzinfo=cutoff.tzinfo)
            if modtime < cutoff:
                print(modtime,cutoff)
                print(f'Removing {file}')

                try:
                    file.unlink()
                except Exception as exc:
                    print("File unlink failed:",exc)
                    shutil.rmtree(file)

        cand_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.backup_waittime)
        canddirs = list(glob.glob(str(cand_dir) + "/candidates/*")) + list(glob.glob(str(cand_dir) + "/injections/*"))
        keepcands = []
        with open(str(cand_dir) + "/cands_for_followup_isot.csv","r") as csvfile:
            rdr = csv.reader(csvfile)
            for row in rdr:
                keepcands.append(row[0])
        for cdir in canddirs:
            #first check if file in keepcands
            candisot = os.path.basename(cdir)[:23]
            if candisot in keepcands and 'injections' not in str(cdir):
                print("keepcand:",candisot)
                #check if it should be backed up
                modtime = datetime.datetime.fromtimestamp(Path(cdir).stat().st_mtime)
                modtime = modtime.replace(tzinfo=cand_cutoff.tzinfo)
                if modtime < cand_cutoff:
                    print(modtime,cand_cutoff)
                    print(f'Backing up '+str(cdir))
                    status = os.system("scp -r " + cdir + " user@dsa-storage.ovro.pvt:" + T4dir + "/")
                    if status != 0:
                        print(f'Error copying {cdir} to dsastorage')
                    else:
                        print(f'Removing ' + str(cdir))
                        shutil.rmtree(Path(cdir))
                        print("done")
            else:
                modtime = datetime.datetime.fromtimestamp(Path(cdir).stat().st_mtime)
                modtime = modtime.replace(tzinfo=cutoff.tzinfo)
                if modtime < cutoff:
                    print(modtime,cutoff)
                    print("Removiing " + str(cdir))
                    shutil.rmtree(Path(cdir))


        #clear nvss and RFC calibrator stuff
        if args.clearcal:
            #print("Clearing NVSS and RFC data...")
            caldirs = list(glob.glob(str(operations_dir) + "/NVSS*")) + list(glob.glob(str(operations_dir) + "/RFC*"))
            subdirs_to_clear += [(os.path.basename(c),"*.out") for c in caldirs]
            subdirs_to_clear += [(os.path.basename(c),"*.npy") for c in caldirs]
            clearcal_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.clearcal_waittime)
            
            """
            for c in caldirs:
                for file in Path(c).glob("*out"):
                    tmp,mjd,tmp = read_raw_vis(str(file),get_header=True)
                    if Time(mjd,format='mjd').datetime < cutoff2:
                        print("Deleting " + str(file))
                        try:
                            file.unlink()
                        except Exception as exc:
                            print("File unlink failed:",exc)
            print("Done")
            """

        print("SUBDIRS TO CLEAR:",subdirs_to_clear)

        #read vis files
        delidx_labels = dict()
        with open(vis_file,"r") as csvfile:
            rdr = csv.reader(csvfile,delimiter=",")
            i = 0
            for row in rdr:
                if "nsfrb_sb01_59484.out" in row[0]: print(row)
                delidx_labels[row[0]] = str(i+1)
                i+= 1
        delidx = []
        
        
        
        #first see if any have already been removed
        for n in delidx_labels.keys():
            #print(str(operations_dir) + "/*/" + str(n))
            if len(glob.glob(str(operations_dir) + "/*/" + str(n))) == 0:
                print(str(operations_dir) + "/*/" + str(n))
                delidx.append(delidx_labels[n])
        

        for subdir, pattern in subdirs_to_clear:
            for file in (operations_dir / subdir).glob(pattern):
                #print(os.path.basename(str(file)),type(file))
                

                modtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
                # modtime is timezone naive, so we set it to utc
                # lxc managed containers are all using utc
                if "NVSS" in str(subdir) or "RFC" in str(subdir):
                    modtime = modtime.replace(tzinfo=clearcal_cutoff.tzinfo)
                    if modtime < clearcal_cutoff:
                        print(modtime,clearcal_cutoff)
                        print(f'Removing {file}')

                        try:
                            file.unlink()
                        except Exception as exc:
                            print("File unlink failed:",exc)
                            shutil.rmtree(file)
                else:
                    modtime = modtime.replace(tzinfo=cutoff.tzinfo)
                    if modtime < cutoff:
                        print(modtime,cutoff)
                        print(f'Removing {file}')
                    
                        try:
                            file.unlink()
                        except Exception as exc:
                            print("File unlink failed:",exc)
                            shutil.rmtree(file)
                
                
                if "lxd110" in str(subdir) and os.path.basename(str(file)) in delidx_labels.keys():
                    delidx.append(delidx_labels[os.path.basename(str(file))])
                    
        #remove from csv
        if len(delidx)>1:
            print("sed -i.bak -e '" + "d;".join(delidx) + "d' " + vis_file)
            status = os.system("sed -i.bak -e '" + "d;".join(delidx) + "d' " + vis_file)
            if status != 0: print("Error updating vis_files.csv")
        elif len(delidx) == 1:
            print("sed -i.bak -e '" + delidx[0] + "d' " + vis_file)
            status = os.system("sed -i.bak -e '" + delidx[0] + "d' " + vis_file)
            if status != 0: print("Error updating vis_files.csv")
        
        
        time.sleep(args.cadence*3600)

if __name__=="__main__":
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--waittime',type=float,help='Time between clearing visibilities in days, default 1',default=1.0)
    parser.add_argument('--cadence',type=float,help='Time between checking for new vis to clear in hours, default 2',default=2.0)
    parser.add_argument('--populate',action='store_true',default=False,help="Don't clear vis, just re-populate the csv")
    parser.add_argument('--clearcal',action='store_true',default=False,help='Clear calibration data dirs (NVSS*, RFC*)')
    parser.add_argument('--clearcal_waittime',type=float,help='Time between clearing calibrator visibilities in days, default 1',default=1.0)
    parser.add_argument('--backup_waittime',type=float,help='Maximum time in days before candidate is deleted or backed up to dsastorage,default 14',default=14.0)
    args = parser.parse_args()

    main(args)
