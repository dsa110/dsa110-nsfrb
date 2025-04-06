import os
from pathlib import Path
import datetime
import argparse
import sys
from astropy.time import Time
import csv
import time
import glob

"""
This script waits for candidates to pass their 1-day expiration, then deletes them
"""

cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/"
def main(args):
    while True:
        #find any rsynced one day ago
        Tnow = Time.now()
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.waittime)
        print(
            f"Removing operation files last modified prior to "
            f"{cutoff.strftime('%Y-%m-%dT%H:%M:%S')} UTC")


        #get list of files
        raw_cands = glob.glob(cand_dir + "raw_cands/candidates_*csv")
        #get isots
        raw_times = [(Time(rc[-32:-9],format='isot') if 'slow' in rc else Time(rc[-27:-4],format='isot')) for rc in raw_cands]
        i = 0
        for file in (Path(cand_dir) / "raw_cands/").glob("candidates_*csv"):
            print(file)
            rt = raw_times[i]
            modtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            # modtime is timezone naive, so we set it to utc
            # lxc managed containers are all using utc
            modtime = modtime.replace(tzinfo=cutoff.tzinfo)
            if modtime < cutoff:
                print(modtime,cutoff)
                print(f'Removing {file}')
                print("Deleting ",rt.isot,"raw candidate files")
                os.system("rm " + cand_dir + "raw_cands/*" + rt.isot + "*")
                os.system("rm " + cand_dir + "backup_raw_cands/*" + rt.isot + "*")

            i += 1
        """
        #get isots
        raw_times = [(Time(rc[-32:-9],format='isot') if 'slow' in rc else Time(rc[-27:-4],format='isot')) for rc in raw_cands]

        #delete the expired ones
        for rt in raw_times:
            dt = (Tnow - rt).datetime.days
            if dt >= args.waittime:
                print("Deleting ",rt.isot,"raw candidate files")
                os.system("rm " + cand_dir + "raw_cands/*" + rt.isot + "*")
                os.system("rm " + cand_dir + "backup_raw_cands/*" + rt.isot + "*")
        """

        #injections
        injections = glob.glob(cand_dir + "final_cands/injections/*/*csv")
        print(injections)
        #get isots
        inj_times = [Time(rc[rc.index("/injections/") + len("/injections/"):rc.index("/injections/") + len("/injections/") + 23],format='isot') for rc in injections]
        i = 0
        for file in (Path(cand_dir) / "final_cands/injections/").glob("*/*csv"):
            print(file)
            rt = inj_times[i]
            modtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            # modtime is timezone naive, so we set it to utc
            # lxc managed containers are all using utc
            modtime = modtime.replace(tzinfo=cutoff.tzinfo)
            if modtime < cutoff:
                print(modtime,cutoff)
                print(f'Removing {file}')
                print("Deleting ",rt.isot,"raw candidate files")
                os.system("rm -r " + cand_dir + "final_cands/injections/*" + rt.isot + "*")
            i += 1
        """        
        #same for injections
        injections = glob.glob(cand_dir + "final_cands/injections/*")
        inj_times = [Time(rc[rc.index("/injections/") + len("/injections/"):rc.index("/injections/") + len("/injections/") + 23],format='isot') for rc in injections]
        for rt in inj_times:
            dt = (Tnow - rt).datetime.days
            if dt >= args.waittime:
                print("Deleting ",rt.isot,"raw candidate files")
                os.system("rm -r " + cand_dir + "final_cands/injections/*" + rt.isot + "*")
        """
        #final cands
        cands = glob.glob(cand_dir + "final_cands/candidates/*/*csv")
        print(cands)
        #get isots
        cand_times = [Time(rc[rc.index("/candidates/") + len("/candidates/"):rc.index("/candidates/") + len("/candidates/") + 23],format='isot') for rc in cands]
        keepcands = []
        with open(cand_dir + "final_cands/cands_for_followup_isot.csv","r") as csvfile:
            rdr = csv.reader(csvfile)
            for row in rdr:
                keepcands.append(row[0])
        i =0
        for file in (Path(cand_dir) / "final_cands/candidates/").glob("*/*csv"):
            print(file)
            rt = cand_times[i]
            modtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            # modtime is timezone naive, so we set it to utc
            # lxc managed containers are all using utc
            modtime = modtime.replace(tzinfo=cutoff.tzinfo)
            if (rt.isot not in keepcands) and modtime < cutoff:
                print(modtime,cutoff)
                print(f'Removing {file}')
                print("Deleting ",rt.isot,"raw candidate files")
                os.system("rm -r " + cand_dir + "final_cands/candidates/*" + rt.isot + "*")
            i += 1
        """
        #same for candidates
        cands = glob.glob(cand_dir + "final_cands/candidates/*")
        cand_times = [Time(rc[rc.index("/candidates/") + len("/candidates/"):rc.index("/candidates/") + len("/candidates/") + 23],format='isot') for rc in cands]
        keepcands = []
        with open(cand_dir + "final_cands/cands_for_followup_isot.csv","r") as csvfile:
            rdr = csv.reader(csvfile)
            for row in rdr:
                keepcands.append(row[0])
        print(keepcands)
        for rt in cand_times:
            dt = (Tnow - rt).datetime.days
            if (rt.isot not in keepcands) and dt >= args.waittime:
                
                print("Deleting ",rt.isot,"raw candidate files")
                os.system("rm -r " + cand_dir + "final_cands/candidates/*" + rt.isot + "*")
        """

        time.sleep(args.cadence*3600)
if __name__=="__main__":
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--waittime',type=float,help='Time between clearing candidates in days, default 1',default=1.0)
    parser.add_argument('--cadence',type=float,help='Time between checking for new candidates to clear in hours, default 2',default=2.0)
    args = parser.parse_args()


    main(args)
