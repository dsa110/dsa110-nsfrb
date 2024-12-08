import os
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

        #get list of files
        raw_cands = glob.glob(cand_dir + "raw_cands/candidates_*csv")

        #get isots
        raw_times = [Time(rc[-27:-4],format='isot') for rc in raw_cands]

        #delete the expired ones
        for rt in raw_times:
            dt = (Tnow - rt).datetime.days
            if dt >= args.waittime:
                print("Deleting ",rt.isot,"raw candidate files")
                os.system("rm " + cand_dir + "raw_cands/*" + rt.isot + "*")
                os.system("rm " + cand_dir + "backup_raw_cands/*" + rt.isot + "*")

        time.sleep(args.cadence*3600)
if __name__=="__main__":
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--waittime',type=float,help='Time between clearing candidates in days, default 1',default=1.0)
    parser.add_argument('--cadence',type=float,help='Time between checking for new candidates to clear in hours, default 2',default=2.0)
    args = parser.parse_args()


    main(args)
