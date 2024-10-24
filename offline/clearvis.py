import os
import argparse
import sys
from astropy.time import Time
import csv
import time

"""
This script waits for visibilities to pass their 1-day expiration, then deletes them
"""

vis_file = os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/vis_files.csv"

def main(args):
    while True:
        #find any rsynced one day ago
        Tnow = Time.now()

        #read vis files
        delidx = []
        with open(vis_file,"r") as csvfile:
            rdr = csv.reader(csvfile,delimiter=",")
            i = 0
            for row in rdr:
                fname = row[1]
                dt = (Tnow - Time(row[0],format='isot')).datetime.days
                if dt >= 1:
                    print("Deleting ",fname)
                    delidx.append(str(i+1))
                    os.system("rm " + str(fname))

                i += 1
        if len(delidx)>1:
            print("sed -i.bak -e '" + "d;".join(delidx) + "d' " + vis_file)
            os.system("sed -i.bak -e '" + "d;".join(delidx) + "d' " + vis_file)
        elif len(delidx) == 1:
            print("sed -i.bak -e '" + delidx[0] + "d' " + vis_file)
            os.system("sed -i.bak -e '" + delidx[0] + "d' " + vis_file)
        time.sleep(args.waittime*3600)
if __name__=="__main__":
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--waittime',type=float,help='Time between clearing visibilities in hours, default 1',default=1.0)
    args = parser.parse_args()


    main(args)
