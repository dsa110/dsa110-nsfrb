from pathlib import Path
import time
from astropy import units as u
from astropy.time import Time
import glob
import argparse
import csv
import datetime
import time
import os
import shutil
import numpy as np
import json
from nsfrb.config import plan_dir,vis_dir
"""
This script copies fast visibilities associated with a given GP observation to a new directory
"""


operations_dir = Path(os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/")
subdirs_to_clear = [
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


def main(args):
    

    if args.populate:
        GP_obs_vis_dir = vis_dir + "GP_observations_" + args.planisot + "/"
        GP_vis_file = vis_dir + "GP_observations_" + args.planisot + "/vis_files.csv"
        with open(GP_vis_file,"w") as csvfile:
            #for subdir, pattern in subdirs_to_clear:
            #files = np.sort(glob.glob(os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/" + subdir + "/" + pattern))
            files = np.sort(glob.glob(GP_obs_vis_dir + "*"))
            for f in files:
                if os.path.basename(str(f)) != 'vis_files.csv':
                    wr = csv.writer(csvfile,delimiter=',')
                    wr.writerow([os.path.basename(str(f)),int(0),""])
                    #print(os.path.basename(str(f)))

        print("Populated csv, returning")
        return 0

    if args.planisot == '':
        print("Requires --planisot argument")
        return 1

    else: 
        #read json file
        jsonfname = plan_dir + "GP_observing_plan_" + args.planisot + ".json"
        with open(jsonfname,"r") as jsonfile:
            plan_metadata = json.load(jsonfile)
        
        #add a field for fast vis file labels
        if 'fast_vis_labels' not in plan_metadata.keys():
            plan_metadata['fast_vis_labels'] = []


        
        start_time = Time(plan_metadata['start_mjd'],format='mjd').datetime
        end_time = Time(plan_metadata['stop_mjd'] + (1/24),format='mjd').datetime
        print(f"Copying operation files modified between "
                f"{start_time.strftime('%Y-%m-%dT%H:%M:%S')} UTC and "
                f"{end_time.strftime('%Y-%m-%dT%H:%M:%S')} UTC")

        #make directory
        GP_obs_vis_dir = vis_dir + "GP_observations_" + args.planisot + "/"
        GP_vis_file = vis_dir + "GP_observations_" + args.planisot + "/vis_files.csv"
        plan_metadata['fast_vis_dir'] = GP_obs_vis_dir
        if len(glob.glob(GP_obs_vis_dir))==0:
            os.system("mkdir " + GP_obs_vis_dir)
        time.sleep(3)
        if len(glob.glob(GP_vis_file))==0:
            os.system("touch " + GP_vis_file)

        csvfile = open(GP_vis_file,"a")
        wr = csv.writer(csvfile,delimiter=',')
        for subdir, pattern in subdirs_to_clear:
            for file in (operations_dir / subdir).glob(pattern):
                #print(os.path.basename(str(file)),type(file))
                

                modtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
                # modtime is timezone naive, so we set it to utc
                # lxc managed containers are all using utc
                modtime = modtime.replace(tzinfo=start_time.tzinfo)
                if modtime >= start_time and modtime <= end_time and len(glob.glob(GP_obs_vis_dir + os.path.basename(str(file))))==0:
                    #print(modtime,cutoff)
                    print(f'Copying {file}')
                    print(f'cp {file} {GP_obs_vis_dir}')
                    os.system(f'cp {file} {GP_obs_vis_dir}')
                    
                    #add to json
                    label = os.path.basename(str(file))
                    label = label[len(label)-label[::-1].index("_")-1:label.index(".out")]
                    print(label)
                    plan_metadata['fast_vis_labels'].append(label)

                    #write to csv
                    wr.writerow([os.path.basename(str(file)),int(0),""])
        csvfile.close()
        #update json
        with open(jsonfname,"w") as jsonfile:
            json.dump(plan_metadata,jsonfile)

        return 0

if __name__=="__main__":
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--planisot',type=str,help='ISOT of GP plan in plan directory',default='')
    parser.add_argument('--populate',action='store_true',default=False,help="Don't clear vis, just re-populate the csv")
    args = parser.parse_args()

    main(args)
