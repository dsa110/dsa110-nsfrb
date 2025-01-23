from pathlib import Path
import glob
import argparse
import csv
import datetime
import time
import os
import shutil


"""
This script waits for visibilities to pass their 1-day expiration, then deletes them
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
vis_file = os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/vis_files.csv"


def main(args):
    

    if args.populate:
        with open(vis_file,"w") as csvfile:
            for subdir, pattern in subdirs_to_clear:
                for file in (operations_dir / subdir).glob(pattern):
                    wr = csv.writer(csvfile,delimiter=',')
                    wr.writerow([os.path.basename(str(file)),int(0),""])
        print("Populated csv, returning")
        return 0


    while True:
        
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
        
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.waittime)
        print(
            f"Removing operation files last modified prior to "
            f"{cutoff.strftime('%Y-%m-%dT%H:%M:%S')} UTC")
        
        
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
                modtime = modtime.replace(tzinfo=cutoff.tzinfo)
                if "59484" in str(file): print(modtime,cutoff,modtime<cutoff)
                if modtime < cutoff:
                    print(modtime,cutoff)
                    print(f'Removing {file}')
                    
                    try:
                        file.unlink()
                    except Exception as exc:
                        print("File unlink failed:",exc)
                        shutil.rmtree(file)
                    if os.path.basename(str(file)) in delidx_labels.keys():
                        delidx.append(delidx_labels[os.path.basename(str(file))])
                    
        #remove from csv
        if len(delidx)>1:
            print("sed -i.bak -e '" + "d;".join(delidx) + "d' " + vis_file)
            os.system("sed -i.bak -e '" + "d;".join(delidx) + "d' " + vis_file)
        elif len(delidx) == 1:
            print("sed -i.bak -e '" + delidx[0] + "d' " + vis_file)
            os.system("sed -i.bak -e '" + delidx[0] + "d' " + vis_file)
        
        
        time.sleep(args.cadence*3600)

if __name__=="__main__":
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--waittime',type=float,help='Time between clearing visibilities in days, default 1',default=1.0)
    parser.add_argument('--cadence',type=float,help='Time between checking for new vis to clear in hours, default 2',default=2.0)
    parser.add_argument('--populate',action='store_true',default=False,help="Don't clear vis, just re-populate the csv")
    args = parser.parse_args()

    main(args)
