from setuptools import setup
from version import get_git_version

setup(name='dsa110-nsfrb',
      version=get_git_version(),
      #version="0.1.0",
      url='http://github.com/dsa110/dsa110-nsfrb',
      python_requires='>3.8',
#      requirements=['seaborn', 'astropy', 'hdbscan', 'progress'],
      packages=['nsfrb','simulations_and_classifications','inject','dsaT4'],
      zip_safe=False)

#get local nsfrb directory
import os
os.system("pwd > metadata.txt")

#make logfile directory outside of git repo
os.system("mkdir ../dsa110-nsfrb-logfiles")
logfiles = ["error_log.txt",
            "inject_error_log.txt",
            "pipe_log.txt",
            "run_log.txt",
            "search_log.txt",
            "process_log.txt",
            "candcutter_log.txt",
            "candcuttertask_log.txt",
            "candcutter_error_log.txt",
            "inject_log.txt",
            "time_log.txt"]
for i in range(len(logfiles)):
    l = logfiles[i]
    os.system("touch ../dsa110-nsfrb-logfiles/" + l)
    os.system("> ../dsa110-nsfrb-logfiles/" + l)


"""
#create file to store trials for candcutter
os.system("touch ../dsa110-nsfrb-candidates/DMtrials.npy")
os.system("touch ../dsa110-nsfrb-candidates/widthtrials.npy")
os.system("touch ../dsa110-nsfrb-candidates/SNRthresh.npy")
"""
#create directory for noise stats if not created already
os.system("mkdir ../dsa110-nsfrb-noise/")
"""
#create candidates directory 
os.system("mkdir ../dsa110-nsfrb-candidates/")
os.system("mkdir ../dsa110-nsfrb-candidates/raw_cands/")
os.system("mkdir ../dsa110-nsfrb-candidates/final_cands/")
os.system("mkdir ../dsa110-nsfrb-candidates/backup_raw_cands/")
"""
#create injections directory
os.system("mkdir ../dsa110-nsfrb-injections/")

#create directory for stored PSFs
os.system("mkdir ../dsa110-nsfrb-PSF/")

#create directory for observing plans
os.system("mkdir ../dsa110-nsfrb-plans/")
import csv

with open("../dsa110-nsfrb-injections/injections.csv","w") as csvfile:
    wr = csv.writer(csvfile,delimiter=',')
    wr.writerow(['ISOT','DM','WIDTH','SNR'])
csvfile.close()    
with open("../dsa110-nsfrb-injections/recoveries.csv","w") as csvfile:
    wr = csv.writer(csvfile,delimiter=',')
    wr.writerow(['ISOT','DM','WIDTH','SNR','PREDICT','PROB'])
csvfile.close()

"""
with open(os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/fpr_test.csv","w") as csvfile:
    csvfile.write("ISOT,SNR")
    #wr = csv.writer(csvfile,delimiter=',')
    #wr.writerow(['ISOT','SNR'])
csvfile.close()
with open(os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/fnr_test.csv","w") as csvfile:
    csvfile.write("ISOT,SNR")
    #wr = csv.writer(csvfile,delimiter=',')
    #wr.writerow(['ISOT','SNR'])
csvfile.close()
"""



#create directory to store most recent time frame
os.system("mkdir ../dsa110-nsfrb-frames/")

"""
#create directories for fast visibilities if not created already
os.system("mkdir ../dsa110-nsfrb-fast-visibilities/")
for s in ["lxd110h03","lxd110h04","lxd110h05","lxd110h06","lxd110h07","lxd110h08","lxd110h10","lxd110h11","lxd110h12","lxd110h14","lxd110h15","lxd110h16","lxd110h18","lxd110h19","lxd110h21","lxd110h22"]:
    os.system("mkdir ../dsa110-nsfrb-fast-visibilities/" + s)
"""

#create ddirectories for images
os.system("mkdir ../dsa110-nsfrb-images/")
