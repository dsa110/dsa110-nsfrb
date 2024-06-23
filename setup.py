from setuptools import setup
from version import get_git_version

setup(name='dsa110-nsfrb',
      #version=get_git_version(),
      version="0.1.0",
      url='http://github.com/dsa110/dsa110-nsfrb',
      python_requires='>3.8',
#      requirements=['seaborn', 'astropy', 'hdbscan', 'progress'],
      packages=['nsfrb'],
      zip_safe=False)

#get local nsfrb directory
import os
os.system("pwd > metadata.txt")

#make logfile directory outside of git repo
os.system("mkdir ../dsa110-nsfrb-logfiles")
logfiles = ["error_log.txt",
            "pipe_log.txt",
            "run_log.txt",
            "search_log.txt",
            "process_log.txt"]
for l in logfiles:
    os.system("touch ../dsa110-nsfrb-logfiles/" + l)
    os.system("> ../dsa110-nsfrb-logfiles/" + l)

#create directory for noise stats if not created already
os.system("mkdir ../dsa110-nsfrb-noise/")


