#!/bin/bash

source ~/.bashrc
source /home/ubuntu/msherman_nsfrb/miniconda/bin/activate casa310nsfrb
/home/ubuntu/msherman_nsfrb/miniconda/envs/casa310nsfrb/bin/python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/clearvis.py --waittime 2 --cadence 2 --clearcal --clearcal_waittime 4
