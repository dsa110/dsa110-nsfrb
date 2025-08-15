#!/bin/bash

source ~/.bashrc
source /home/ubuntu/msherman_nsfrb/miniconda/bin/activate casa310nsfrb
taskset --cpu-list 4 /home/ubuntu/msherman_nsfrb/miniconda/envs/casa310nsfrb/bin/python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/clearvis.py --waittime 2 --cand_waittime 0.25 --cadence 0.5 --clearcal --clearcal_waittime 1
