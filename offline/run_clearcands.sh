#!/bin/bash

source ~/.bashrc
source /home/ubuntu/msherman_nsfrb/miniconda/bin/activate casa310nsfrb
/home/ubuntu/msherman_nsfrb/miniconda/envs/casa310nsfrb/bin/python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/clearcands.py --waittime 7
