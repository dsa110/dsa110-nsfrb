#!/bin/bash

#run this script to receive sub-image cutouts for NSFRB candidates and start ML classifier 
#> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt
#> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/searchlog_flags.txt
python run_classifier.py 2>"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_warnings/classlog_warnings.txt"


