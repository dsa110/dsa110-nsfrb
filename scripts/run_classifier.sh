#!/bin/bash

#run this script to receive sub-image cutouts for NSFRB candidates and start ML classifier 
#> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt
#> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/searchlog_flags.txt
python run_classifier.py -v 2>classlog_warnings.txt


