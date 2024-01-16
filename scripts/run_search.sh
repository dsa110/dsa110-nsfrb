#!/bin/bash

#run this script to open server and receive 32x32x25x16 image data
> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt
> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/searchlog_flags.txt
> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt
#cd /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/
script -q -c "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/socket_server_test_V3.out" /dev/null | python run_search.py -v 2>searchlog_warnings.txt #| python get_cutouts.py 
