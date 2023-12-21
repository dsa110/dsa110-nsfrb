#!/bin/bash

#run this script to open server and receive 32x32x25x16 image data
> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt
#cd /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/
script -q -c "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/socket_server_test_V3.out -f" /dev/null | python run_search.py

