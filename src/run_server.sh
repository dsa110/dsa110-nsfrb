#!/bin/bash

#run this script to open server and receive 32x32x25x16 image data
> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt
script -q -c "./socket_server_test_V3.out -f" /dev/null | python startprocess_V3.py


