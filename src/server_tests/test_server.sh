#!/bin/bash

#run this script to open server and receive 32x32x25x16 image data
> /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt
script -q -c "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/socket_server_test_V3.out" /dev/null | python test_server.py

