#!/bin/bash

#run this script to open server and receive 32x32x25x16 image data
script -q -c ./socket_server_test_V3.out /dev/null | python startprocess_V3.py


