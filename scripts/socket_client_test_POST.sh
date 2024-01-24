#!/bin/bash

#curl http://10.41.0.182:8080/echo
#curl -F "file=@senddata.dat;type=text/plain" http://10.41.0.182:8080/senddata.dat --verbose --trace-ascii here.txt --keepalive-time 5
#curl -F "file=@simulated1.npy;type=text/plain" http://10.41.182:8080/simulated1.npy --verbose --trace-ascii here.txt --keepalive-time 5
#curl -F "file=@simulated1test.npy;type=application/octet-stream" http://10.41.182:8080/simulated1test.npy --verbose --trace-ascii here.txt --keepalive-time 5

TIMEFORMAT='It took %R seconds.'
time {
curl -F "file=@empty_sent.npy;type=application/octet-stream" http://10.41.0.94:8080/empty_sent.npy --verbose --trace-ascii /media/ubuntu/ssd/sherman/code/here.txt --keepalive-time 5 --http0.9

}

#etcd status keys
#/cmd/cal -- when to run calibration
#/mond/cal -- 
# systemctl status -- see what services running
# want to see how etcd keys configured for cal23 services, copy for slow transient/ template for nsfrb service data transfer
# benchmark/timing tests for data transfer
#draw out networking diagram


#offline implementation
## keep imaging, search separately as much as possible
## need to transfer visibilities to h23 from corr nodes for imaging (http?)
