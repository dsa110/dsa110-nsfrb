#!/bin/bash

python run_astrocal.py --buff 20 --numsources_RFC 100 --numsources_NVSS 100 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
#date -u -I -d '-1 day' | xargs -n 1 ./copy_cal_images.sh

while sleep 43200 
do 
	#echo hello world
	python run_astrocal.py --buff 20 --numsources_RFC 100 --numsources_NVSS 100 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
	#date -u -I -d '-1 day' | xargs -n 1 ./copy_cal_images.sh
done
