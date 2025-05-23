#!/bin/bash

python run_astrocal.py --buff_speccal 5 --buff_astrocal 20 --numsources_RFC 25 --numsources_NVSS 25 --minsrc_NVSS 0 --minsrc_RFC 0 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
python run_astrocal.py --buff_speccal 5 --buff_astrocal 20 --numsources_RFC 25 --numsources_NVSS 25 --minsrc_NVSS 25 --minsrc_RFC 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
python run_astrocal.py --buff_speccal 5 --buff_astrocal 20 --numsources_RFC 25 --numsources_NVSS 25 --minsrc_NVSS 50 --minsrc_RFC 50 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
python run_astrocal.py --buff_speccal 5 --buff_astrocal 20 --numsources_RFC 25 --numsources_NVSS 25 --minsrc_NVSS 75 --minsrc_RFC 75 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
#date -u -I -d '-1 day' | xargs -n 1 ./copy_cal_images.sh

while sleep 43200 
do 
	#echo hello world
	python run_astrocal.py --buff_speccal 5 --buff_astrocal 20 --numsources_RFC 25 --numsources_NVSS 25 --minsrc_NVSS 0 --minsrc_RFC 0 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
	python run_astrocal.py --buff_speccal 5 --buff_astrocal 20 --numsources_RFC 25 --numsources_NVSS 25 --minsrc_NVSS 25 --minsrc_RFC 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
	python run_astrocal.py --buff_speccal 5 --buff_astrocal 20 --numsources_RFC 25 --numsources_NVSS 25 --minsrc_NVSS 50 --minsrc_RFC 50 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
	python run_astrocal.py --buff_speccal 5 --buff_astrocal 20 --numsources_RFC 25 --numsources_NVSS 25 --minsrc_NVSS 75 --minsrc_RFC 75 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2
	#date -u -I -d '-1 day' | xargs -n 1 ./copy_cal_images.sh
done
