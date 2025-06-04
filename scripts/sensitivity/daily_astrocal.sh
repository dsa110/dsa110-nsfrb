#!/bin/bash

python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only
for i in 25 50 75 100 125 150 175 200 225 250 275;
do
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --minsrc_NVSS $i --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only
done


for i in 25 50 75 100 125 150 175 200 225 250 275;
do
        python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --minsrc_NVSS $i --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --fluxmax 100
done
#date -u -I -d '-1 day' | xargs -n 1 ./copy_cal_images.sh

while sleep 43200 
do 
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only
	for i in 25 50 75 100 125 150 175 200 225 250 275;
	do      
		python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --minsrc_NVSS $i --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only
	done    
        for i in 25 50 75 100 125 150 175 200 225 250 275;	
	do
		python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --minsrc_NVSS $i --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --fluxmax 100
	done
	#echo hello world
	#date -u -I -d '-1 day' | xargs -n 1 ./copy_cal_images.sh
done
