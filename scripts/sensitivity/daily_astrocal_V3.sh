#!/bin/bash

#astrometry
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only
#for i in $(seq 1 5);
#do
#brightest sources
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only
#random sources (so that full flux range covered
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmin 500
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500
#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500 --fluxmin 250
#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 250

####exact
#brightest sources
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --exactposition
#random sources (so that full flux range covered
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmin 500 --exactposition
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500 --exactposition

#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmin 750 --exactposition
#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 750 --fluxmin 500 --exactposition
#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500 --fluxmin 250 --exactposition
#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 250 --exactposition
#done

while sleep 43200
do
        #astrometry
        python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only

	#brightest sources
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only
	#random sources (so that full flux range covered
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmin 500
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500
	#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500 --fluxmin 250
	#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 250

	####exact
	#brightest sources
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --exactposition
	#random sources (so that full flux range covered
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmin 500 --exactposition
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500 --exactposition


done
