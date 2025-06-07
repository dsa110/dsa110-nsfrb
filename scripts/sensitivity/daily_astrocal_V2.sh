#!/bin/bash

python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 100
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 100
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 100
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 100
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 100

while sleep 43200 
do
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 100
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 100
done
