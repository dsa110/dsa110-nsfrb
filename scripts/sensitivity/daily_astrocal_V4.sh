####exact
#pulsars
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 5 --exactposition --includepulsars
#brightest sources
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --exactposition 
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --exactposition
#random sources (so that full flux range covered
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmin 500 --exactposition
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500 --exactposition

while sleep 10800
do
        #astrometry
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --outriggers
        python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only

	#pulsars
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 5 --exactposition --includepulsars
	####exact
	#brightest sources
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --exactposition
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --exactposition
	#random sources (so that full flux range covered
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmin 500 --exactposition
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --speccal_only --randomsources --fluxmax 500 --exactposition


done
