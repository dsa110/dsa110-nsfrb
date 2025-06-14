####exact
#astrometry
#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 20 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST 
#bright
#python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size 301 --decrange 0.5 --flagBPASS --flagBPASSBURST 
#random
python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST
python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST
python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST
python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST

#astrometry (outriggers)
python run_astrocal.py --buff_speccal 10 --buff_astrocal 100 --numsources_RFC 20 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --outriggers --flagBPASS --flagBPASSBURST --image_size 501

while sleep 10800
do
        #astrometry
        python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 100 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST

	#bright
	python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size 301 --decrange 0.5 --flagBPASS --flagBPASSBURST
	#random
	python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST
	python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST
	python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST
	python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST

	#astrometry (outriggers)
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 100 --numsources_RFC 20 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --outriggers --flagBPASS --flagBPASSBURST --image_size 501
done
