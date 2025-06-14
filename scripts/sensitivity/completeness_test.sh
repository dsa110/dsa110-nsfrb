####exact

#astrometry
#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --outriggers
#python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only


#(1) flux calibration: measure flux of 25 brightest sources
python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --completeness --completenessxmatch 30 --minsrc_NVSS 0 --image_size 301 --decrange 0.25 --init_speccal
for i in {25..2500..25};
do
	echo $i
	python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --completeness --completenessxmatch 30 --minsrc_NVSS $i --image_size 301 --decrange 0.25
done
#(2) completeness: want to cover all sources possible, select randomly, repeat multiple times
#python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --completeness --completenessxmatch 30 --randomsources
#python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --completeness --completenessxmatch 30 --randomsources
#python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --completeness --completenessxmatch 30 --randomsources
#python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --completeness --completenessxmatch 30 --randomsources
#python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --completeness --completenessxmatch 30 --randomsources


# completeness requires process server, so don't repeat unsupervised. only run this in breaks between GP data processing
