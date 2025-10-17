####exact
#sleep 52200
dec=44.47
cpunum=20
reftime=$(date +%Y-%m-%dT)00:00:00
echo $reftime

## --ngulps 15
taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size 301 --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --getrealtimecals --search_dec $dec
taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --getrealtimecals --search_dec $dec

taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 90 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --getrealtimecals --search_dec $dec
while sleep 86400
do
	reftime=$(date +%Y-%m-%dT)00:00:00
	echo $reftime
	#astrometry
        taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 90 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --getrealtimecals --search_dec $dec

	#bright
	taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size 301 --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --getrealtimecals --search_dec $dec
	#random
	taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange 0.5 --randomsources --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --getrealtimecals --search_dec $dec
done
