####exact
#sleep 52200
dec=16.27 #54.57
decrange=0.25
cpunum=20
reftime=$(date +%Y-%m-%dT)00:00:00
echo $reftime

## --ngulps 15
taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size 301 --decrange $decrange --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --getrealtimecals --search_dec $dec
taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange $decrange --randomsources --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --getrealtimecals --search_dec $dec

taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 90 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --getrealtimecals --search_dec $dec --decrange $decrange
while sleep 86400
do
	reftime=$(date +%Y-%m-%dT)00:00:00
	echo $reftime
	#astrometry
        taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 20 --numsources_RFC 10 --specresid_th 0.3 --ngulps 90 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --getrealtimecals --search_dec $dec --decrange $decrange

	#bright
	taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size 301 --decrange $decrange --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --getrealtimecals --search_dec $dec
	#random
	taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --numsources_NVSS 25 --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --minsrc_NVSS 25 --image_size 301 --decrange $decrange --randomsources --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --getrealtimecals --search_dec $dec
done
