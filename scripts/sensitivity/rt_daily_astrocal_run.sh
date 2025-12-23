####exact
cpunum=20
#sleep 34200
reftime=$(date +%Y-%m-%dT)00:00:00
#reftime="2025-12-22T00:00:00"
echo $reftime
gsize=175
dec=16.27
testcoord="J000000+162121"

#bright
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals  --search_dec $dec --update_only
#astromety
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 87  --specresid_th 0.3 --ngulps 35 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize --search_dec $dec --target J000000+543412 --targetMJD 60933.333333333336 --target_timerange 8760 --target_decrange 2 --update_only
#astrometry
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 87  --specresid_th 0.3 --ngulps 35 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize --search_dec $dec --target J000000+543412 --targetMJD 60933.333333333336 --target_timerange 8760 --target_decrange 2 
#bright
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals  --search_dec $dec --update_only
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 87  --specresid_th 0.3 --ngulps 29 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize --search_dec $dec --target J000000+543412 --targetMJD 60933.333333333336 --target_timerange 8760 --target_decrange 2  --u
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals  --search_dec $dec --astrocal_only


#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 10  --specresid_th 0.3 --ngulps 5 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize --search_dec $dec --target J000000+543412 --targetMJD 60933.333333333336 --target_timerange 8760 --target_decrange 2 
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals  --search_dec $dec


#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 87  --specresid_th 0.3 --ngulps 35 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize --search_dec $dec --target $testcoord --targetMJD 60933.333333333336 --target_timerange 8760 --target_decrange 2
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals  --search_dec $dec --init_speccal
while sleep 7200 #28800
do
	reftime=$(date +%Y-%m-%dT)00:00:00
	echo $reftime
	#bright
	taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals  --search_dec $dec
	#astrometry
	taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 87  --specresid_th 0.3 --ngulps 35 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize --search_dec $dec --target $testcoord --targetMJD 60933.333333333336 --target_timerange 8760 --target_decrange 2

done
