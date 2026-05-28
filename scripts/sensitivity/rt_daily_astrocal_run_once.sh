####exact
cpunum=20
#sleep 34200
reftime=$(date +%Y-%m-%dT)00:00:00
refmjd=$(echo "($(date +%s)/86400) + 40587" | bc -l)
#reftime="2025-12-22T00:00:00"
echo $reftime
gsize=175
dec=16.27
testcoord="J000000+162121"

taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 87  --specresid_th 0.3 --ngulps 7 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --reftime $reftime --userealtimecals --image_size $gsize --search_dec $dec --target $testcoord --targetMJD $refmjd --target_timerange 8760 --target_decrange 2 --newsources --update_only $1
taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --nummeasure 25 --reftime $reftime --userealtimecals  --search_dec $dec $1
