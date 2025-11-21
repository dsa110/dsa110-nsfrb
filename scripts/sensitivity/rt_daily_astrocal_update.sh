####exact
cpunum=20
#sleep 34200
reftime=$(date +%Y-%m-%dT)00:00:00
echo $reftime
gsize=175
dec=54.57

#bright
taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals --search_dec $dec --update_only
#astrometry
taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 87  --specresid_th 0.3 --ngulps 90 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize --search_dec $dec --update_only


