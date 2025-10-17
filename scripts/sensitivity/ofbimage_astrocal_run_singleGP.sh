####exact
cpunum="15-20"
#sleep 34200
#reftime="2025-09-15T00:00:00"
gsize=175
namefile="GP_allnames2.txt"
gpfile="GP_observations_2025-02-19T00:00:00.000"

#astrometry
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 29  --specresid_th 0.3 --ngulps 10 --timebin 25 --bmin 20 --robust -2 --flagBPASS --flagBPASSBURST --newsources --userealtimecals --image_size $gsize --ofbimage --decrange 0.5 --fluxmin 400 --exactposition --singlesample --astrocal_only --update_only --GPdir $gpfile

#individual dec solutions
#individual dec solutions
#for i in $(seq -f "%02g" 0 10 90);
#do
#        echo J000000+${i}0000
#        taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 29  --specresid_th 0.3 --ngulps 10 --timebin 25 --bmin 20 --robust -2 --flagBPASS --flagBPASSBURST --newsources --userealtimecals --image_size $gsize --ofbimage --decrange 0.5 --fluxmin 400 --exactposition --singlesample --astrocal_only --target J000000+${i}0000 --update_only --target_timerange 8760 --targetMJD 60724.0 --target_decrange 10 --GPdir $gpfile
#done
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10 --buff_astrocal 29  --specresid_th 0.3 --ngulps 10 --timebin 25 --bmin 20 --robust -2 --flagBPASS --flagBPASSBURST --newsources --userealtimecals --image_size $gsize --ofbimage --decrange 0.5 --fluxmin 400 --exactposition --singlesample --astrocal_only --target J045043+405613 --update_only --target_timerange 8760 --targetMJD 60724.0 --target_decrange 0.016 --GPdir $gpfile

#spec cal

taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --ofbimage --flagcorrs 0 1 --init_speccal --GPdir $gpfile --ofbimagexmatch
#taskset --cpu-list $cpunum python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --ofbimage --fluxmax 100 --randomsources --numsources_NVSS 25 --flagcorrs 0 1 --GPdir $gpfile



