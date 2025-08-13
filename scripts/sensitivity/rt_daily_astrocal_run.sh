####exact
reftime=$(date +%Y-%m-%dT)00:00:00
echo $reftime
gsize=175

#bright
python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals 
#astrometry
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20  --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize
sleep 16200


#bright
python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals  
#astrometry
python run_astrocal.py --buff_speccal 10 --buff_astrocal 20  --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize



while sleep 86400
do
	reftime=$(date +%Y-%m-%dT)00:00:00
	echo $reftime
	#bright
	python run_astrocal.py --buff_speccal 10  --specresid_th 0.3 --ngulps 1 --bmin 20 --robust -2 --speccal_only --exactposition --singlesample --image_size $gsize --decrange 0.5 --flagBPASS --flagBPASSBURST --newsources --nummeasure 25 --reftime $reftime --userealtimecals  
	#astrometry
	python run_astrocal.py --buff_speccal 10 --buff_astrocal 20  --specresid_th 0.3 --ngulps 15 --timebin 25 --bmin 20 --robust -2 --astrocal_only --flagBPASS --flagBPASSBURST --newsources --reftime $reftime --userealtimecals --image_size $gsize

done
