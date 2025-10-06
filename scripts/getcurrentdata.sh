#!/bin/bash

tstamp=$(date +%Y-%m-%dT%H:%M:%S.%N)
corrs=("h03" "hh04" "h05" "h06" "hh07" "h08" "hhh10" "h11" "hh12" "h14" "h15" "h16" "h18" "h19" "h21" "h22")
fs=("h03" "h04" "h05" "h06" "h07" "h08" "h10" "h11" "h12" "h14" "h15" "h16" "h18" "h19" "h21" "h22")
sbs=("sb00" "sb01" "sb02" "sb03" "sb04" "sb05" "sb06" "sb07" "sb08" "sb09" "sb10" "sb11" "sb12" "sb13" "sb14" "sb15")
echo $1
while sleep $1; do
	for i in $(seq 0 15); do
		scp ${corrs[$i]}.pro.pvt:/tmp/NSFRB_IMAGE_TMP.npy $NSFRBDATA/dsa110-nsfrb-fast-visibilities/lxd110${fs[$i]}/${tstamp}_${sbs[$i]}_NSFRB_IMAGE_TMP.npy &
		scp ${corrs[$i]}.pro.pvt:/tmp/NSFRB_VIS_TMP.out $NSFRBDATA/dsa110-nsfrb-fast-visibilities/lxd110${fs[$i]}/${tstamp}_${sbs[$i]}_NSFRB_VIS_TMP.out &
		echo ${corrs[$i]} ${tstamp}
	done
done
