#!/bin/bash


#corrs=("h03" "hh04" "h05" "h06" "hh07" "h08" "hhh10" "h11" "hh12" "h14" "h15" "hh16" "h18" "h19" "h21" "h22")
corrs=("n03" "n04" "n05" "n06" "n07" "n08" "n10" "n11" "n12" "n14" "n15" "n16" "n18" "n19" "n21" "n22")

for i in ${!corrs[@]}; do
	ssh ${corrs[$i]}.pro.pvt "ip address" | grep "10.41.0\|10.42.0"
	echo ""

done
