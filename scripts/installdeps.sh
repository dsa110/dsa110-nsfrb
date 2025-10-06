#!/bin/bash


corrs=("hh04" "h05" "h06" "hh07" "h08" "hhh10" "h11" "hh12" "h14" "h15" "h16" "h18" "h19" "h21" "h22")

for i in ${!corrs[@]}; do
	ssh ${corrs[$i]}.pro.pvt "sudo apt-get install libblas-dev liblapack-dev liblapacke-dev"
	echo ""

done
