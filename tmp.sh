#!/bin/bash
corrs=("h03" "hh04" "h05" "h06" "hh07" "h08" "hhh10" "h11" "hh12" "h14" "h15" "hh16" "h18" "h19" "h21" "h22")
fullnames=("lxd110h03" "lxd110h04" "lxd110h05" "lxd110h06" "lxd110h07" "lxd110h08" "lxd110h10" "lxd110h11" "lxd110h12" "lxd110h14" "lxd110h15" "lxd110hh16" "lxd110h18" "lxd110h19" "lxd110h21" "lxd110h22")
sbs=("sb00" "sb01" "sb02" "sb03" "sb04" "sb05" "sb06" "sb07" "sb08" "sb09" "sb10" "sb11" "sb12" "sb13" "sb14" "sb15")


for i in ${!corrs[@]}; do
	ssh ${corrs[$i]}.pro.pvt "rm /tmp/nsfrb*.out" 
done
