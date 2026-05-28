#!/bin/bash



#corrs=("h03" "hh04" "h05"  "h06" "hh07" "h08" "hhh10" "h11" "hh12" "h14" "h15" "hh16" "h18" "h19" "h21" "h22")
corrs=("n03" "n04" "n05"  "n06" "n07" "n08" "n10" "n11" "n12" "n14" "n15" "n16" "n18" "n19" "n21" "n22")
sbs=("SB00" "SB01" "SB02" "SB03" "SB04" "SB05" "SB06" "SB07" "SB08" "SB09" "SB10" "SB11" "SB12" "SB13" "SB14" "SB15") 
for i in ${!corrs[@]}; do
        #ssh ${corrs[$i]}.pro.pvt "ip address" | grep "10.41.0\|10.42.0"
	echo "copying gridsize_$1 files to ${corrs[$i]}.pro.pvt"
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-injections/realtime_staging/gridsize_$1/*SB$i.npy ${corrs[$i]}.pro.pvt:/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/inject/realtime_staging_sb/
	echo ""

done
