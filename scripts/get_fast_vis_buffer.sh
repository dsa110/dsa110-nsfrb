#!/bin/bash


corrs=("h03" "hh04" "h05" "h06" "hh07" "h08" "hhh10" "h11" "hh12" "h14" "h15" "hh16" "h18" "h19" "h21" "h22")
sbs=("sb00" "sb01" "sb02" "sb03" "sb04" "sb05" "sb06" "sb07" "sb08" "sb09" "sb10" "sb11" "sb12" "sb13" "sb14" "sb15")
for i in ${!corrs[@]}; do
        rsync -avv ${corrs[$i]}.pro.pvt:/tmp/*${sbs[$i]}*.out /dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/sensitivitydata/
        echo ""

done

