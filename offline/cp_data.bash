#!/bin/bash
corrs=("h03" "hh04" "h05" "h06" "hh07" "h08" "h10" "h11" "hh12" "h14" "h15" "h16" "h18" "h19" "h21" "h22")
fullnames=("lxd110h03" "lxd110h04" "lxd110h05" "lxd110h06" "lxd110h07" "lxd110h08" "lxd110h10" "lxd110h11" "lxd110h12" "lxd110h14" "lxd110h15" "lxd110h16" "lxd110h18" "lxd110h19" "lxd110h21" "lxd110h22")
sbs=("sb00" "sb01" "sb02" "sb03" "sb04" "sb05" "sb06" "sb07" "sb08" "sb09" "sb10" "sb11" "sb12" "sb13" "sb14" "sb15")

while :
      do

	  for i in ${!corrs[@]}; do
	      rsync -avv --remove-source-files ${corrs[$i]}.pro.pvt:./data/nsfrb_*.out ${NSFRBDATA}dsa110-nsfrb-fast-visibilities/${fullnames[$i]}/ > tmp1.txt
	      #rsync -avv lxd110h23.pro.pvt:/media/ubuntu/ssd/sherman/code/testfile*.txt . > tmp1.txt
	      echo "finished rsync"
	      cat tmp1.txt
	      head -n -5 tmp1.txt > tmp2.txt && tail -n +4 tmp2.txt > tmp1.txt && rm tmp2.txt
              echo "finished trimming"
	      cat tmp1.txt

	      cat tmp1.txt | while read l
	      do
                  IFS=' ' read -r -a array <<< "$l"
		  echo "${array[0]},0," >> ${NSFRBDATA}dsa110-nsfrb-fast-visibilities/vis_files.csv
	      done	
	      rm tmp1.txt
	      echo "finished transfer"
	      #tail -32 ${NSFRBDATA}dsa110-nsfrb-fast-visibilities/vis_files.csv
	  done

	  sleep 10

done


