#!/bin/bash

fileplan=$1
calseconds=3506716800
ants=("1" "2" "3" "4" "5" "6" "7" "8" "9" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "24" "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45" "46" "47" "48" "49" "50" "51" "102" "116" "68" "69" "70" "71" "72" "73" "74" "75" "76" "77" "78" "79" "80" "81" "82" "83" "84" "85" "86" "87" "88" "89" "90" "91" "92" "93" "94" "95" "96" "97" "98" "99" "100" "101" "103" "104" "105" "106" "107" "108" "109" "110" "111" "112" "113" "114" "115") #("0" "1" "2")
cat $fileplan | while read l
do
	IFS=',' read -r -a array <<< "$l"
	mjd=${array[0]}
	elev=${array[1]}

	#get duration in seconds to wait
	now=$(date +%s)
	waitseconds=$(echo "${mjd}*86400 - ${calseconds} - ${now}" | bc)
	echo ${mjd}
	echo $(echo "${mjd}*86400" | bc)
	echo $(echo "${mjd}*86400 - ${calseconds}" | bc)
	echo ${now}
	echo $waitseconds

	#sleep for required time IF >0
	if  [[ $waitseconds == -* ]]; then
		echo "not running"
	else
		echo "running"
		sleep 0 #$waitseconds

		#now run elevation slew
		for i in "${ants[@]}"
		do
			dsacon move ${i} ${elev}
			#echo "dsacon move --antnum ${i} --elev ${elev}"
		done
	fi
	echo "-"


done
