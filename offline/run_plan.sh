#!/bin/bash

fileplan=$1
calseconds=3506716800
ants=("0" "1" "2")
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
			#dsacon move ${i} ${elev}
			echo "dsacon move --antnum ${i} --elev ${elev}"
		done
	fi
	echo "-"


done
