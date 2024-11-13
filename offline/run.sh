#!/bin/bash

runfile=$1
maxrun=$2
ngulp=$3
donefiles=() #"00-00-00T00:00:00.000")
linenumber=1
fname="${NSFRBDATA}dsa110-nsfrb-fast-visibilities/vis_files.csv"
cat $fname | while read l
do
	#array=$(echo $l | tr "," "\n")
	
	IFS=',' read -r -a array <<< "$l"
	labelinit=${array[0]}
        label="${labelinit: 10}"
        label="${label: 0:-4}"
        count="${array[1]}"
	

	#run imager if not already run for this isot and if specified
	if ([[ "$runfile" == "all" ]] || [[ "$label" == "$runfile" ]]) && (( $count < $maxrun )); then
		prerun=0
		for dfile in "${donefiles[@]}"
		do
			#echo $dfile $label
			if [[ "$dfile" == "$label" ]]; then
				prerun=1
				count=$((count + 1))
				echo "$labelinit,$count" >> "newfile.csv"
			fi
		done
		if (( $prerun==0 )); then

			#python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py _0 --verbose --offline --num_gulps 1 --save --num_time_samples 25 --search --sb --num_chans 16
			python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py $label --verbose --offline --num_gulps $ngulp --save --num_time_samples 25 --sb --nchans_per_node 2 #>>/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-logfiles/inject_log.txt 2>&1
			#python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py $label $tstamp --verbose --offline --num_gulps 1 --save --num_time_samples 25 --search >>/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-logfiles/inject_log.txt 2>&1
			donefiles+=($label)
			count=$((count + 1))
			
			#update count in file
			echo "$labelinit,$count" >> "newfile.csv"
		fi

	else
		echo "$labelinit,$count" >> "newfile.csv"
	fi
	
	


	linenumber=$((linenumber + 1))
	echo $linenumber,$count
	
done

mv "newfile.csv" $fname
