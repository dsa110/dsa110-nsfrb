#!/bin/bash
#echo "start here"

> newfile.csv

runfile=$1
maxrun=$2
ngulp=$3
filedir=$4
echo "here"
echo $filedir
echo "here"
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
	#echo $label $runfile
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
			if [ -z "$4" ]; then
				python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py $label --verbose --offline --num_gulps $ngulp --gulp_offset 0 --save --num_time_samples 5 --sb --nchans_per_node 2 --wstack #>>/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-logfiles/inject_log.txt 2>&1
			else
				python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py $label --verbose --offline --num_gulps $ngulp --gulp_offset 0 --save --num_time_samples 5 --sb --nchans_per_node 2 --wstack --filedir $filedir

			fi
			donefiles+=($label)
			count=$((count + 1))
			
			#update count in file
			echo "$labelinit,$count" >> "newfile.csv"
		fi

	else
		echo "$labelinit,$count" >> "newfile.csv"
	fi
	
	


	linenumber=$((linenumber + 1))
	#echo $linenumber,$count
	
done

mv "newfile.csv" $fname
