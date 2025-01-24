#!/bin/bash
#echo "start here"

> newfile.csv

runfile=$1
maxrun=$2
ngulp=$3
gulpoffset=$4
inject=$5
injectflag=" "
if ([[ $inject > 0 ]]); then
	injectflag=" --inject"
fi

filedir=$6
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
	
	#get date file was created
	labeldate=$(stat -c '%y' ${NSFRBDATA}dsa110-nsfrb-fast-visibilities/*/${labelinit})
	labeldate=${labeldate:0:10}
	#echo $labelinit $labeldate
		

	#run imager if not already run for this isot and if specified
	#echo $label $runfile
	if ([[ "$runfile" == "all" ]] || [[ "$label" == "$runfile" ]] || [[ "$labeldate" == "$runfile" ]]) && (( $count < $maxrun )); then
		echo $label
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
			if [ -z "$6" ]; then
				python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py $label --verbose --offline --num_gulps $ngulp --gulp_offset $4 --num_time_samples 25 --sb --nchans_per_node 2 --search$injectflag --num_inject $inject --snr_inject 100000000 --dm_inject 0 --width_inject 2 --offsetRA_inject 0 --offsetDEC_inject 0 --inject_noiseless #>>/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-logfiles/inject_log.txt 2>&1
			else
				python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py $label --verbose --offline --num_gulps $ngulp --gulp_offset $4 --num_time_samples 25 --sb --nchans_per_node 2 --filedir $filedir --search$injectflag --num_inject $inject --snr_inject 100000000 --dm_inject 0 --width_inject 2 --offsetRA_inject 0 --offsetDEC_inject 0 --inject_noiseless 

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
