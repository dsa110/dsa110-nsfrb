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
	labeldateall=$(stat -c '%y' ${NSFRBDATA}dsa110-nsfrb-fast-visibilities/*/${labelinit})
	labeldate=${labeldateall:0:10}
	labelseconds=$(date -d "${labeldateall:0:19}" +%s)
	#echo $labelinit $labeldate
	#echo $labelseconds $(date -d "${runfile}" +%s)
	if ([[ "${runfile:0:1}" == "_" ]] && [[ "$label" == "$runfile" ]]); then
		echo "here" "$label" "$runfile" "$count" "$maxrun"
	fi
	if ([[ "${runfile:0:1}" != "_" ]] && [[ "$labeldate" == "${runfile:0:10}" ]]); then # && [[ "$(date -d "${runfile}" +%s)" -lt "$labelseconds" ]]); then
		echo "here" "$label" "$runfile" "${labeldateall:0:19}" "$count" "$maxrun" "$labelseconds" "$(date -d "${runfile}" +%s)"
	fi
	#run imager if not already run for this isot and if specified
	#echo $label $runfile
	if ([[ "$runfile" == "all" ]] || ([[ "${runfile:0:1}" == "_" ]] && [[ "$label" == "$runfile" ]]) || ([[ "${runfile:0:1}" != "_" ]] && [[ "$labeldate" == "${runfile:0:10}" ]] && [[ "$(date -d "${runfile}" +%s)" -lt "$labelseconds" ]])) && (( $count < $maxrun )); then
	#if ([[ "$runfile" == "all" ]] || ([[ "${runfile:0:1}" == "_" ]] && [[ "$label" == "$runfile" ]]) || ([[ "${runfile:0:1}" != "_" ]] && (( "$(date -d "${runfile}" +%s)" < $labelseconds )) ) && [[ "$labeldate" == "$runfile" ]]) && (( $count < $maxrun )); then
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
				python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py $label --verbose --offline --num_gulps $ngulp --gulp_offset $4 --num_time_samples 25 --sb --nchans_per_node 8$injectflag --num_inject $inject --width_inject 2 --inject_noiseless --search --gridsize 301 --flagBPASS --sleeptime 0 #>>/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-logfiles/inject_log.txt 2>&1
			else
				python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline/offline_imager.py $label --verbose --offline --num_gulps $ngulp --gulp_offset $4 --num_time_samples 25 --sb --nchans_per_node 8 --filedir $filedir$injectflag --num_inject $inject --width_inject 2 --inject_noiseless --search --gridsize 301 --flagBPASS --sleeptime 0

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
