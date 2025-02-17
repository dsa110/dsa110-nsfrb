#!/bin/bash
#test
cwd=$(cat ../metadata.txt)

while :
do

	clear
	echo $1 second refresh interval...
	echo "------------------------------------------------------	Start NSFRB Status Report	------------------------------------------------------"
	echo ""
        echo ">>>>>>>>>>>>>>> realtime_imager_log.txt"
        cat ${cwd}/realtime/realtime_imager_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt
        echo ">>>>>>>>>>>>>>>"
	echo "------------------------------------------------------	End NSFRB Status Report		------------------------------------------------------"
	sleep $1
done
