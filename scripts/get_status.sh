#!/bin/bash

while :
do

	clear
	echo $1 second refresh interval...
	echo "------------------------------------------------------	Start NSFRB Status Report	------------------------------------------------------"
	echo ""
	echo ">>>>>>>>>>>>>>> run_log.txt"
	tail -10 /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt
	echo ">>>>>>>>>>>>>>>"

	echo ""
	echo ">>>>>>>>>>>>>>> search_log.txt"
	tail -10 /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt
	echo ">>>>>>>>>>>>>>>"

	echo ""
	echo ">>>>>>>>>>>>>>> pipe_log.txt"
	tail -10 /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/pipe_log.txt
	echo ">>>>>>>>>>>>>>>"
	echo "------------------------------------------------------	End NSFRB Status Report		------------------------------------------------------"
	sleep $1
done
