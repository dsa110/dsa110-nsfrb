#!/bin/bash
#test
cwd=$(cat ../metadata.txt)

while :
do

	clear
	echo $1 second refresh interval...
	echo "------------------------------------------------------	Start NSFRB Status Report	------------------------------------------------------"
	echo ""
        echo ">>>>>>>>>>>>>>> binary_log.txt"
        cat ${cwd}-logfiles/binary_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt
        echo ">>>>>>>>>>>>>>>"
	
	
	echo ""
	echo ">>>>>>>>>>>>>>> run_log.txt"
	tail -$2 ${cwd}-logfiles/run_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt
	echo ">>>>>>>>>>>>>>>"

	echo ""
	echo ">>>>>>>>>>>>>>> search_log.txt"
	tail -$2 ${cwd}-logfiles/search_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt
	echo ">>>>>>>>>>>>>>>"

	echo ""
	echo ">>>>>>>>>>>>>>> pipe_log.txt"
	tail -$2 ${cwd}-logfiles/pipe_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/pipe_log.txt
	echo ">>>>>>>>>>>>>>>"

        echo ""
        echo ">>>>>>>>>>>>>>> error_log.txt"
        tail -$2 ${cwd}-logfiles/error_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/server_log.txt
        echo ">>>>>>>>>>>>>>>"


        echo ""
        echo ">>>>>>>>>>>>>>> process_log.txt"
        tail -$2 ${cwd}-logfiles/process_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt
        echo ">>>>>>>>>>>>>>>"
	echo "------------------------------------------------------	End NSFRB Status Report		------------------------------------------------------"
	sleep $1
done
