#!/bin/bash
#test
cwd=$(cat ../metadata.txt)

while :
do

	clear
	echo $1 second refresh interval...
	echo "------------------------------------------------------	Start NSFRB Status Report	------------------------------------------------------"
	echo ""
        #echo ">>>>>>>>>>>>>>> binary_log.txt"
        #cat ${cwd}-logfiles/binary_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt
        #echo ">>>>>>>>>>>>>>>"
	
	
	if [ -z "$3" ]; then



        	echo ""
        	echo ">>>>>>>>>>>>>>> error_log.txt"
        	tail -$2 ${cwd}-logfiles/error_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/server_log.txt
        	echo ">>>>>>>>>>>>>>>"


		echo ""
        	echo ">>>>>>>>>>>>>>> candcutter_log.txt"
        	tail -$2 ${cwd}-logfiles/candcutter_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt
        	echo ">>>>>>>>>>>>>>>"

		echo ""
        	echo ">>>>>>>>>>>>>>> candcuttertask_log.txt"
        	tail -$2 ${cwd}-logfiles/candcuttertask_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt
        	echo ">>>>>>>>>>>>>>>"

		echo ""
        	echo ">>>>>>>>>>>>>>> candcutter_error_log.txt"
        	tail -$2 ${cwd}-logfiles/candcutter_error_log.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt
        	echo ">>>>>>>>>>>>>>>"


		echo ""
		echo ">>>>>>>>>>>>>>> journalctl.txt"
		journalctl --user --since today -r -n $2 > ${cwd}-logfiles/journalctl.txt
		cat ${cwd}-logfiles/journalctl.txt
		echo ">>>>>>>>>>>>>>>"


	else
		echo ""
                echo ">>>>>>>>>>>>>>> $3.txt"
                tail -$2 ${cwd}-logfiles/$3.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt
                echo ">>>>>>>>>>>>>>>"
	fi
	echo "------------------------------------------------------	End NSFRB Status Report		------------------------------------------------------"
	sleep $1
done
