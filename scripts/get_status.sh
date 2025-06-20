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


		for f in ${cwd}-logfiles/*_log.txt;
		do
			echo ""
			echo  ">>>>>>>>>>>>>>> $f"
			tail -$2 $f
			echo  ">>>>>>>>>>>>>>>"

		done 

		echo ""
		echo ">>>>>>>>>>>>>>> journalctl.txt"
		journalctl --user --since today -r -u T4manager.service -n $2 > ${cwd}-logfiles/journalctl.txt
		cat ${cwd}-logfiles/journalctl.txt
		echo ">>>>>>>>>>>>>>>"


	else
		for f in "${@:4}"
                do
                        echo ""
                        echo  ">>>>>>>>>>>>>>> ${cwd}-logfiles/$f.txt"
                        tail -$2 ${cwd}-logfiles/$f.txt
                        echo  ">>>>>>>>>>>>>>>"

                done

		echo ""
                echo ">>>>>>>>>>>>>>> $3.txt"
                tail -$2 ${cwd}-logfiles/$3.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt
                echo ">>>>>>>>>>>>>>>"
	fi
	echo "------------------------------------------------------	End NSFRB Status Report		------------------------------------------------------"
	sleep $1
done
