#!/bin/bash
#test
cwd=$(cat ../metadata.txt)
> ${cwd}-images/noisestatusfile.txt

python -c "from nsfrb.plotting import noisestatusplot; noisestatusplot()" &

trap 'kill -9 $(pgrep -f noisestatusplot)' EXIT

while :
do

	clear
	echo $1 second refresh interval...
	echo "------------------------------------------------------	Start NSFRB Status Report	------------------------------------------------------"
	echo ""
        cat ${cwd}-images/noisestatusfile.txt
	echo ""

	echo ""
        echo ">>>>>>>>>>>>>>> noiseoffline.txt"
        tail -$2 ${cwd}-noise/noiseoffline.txt #/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt
        echo ">>>>>>>>>>>>>>>"
	echo "------------------------------------------------	End NSFRB Time Status Report  ------------------------------------------------"
	sleep $1
done

