#!/bin/bash
#test
cwd=$(cat ../metadata.txt)
> ${cwd}-images/timestatusfile.txt

python -c "from nsfrb.plotting import timestatusplot; timestatusplot(showsamps=90)" &

trap 'kill -9 $(pgrep -f timestatusplot)' EXIT

while :
do

	clear
	echo $1 second refresh interval...
	echo "------------------------------------------------------	Start NSFRB Status Report	------------------------------------------------------"
	echo ""
        cat ${cwd}-images/timestatusfile.txt
	echo ""
	echo "------------------------------------------------	End NSFRB Time Status Report  ------------------------------------------------"
	sleep $1
done

