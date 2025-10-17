#!/bin/bash
starttime=$(date -d "$starttime -3 hour") #"2025-09-30T10:24:38"
echo $starttime
reftime="2025-09-30"
for i in $(seq 1 $2);
do
	nowtime=$(date -d "$starttime +$i second" +%Y-%m-%dT%H:%M:%S) 
	echo $nowtime
	sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/T4A/*${nowtime}* /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
	sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/T4B/*${nowtime}* /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
	#ls -d /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/candidates/*${nowtime}* | xargs -n 1 basename >> /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/cands_for_followup_isot.csv
	sudo cp -r /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/candidates/*${nowtime}* /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
	sudo ls /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/*${nowtime}.csv | xargs -n 1 tail -n +2 | cut -d ',' -f 1 | xargs -I {} sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/T4A/NSFRB{}.json /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
	sudo ls /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/*${nowtime}*.csv | xargs -n 1 tail -n +2 | cut -d ',' -f 1 | xargs -I {} sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/T4B/NSFRB{}.json /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
done
