#!/bin/bash

echo "executing GP survey: ${1}"
plandir="/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-plans/"
planjson="/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-plans/${1}.json"
obstime=$(jq '.full_obs_time_hr' $planjson)
nowtime=$(date +%s)
gpname=${1}
gptime=${gpname:18}
gptimes=$(date -d $gptime +%s)
deltime=$(( $gptimes - $nowtime ))
echo "delay to next GP start time at pass in ${deltime} seconds"
sleep $deltime
reftime=$(date +%Y-%m-%d)
starttime=$(date)

echo "copying data to /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/"
sudo mkdir /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
#sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/backup_raw_cands/*T??:??:??.???_input.npy /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
#sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/backup_raw_cands/*T??:??:??.???_searched.npy /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
ss=$(bc <<< "$obstime * 60")
for i in $(seq 1 $ss);
do
	nowtime=$(date -d "$starttime +$i second" +%Y-%m-%dT%H:%M:%S) 
	sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/backup_raw_cands/*${nowtime}*_input.npy /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
	sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/backup_raw_cands/*${nowtime}*_search.npy /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
	sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/*/*${nowtime}* /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
	ls -d /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/candidates/*${nowtime}* | xargs -n 1 basename >> /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/cands_for_followup_isot.csv
	sudo cp -r /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/candidates/*${nowtime}* /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
	sudo ls /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/*${nowtime}.csv | xargs -n 1 tail -n +2 | cut -d ',' -f 1 | xargs -I {} sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/T4A/NSFRB{}.json /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
	sudo ls /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/*${nowtime}*.csv | xargs -n 1 tail -n +2 | cut -d ',' -f 1 | xargs -I {} sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/T4B/NSFRB{}.json /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
done

sudo chown -R ubuntu:ubuntu /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
sudo chmod -R +rwx /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/

echo "ALL DONE"
