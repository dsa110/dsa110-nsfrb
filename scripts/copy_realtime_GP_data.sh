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
echo "clearing backup cands..."
rm /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/backup_raw_cands/*
echo "stopping procserver..."
systemctl --user stop clearvis
systemctl --user stop rt_injector_test
systemctl --user stop procserver_RX
systemctl --user stop procserver_search
systemctl --user stop T4manager
echo "activating savesearch mode..."
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/process_server/run_proc_server_search  /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_proc_server_search
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/process_server/run_proc_server_search  /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_proc_server_search_savesearch
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/dsaT4/run_T4_manager /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_T4_manager
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/dsaT4/run_T4_manager /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_T4_manager_savesearch
sed -i '30s/$/ --savesearch --realtimegp/' /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_proc_server_search_savesearch
sed -i -e 's/--trigger//g' /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_T4_manager_savesearch
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_proc_server_search_savesearch /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/process_server/run_proc_server_search 
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_T4_manager_savesearch /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/dsaT4/run_T4_manager
echo "starting procserver, savesearch on..."
systemctl --user start T4manager
systemctl --user start procserver_search
sleep 30
systemctl --user start procserver_RX
systemctl --user start realtime_gp
echo "observing for $obstime hours..."
sleep ${obstime}h
echo "done observation, stopping procserver..."
systemctl --user stop procserver_RX
systemctl --user stop procserver_search
systemctl --user stop T4manager
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_proc_server_search /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/process_server/run_proc_server_search
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_T4_manager /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/dsaT4/run_T4_manager

echo "wait for fast vis to finish copying..."
sleep 1800
systemctl --user stop realtime_gp

echo "starting procserver normally..."
systemctl --user start T4manager
systemctl --user start procserver_search
sleep 30
systemctl --user start procserver_RX

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
