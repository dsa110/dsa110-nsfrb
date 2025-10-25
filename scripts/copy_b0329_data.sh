#!/bin/bash

echo "delay to next pulsar pass in ${1} seconds"
sleep $1
reftime=$(date +%Y-%m-%d)
starttime=$(date)
echo "clearing backup cands..."
rm /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/backup_raw_cands/*
echo "stopping procserver..."
systemctl --user stop procserver_RX
systemctl --user stop procserver_search
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_proc_server_search_savesearch /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/process_server/run_proc_server_search 
echo "starting procserver, savesearch on..."
systemctl --user start procserver_search
sleep 60
systemctl --user start procserver_RX
echo "observing for ${2} seconds..."
sleep $2
echo "done observation, stopping procserver..."
systemctl --user stop procserver_RX
systemctl --user stop procserver_search
cp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/tmp_run_proc_server_search /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/process_server/run_proc_server_search
echo "starting procserver normally..."
systemctl --user start procserver_search
sleep 60
systemctl --user start procserver_RX
echo "copying data to /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/"
sudo mkdir /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/backup_raw_cands/*T??:??:??.???_input.npy /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/backup_raw_cands/*T??:??:??.???_searched.npy /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
for i in $(seq 1 $2);
do
	nowtime=$(date -d "$starttime +$i second" +%Y-%m-%dT%H:%M:%S) 
	sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/*/*${nowtime}* /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
	ls -d /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/candidates/*${nowtime}* | xargs -n 1 basename >> /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/cands_for_followup_isot.csv
	sudo cp -r /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/final_cands/candidates/*${nowtime}* /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
	sudo ls /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/*${nowtime}.csv | xargs -n 1 tail -n +2 | cut -d ',' -f 1 | xargs -I {} sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/T4A/NSFRB{}.json /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
	sudo ls /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/*${nowtime}*.csv | xargs -n 1 tail -n +2 | cut -d ',' -f 1 | xargs -I {} sudo cp /dataz/dsa110/nsfrb/dsa110-nsfrb-candidates/init_cands/T4B/NSFRB{}.json /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
done

sudo chown -R ubuntu:ubuntu /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/
sudo chmod -R +rwx /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/PSRB0329_FOR_PAPER/${reftime}_PSRB0329+54/

