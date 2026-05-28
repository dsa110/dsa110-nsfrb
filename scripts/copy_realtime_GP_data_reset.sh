#!/bin/bash

echo "done observation, stopping procserver..."
systemctl --user stop procserver_RX
systemctl --user stop procserver_search
systemctl --user stop T4manager
systemctl --user stop rt_injector_test
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
systemctl --user start rt_injector_test


echo "copying fast visibilities"
#sudo mkdir /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/fast_visibilities/
#sudo mv /dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/GP_observations_/* /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
./clear_realtime_GP_data.sh

sudo chown -R ubuntu:ubuntu /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
sudo chmod -R +rwx /dataz/dsa110/nsfrb/dsa110-nsfrb-followup/REALTIME_GP_SEARCH/GP_candidates_${gptime}/
systemctl --user start clearvis

echo "ALL DONE"
