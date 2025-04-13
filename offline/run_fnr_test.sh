#!/bin/bash
  
systemctl --user start procserver_fnr.service 
cd /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/offline
#ls /dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/lxd110h03/*out
#cat /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/fpr_set.txt | xargs -I {} basename {} .out | tail --bytes=+11 | xargs -I {} ./run.sh {} 100 1 0 1 "--snr_min_inject 1e5 --snr_max_inject 1e8"
cat /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/fpr_set.txt | while read l
do
        bname=$(basename $l .out)
        fnum="${bname:10}"

        echo $fnum
        ./run.sh $fnum 100 90 0 90 "--snr_min_inject 1e1 --snr_max_inject 1e5"
done
systemctl --user stop procserver_fnr.service 
