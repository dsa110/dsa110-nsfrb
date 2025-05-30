#!/bin/bash

#clear images
ssh lxd110h20.pro.pvt "rm /home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/*png"


if [ -z "$1" ]; then
	caldate=$(date -I)
else
	caldate=$1
fi
echo $caldate

#only copy files from today
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/NVSS*_$caldate*image_speccal*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/NVSS*_$caldate*speccal*refimage*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/RFC*_$caldate*astrocal*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
#scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/NVSS*_$caldate*baselines.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
#scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/NVSS*_$caldate*fstopimage.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*NVSStotal*cal.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*RFCtotal*cal.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*RFCtotal*cal_drift.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/READMEnsfrbcal.pdf lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
