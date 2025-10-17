#!/bin/bash

#clear images
ssh lxd110h20.pro.pvt "rm /home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/*png"
ssh lxd110h20.pro.pvt "rm /home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/*json"

if [ -z "$1" ]; then
	caldate=$(date -I)
else
	caldate=$1
fi
echo $caldate

#only copy files from today
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/NVSS*_$caldate*image_speccal*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/NVSS*_$caldate*speccal*refimage*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
if [ "$1" == "ofbimage" ]; then
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/NVSS*_ofbimage*image_speccal*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/NVSS*_ofbimage*speccal*refimage*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
fi
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/PSR*_$caldate*image_speccal*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/PSR*_$caldate*speccal*refimage*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/RFC*_$caldate*astrocal*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
if [ "$1" == "ofbimage" ]; then 
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/RFC*_ofbimage*astrocal*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/RFC*astrocal*GP_obs*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/J*RFC*astrocal*GP_obs*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
fi
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*NVSStotal*cal*.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*NVSStotal*cal*_calibrated.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*NVSStotal*cal*_calibrated_violin.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*NVSStotal*cal*_calibrated_scatter.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*NVSStotal*cal*_calibratedSNR.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*NVSStotal*cal*_calibratedSNRFLUX.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*NVSStotal*cal*_completeness.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*RFCtotal*cal.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*RFCtotal*cal_drift.png lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/

scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/READMEnsfrbcal.pdf lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_speccal.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_astrocal.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB*J*_astrocal.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB*J*_speccal.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_excludecal.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_noisestats.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/rt_speccal_timestamps_$caldate*.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/rt_astrocal_timestamps_$caldate*.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/*speccal*_calibratedSNRFLUX_PSRB0329.pdf lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/

if [ "$1" == "ofbimage" ]; then
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_speccal_ofbimage*.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_astrocal_ofbimage*.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_*J*astrocal_*GP*.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_excludecal_ofbimage*.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
	scp /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/NSFRB_noisestats_ofbimage*.json lxd110h20.pro.pvt:/home/ubuntu/proj/websrv/webPLOTS/nsfrbcal/
fi
