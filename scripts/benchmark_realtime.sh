#!/bin/bash
#create psrdada buffer
#dada_junkdb -b 14899200 -k caba -r 4.440 -t 3.355 -g -m 0 -s 10 ../realtime/hdrtest.txt

#backup noise and frame dirs
cp -r $NSFRBDIR-noise $NSFRBDIR-noise-backup
cp -r $NSFRBDIR-frames $NSFRBDIR-frames-backup

#on exit, restore
trap 'rm -r $NSFRBDIR-noise; mv $NSFRBDIR-noise-backup $NSFRBDIR-noise; rm -r $NSFRBDIR-frames; mv $NSFRBDIR-frames-backup $NSFRBDIR-frames' EXIT


#current MJD
startUNIX=$(date +%s)
startUNIXDAYS=$((startUNIX / 86400))
startMJD=$(echo "scale=10; ($startUNIX / (24*60*60)) + 40587.5" | bc) #$((startUNIXDAYS + 40587)) #$(echo tmp | awk '{print ($startUNIX/86400.0) + 2440587.5}') #- 2400000.5}')
startISOT=$(date -d "@$startUNIX" +%Y-%m-%d-%H:%M:%S)
echo $startUNIX $startMJD $startISOT $nowUNIX
COUNTER=0
DEC=71.6

sed "29s/MJD/MJD ${startMJD}/" ../realtime/hdrtemplate.txt > ./hdrtestnew.txt
sed -i "30s/SB/SB 1/" ./hdrtestnew.txt
sed -i "31s/DEC/DEC ${DEC}/" ./hdrtestnew.txt
sed -i "25s/UTC_START/UTC_START ${startISOT}/" ./hdrtestnew.txt

dada_junkdb -k caba -r 4.440 -t 3600 -g -m 0 -s 0.1 ./hdrtestnew.txt
