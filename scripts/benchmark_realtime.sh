#!/bin/bash
#create psrdada buffer
#dada_junkdb -b 14899200 -k caba -r 4.440 -t 3.355 -g -m 0 -s 10 ../realtime/hdrtest.txt

#current MJD
startUNIX=$(date +%s)
startMJD=$(echo tmp | awk '{print ($startUNIX/86400.0) + 2440587.5 - 2400000.5}')
startISOT=$(date -d "@$startUNIX" +%Y-%m-%d-%H:%M:%S)
echo $startUNIX $startMJD $startISOT $nowUNIX
COUNTER=0
DEC=71.6

while true
do 
	nowUNIX=$(echo $startUNIX $COUNTER | awk '{printf "%.2f", ($1 + (3.355*$2))}')
	nowMJD=$(echo $nowUNIX | awk '{printf "%.10f", (($1/86400.0) + 2440587.5 - 2400000.5)}')
	nowISOT=$(date -d "@$nowUNIX" +%Y-%m-%d-%H:%M:%S.%3N)
	echo $COUNTER $nowUNIX $nowMJD $nowISOT
	COUNTER=$((COUNTER + 1))

	sed "29s/MJD/MJD ${nowMJD}/" ../realtime/hdrtemplate.txt > ./hdrtestnew.txt
	sed -i "30s/SB/SB 1/" ./hdrtestnew.txt
	sed -i "31s/DEC/DEC ${DEC}/" ./hdrtestnew.txt
	sed -i "25s/UTC_START/UTC_START ${nowISOT}/" ./hdrtestnew.txt

	dada_junkdb -b 14899200 -k caba -r 4.440 -t 3.355 -g -m 0 -s 0.1 ./hdrtestnew.txt 
done
