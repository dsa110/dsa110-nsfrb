#!/bin/bash
#create psrdada buffer
#dada_junkdb -b 14899200 -k caba -r 4.440 -t 3.355 -g -m 0 -s 10 ../realtime/hdrtest.txt

#backup noise and frame dirs
cp -r $NSFRBDIR-noise $NSFRBDIR-noise-backup
cp -r $NSFRBDIR-frames $NSFRBDIR-frames-backup

#on exit, restore
trap 'rm -r $NSFRBDIR-noise; mv $NSFRBDIR-noise-backup $NSFRBDIR-noise; rm -r $NSFRBDIR-frames; mv $NSFRBDIR-frames-backup $NSFRBDIR-frames' EXIT



dada_junkdb -k caba -r 4.440 -t 3600 -g -m 0 -s 0.1 ./hdrtestnew.txt
