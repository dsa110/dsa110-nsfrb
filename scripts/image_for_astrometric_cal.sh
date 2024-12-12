#!/bin/bash

cd ../offline/

fnums=("76376" "76438" "76494" "76556" "76612" "76674" "76736" "76792" "76854" "76916" "76978" "77034" "77096" "77158" "77220" "77270" "77332" "77400" "77456" "77512" "77574" "77636" "77692")
for i in ${!fnums[@]}; do
	echo _${fnums[$i]}	
	./run.sh _${fnums[$i]} 1000 1 0
done
