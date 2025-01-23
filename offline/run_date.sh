#!/bin/bash


rundate=$1
fname="${NSFRBDATA}dsa110-nsfrb-fast-visibilities/vis_files.csv"



tmp=$(stat -c '%y' /dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/lxd110h03/nsfrb_sb00_113118.out)
