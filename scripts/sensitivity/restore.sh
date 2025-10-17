#!/bin/bash


reftime=$1 #(date +%Y-%m-%d) 

cp $NSFRBDIR-tables/NSFRB_astrocal_backup$reftime.json $NSFRBDIR-tables/NSFRB_astrocal.json
cp $NSFRBDIR-tables/NSFRB_speccal_backup$reftime.json $NSFRBDIR-tables/NSFRB_speccal.json
cp $NSFRBDIR-tables/NSFRB_excludecal_backup$reftime.json $NSFRBDIR-tables/NSFRB_excludecal.json
cp $NSFRBDIR-tables/NSFRB_noisestats_backup$reftime.json $NSFRBDIR-tables/NSFRB_noisestats.json

