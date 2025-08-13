#!/bin/bash

reftime=$(date +%Y-%m-%d)
imgdir=$NSFRBDIR-images/
tbdir=$NSFRBDIR-tables/

cp "${imgdir}RFCtotal_astrocal.png" "${imgdir}RFCtotal_astrocal_backup${reftime}.png"
cp "${imgdir}RFCtotal_astrocal_drift.png" "${imgdir}RFCtotal_astrocal_drift_backup${reftime}.png"
cp "${imgdir}NVSStotal_image_exact_speccal.png" "${imgdir}NVSStotal_image_exact_speccal_backup${reftime}.png"
cp "${imgdir}NVSStotal_image_exact_speccal_calibrated.png" "${imgdir}NVSStotal_image_exact_speccal_calibrated_backup${reftime}.png"
cp "${imgdir}NVSStotal_image_exact_speccal_calibratedSNR.png" "${imgdir}NVSStotal_image_exact_speccal_calibratedSNR_backup${reftime}.png"
cp "${imgdir}NVSStotal_image_exact_speccal_calibratedSNRFLUX.png" "${imgdir}NVSStotal_image_exact_speccal_calibratedSNRFLUX_backup${reftime}.png"
cp "${imgdir}NVSStotal_image_exact_speccal_calibrated_scatter.png" "${imgdir}NVSStotal_image_exact_speccal_calibrated_scatter_backup${reftime}.png"
cp "${imgdir}NVSStotal_image_exact_speccal_calibrated_violin.png" "${imgdir}NVSStotal_image_exact_speccal_calibrated_violin_backup${reftime}.png"
cp "${imgdir}NVSStotal_image_exact_speccal_completeness.png" "${imgdir}NVSStotal_image_exact_speccal_completeness_backup${reftime}.png"
cp "${tbdir}NSFRB_astrocal.json" "${tbdir}NSFRB_astrocal_backup${reftime}.json"
cp "${tbdir}NSFRB_excludecal.json" "${tbdir}NSFRB_excludecal_backup${reftime}.json"
cp "${tbdir}NSFRB_speccal.json" "${tbdir}NSFRB_speccal_backup${reftime}.json"
cp "${tbdir}NSFRB_noisestats.json" "${tbdir}NSFRB_noisestats_backup${reftime}.json"
