#!/bin/bash
conda init
conda activate casa310nsfrb
python _getbaselines.py ${@}

