import json
from nsfrb import config
import os
import sys
from astropy.time import Time
f = open(config.table_dir + "rt_speccal_timestamps_"+sys.argv[1]+"T00:00:00.000.json","r")
tab = json.load(f)
f.close()

for k in tab.keys():
    print(k,Time(tab[k],format='mjd').isot)
