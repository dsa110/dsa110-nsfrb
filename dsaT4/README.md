# dsaT4

This folder defines the T4 triggering system for the NSFRB search. It submits candidates to the dask client to copy
baseband voltage data to the h24 candidates directory.

## Structure
- `T4_manager`: This file is modelled off of the T3 manager (see https://github.com/dsa110/dsa110-T3/blob/development/dsaT3/T3\_manager.py) and defines functions to write candidate json files and submit them to the dask scheduler
- `data_manager`: This file is modelled off of the T3 data manager (see https://github.com/dsa110/dsa110-T3/blob/development/dsaT3/data\_manager.py) and defines the NSFRBDataManager class which organizes candidate sub-directories

## Usage
- Only interface using the T4\_manager. Candidate json files are written to the $NSFRBDATA/dsa110-nsfrb-candidates directory using:

```
from dsaT4 import T4_manager as T4m
fname = T4m.nsfrb_to_json(cand_isot,snr,width,dm,ra,dec,trigname,final_cand_dir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + trigname + "/")
```

- Submit the json to the dask scheduler using the json file name:
```
T4m.submit_cand_nsfrb(fname,logfile)
```
