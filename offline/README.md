# Offline

This folder contains scripts for copying and clearing fast visibilities used for the offline NSFRB pipeline.

## Structure

- `getfastvis_service.py`: This script runs a service that queries the ETCD manager and rsyncs fast visibility data from the corr nodes to `dsa110-nsfrb-fast-visibilities`. It runs continuously in the screen `getfastvisservice`
- `clearvis.sh`: This bash script clears all visibilities in the `dsa110-nsfrb-fast-visibilities` directory. Make sure to backup to `dsastorage:/mnt/data/dsa110-nsfrb-fast-visibilities` first.

## Usage

To run the fast visibility service:

```bash
python getfastvis_service.py
```

To copy fast visibilities to dsastorage, run:

```bash
scp dsa110-nsfrb-fast-visibilities/lxd110hXX/* dsastorage:/mnt/data/dsa110-nsfrb-fast-visibilities/lxd110hXX
```

replacing XX with the desired corr node. Then to clear visibilities from the local machine:

```bash
./offline/clearvis.sh
```
