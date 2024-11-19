# Offline

This folder contains scripts for copying and clearing fast visibilities used for the offline NSFRB pipeline.

## Structure

- `run`: This script runs the offline imager on specified visibilities saved to disk. It can be run for an individual observation or on all visibilities that have been processed less than the specified number of times
- `offline_imager`: This defines the main offline imager routine which images visibilities on a given sub-band and either saves to disk or sends them to the process server for offline searching.
- `cp_data`:  This script rsyncs fast visibility data from the corr nodes to `dsa110-nsfrb-fast-visibilities`. It runs continuously in the screen `vikram`
- `clearcands`: This script periodically checks for and removes outdated raw candidate files to clear space. It runs continuously in the screen `clearcands`
- `clearvis`: This script periodically checks for and removes outdated raw visibility files to clear space. It runs continuously in the screen `clearvis`


## Usage

To run the fast visibility service:

```bash
python cp_data.sh
```

To run the `clearcands` service:

```bash
python clearcands --waittime [time between clearing candidates]
```

and to run the `clearvis` service:

```bash
python clearvis --waittime [time between clearing visibilities in hours] --cadence [time between checking for old visibilities] --populate [if set, manually updates internal list of visibility files on disk]
```

To copy fast visibilities to dsastorage, run:

```bash
scp $NSFRBDATA/dsa110-nsfrb-fast-visibilities/lxd110hXX/* dsastorage:/mnt/data/dsa110-nsfrb-fast-visibilities/lxd110hXX
```
replacing XX with the desired corr node. 

