#!/bin/bash

pgrep -f run_realtime_injector | xargs -n 1 sudo renice -n 0 -p
pgrep -f run_proc_server | xargs -n 1 sudo renice -n -10 -p
pgrep -f run_cand_cutter | xargs -n 1 sudo renice -n 0 -p
pgrep -f daily_astrocal | xargs -n 1 sudo renice -n 10 -p
pgrep -f clearvis.sh | xargs -n 1 sudo renice -n 10 -p
pgrep -f clearvis.py | xargs -n 1 sudo renice -n 10 -p
