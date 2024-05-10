# Scripts 

This folder contains scripts to test and monitor the DSA-110 NSFRB search pipeline.

## Structure

The folder is structured as follows:

- `ms_vis_extraction.py [num_gulp] [num_time_samples]`: This function runs the imaging pipeline on a test visibility .ms file and sends the output using `nsfrb.TXclient` to the process sever. It takes the following command-line arguments:
	- `num_gulp`: Number of gulps
	- `num_time_samples`: Number of time samples to extract from the .ms file
	- `--verbose`: Enable verbose output, default = False
	- `--plot-uv-analysis`: Enable UV analysis plotting
	- `--plot-dirty-images`: Enable dirty images plotting

- `get_status.sh [wait_time] [num_lines]`: This function reads log files currently in the `dsa110-nsfrb/tmpoutput` and `dsa110-nsfrb/process_server` folders and outputs them to terminal for periodic monitoring. It takes the following command-line arguments:
	- `wait_time`: time in seconds between updates
	- `num_lines`: number of lines displayed from the end of each log file

- `SearchAlgorithmTuning_V1.ipynb`: This is a jupyter notebook used for prototyping and testing the search system.

- `get_status.sh`: This function reads log files currently in the `dsa110-nsfrb/tmpoutput` folder and outputs them to terminal for periodic monitoring, taking the wait time between updates in seconds as the argument

- `socket_client_test_PUT.sh`: This script uses curl (https://curl.se/) to take an input datafile and send using an http PUT command to the T4 server. 
- `socket_client_test_POST.sh` [deprecated] : This script uses curl (https://curl.se/) to take an input datafile and send using an http POST command to the T4 server. This script has been replaced by the more reliable and efficient `socket_client_test_PUT.sh`.

Note the following scripts and folders are DEPRECATED as of 2024-05-10 and are pending deletion in a later version:

- `script_warnings`
- `script_flags`
- `run_NSFRB`
- `kill_NSFRB`
- `run_classifier`
- `run_search`

## Usage

To run the imaging test script:

```bash
python ms_vis_extraction.py [num_gulp] [num_time_samples]
```

To monitor the log files:

```bash
./get_status.sh [wait_time] [num_lines]
```

