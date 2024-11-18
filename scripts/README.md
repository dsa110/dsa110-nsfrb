# Scripts 

This folder contains miscellaneous scripts and notebooks to test and monitor the DSA-110 NSFRB search pipeline. These scripts are not used during routine operation of the offline or realtime NSFRB pipelines.

## Structure

The folder is structured as follows:

- `ms_fast_vis_extraction.py,ms_vis_extraction.py [num_gulp] [num_time_samples]`: This function runs the imaging pipeline on a test visibility .ms file and sends the output using `nsfrb.TXclient` to the process sever. It takes the following command-line arguments:
	- `num_gulp`: Number of gulps
	- `num_time_samples`: Number of time samples to extract from the .ms file
	- `--verbose`: Enable verbose output, default = False
	- `--plot-uv-analysis`: Enable UV analysis plotting
	- `--plot-dirty-images`: Enable dirty images plotting

- `get_status.sh [wait_time] [num_lines]`: This function reads log files currently in the `dsa110-nsfrb/tmpoutput` and `dsa110-nsfrb/process_server` folders and outputs them to terminal for periodic monitoring. It takes the following command-line arguments:
	- `wait_time`: time in seconds between updates
	- `num_lines`: number of lines displayed from the end of each log file

- `SearchAlgorithmTuning_V1.ipynb`: This is a jupyter notebook used for prototyping and testing the search system.

- `clear_casa_logs.sh`: This script clears casa log files resulting from imaging

- `init_noise_from_fast_vis_UVW.py`: This script initializes noise statistics using the example files listed in `init_noise_fnames.txt`

- `ms_fast_vis_extraction.py,ms_vis_extraction.py`: These are test scripts used to verify the imager and transmission to the process server

- `noise_testing.py,snr_calibration.py`: These scripts were used to define the scale factor between noise in visibilities and noise in images, the result of which is saved as `nsfrb.config.vis_to_img_slope`

- `sort_training_set.py`: This script sorts image cutouts saved as training data after they have been manually labelled in the `$DSANSFRBDATA/dsa110-nsfrb-training/simulated/labels.csv` file.

## Usage

The only user-facing script to be used during normal NSFRB operations is `get_status.sh`. To monitor the log files:

```bash
./get_status.sh [wait_time] [num_lines]
```

