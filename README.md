# dsa110-nsfrb

This repository contains code for the DSA-110 Not-So-Fast Radio Burst (NSFRB) search pipeline, which uses low time resolution (130 ms) radio images to search for transient radio emission on 1-10 second timescales. The pipeline is currently being developed as part of DSA-110 Completion efforts. Below is the offline system design (updated 2024-11-18):

![NSFRBoffline](https://github.com/dsa110/dsa110-nsfrb/blob/development/NSFRB_T4_Offline_System_Diagram.png?raw=True)


Upon verification and testing of the completed offline pipeline, the NSFRB search will be scaled to realtime operation commensal with the FRB search. Below is the realtime system design (updated 2024-11-18):

![NSFRBrealtime](https://github.com/dsa110/dsa110-nsfrb/blob/development/NSFRB_T4_Realtime_System_Diagram.png?raw=True)

Software requirements are listed in `requirements.txt`, and the `casa310nsfrb_env.yml` environment is provided with a subset of required modules installed. To install, first cd to the `dsa110-nsfrb` installation directory. After creating and activating the conda environment using:

```bash
conda env create -f casa310nsfrb_env.yml
conda activate casa310nsfrb_env.yml
```

Before installing `dsa110-nsfrb`, first edit the `setup.py` script and set `CORR_INSTALL` to True if installing on a single corr node as a real-time imager, or False if installing as a process server and post-processing node. Then install by running:

```bash
pip install .
source ~/.bashrc
```

from the bash command line. The second line is only required for the first installation in order to setup the required environment variables. For corr node installation, a service file is provided as `realtime/rt_imager.service`, which should be setup with paths correctly pointing to the current working directory. This can be copied to the e.g. `/etc/systemd/user/` directory to run the real-time imager as a service. 

The following sub-modules are defined:

- `nsfrb`: Helper functions used by the process server and imaging client to run the NSFRB pipeline.
	- `imaging`
	- `searching`
	- `simulating`
	- `classifying`
	- `TXclient`
	- `pipeline`
	- `plotting`
	- `output_logging`
	- `noise`
	- `config`
	- `jax_funcs`
	- `planning`
- `process_server`: Scripts to run the T4 process server, which receives images over HTTP from each correlator node and runs the search and classification pipelines.
	- `run`
	- `run_poc_server`
	- `kill_proc_server`
	- `process_flags`
- `cand_cutter`: Scripts to run the offline candidate clusterer and ML classifier, and send most likely candidates to slack.
	- `run`
	- `run_cand_cutter`
	- `kill_cand_cutter`
- `simulations_and_classifications`: Scripts for simulating RFI and source images with the DSA-110 baseline coverage.
	- `generate_rfi_images`
	- `generate_source_images`
	- `generate_offset_source_images`
	- `generate_PSF_images`
	- `rfi_classification_pytorch.ipynb`
	- `model_weights.pth`
	- `png_to_npy`
- `inject`: Scripts for injecting bursts to the NSFRB pipeline
	- `run_injector`
	- `kill_injector`
	- `inject_burst_image`
	- `injecting`
- `offline`: Service routines to facilitate copying fast visibilities for the offline NSFRB system and imaging them.
	- `run`
	- `offline_imager`
	- `cp_data`
	- `clearcands`
	- `clearvis`
- `scripts': Test scripts and jupyter notebooks
	- `get_status`
	- `ms_vis_extraction`
	- `SearchAlgorithmTuning_V1.ipynb`
	- `clear_casa_logs.sh`
	- `init_noise_from_fast_vis_UVW`
	- `init_noise_fnames`
	- `noise_testing`
	- `sort_training_set`
	- `snr_calibration`

External folders containing log files (`dsa110-nsfrb-logfiles`), candidates (`dsa110-nsfrb-candidates`), image frames (`dsa110-nsfrb-frames`), and fast visibilities (`dsa110-nsfrb-fast-visibilities) are created upon installation.

Other Files:

- `DSA110_Station_Coordinates.csv`: Locations of DSA-110 antennas
- `metadata.txt`: Metadata (including working directory) for internal use
- `casa38nsfrb_env.yml`: yaml file to create NSFRB Python 3.8 environment (deprecated)
- `casa310nsfrb_env.yml`: yaml file to create NSFRB Python 3.10 environment.

This effort is conducted by Myles Sherman, Nikita Kosogorov, Casey Law, Vikram Ravi, Liam Connor, and the DSA-110 Team.
