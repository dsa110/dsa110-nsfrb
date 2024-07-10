# dsa110-nsfrb

This repository contains code for the DSA-110 Not-So-Fast Radio Burst (NSFRB) search pipeline, which uses low time resolution (130 ms) radio images to search for transient radio emission on 1-10 second timescales. The pipeline is currently being developed as part of DSA-110 Completion efforts.

Software requirements are listed in `requirements.txt`, and the module can be installed by runnning:

```bash
pip install .
```

from the bash command line. The following sub-modules are defined:

- `nsfrb`: Helper functions used by the process server and imaging client to run the NSFRB pipeline.
	- `imaging`
	- `searching`
	- `simulating`
	- `classifying`
	- `TXclient`
	- `plotting`
	- `output_logging`
	- `noise`
	- `config`
	- `jax_funcs`
- `process_server`: Scripts to run the T4 process server, which receives images over HTTP from each correlator node and runs the search and classification pipelines.
	- `run_COMBINED`
	- `run_poc_server`
	- `kill_proc_server`
	- `process_flags`
- `simulations_and_classifications`: Scripts for simulating RFI and source images with the DSA-110 baseline coverage.
	- `generate_rfi_images`
	- `generate_source_images`
	- `rfi_classification_pytorch.ipynb`
- `inject`: Scripts for injecting bursts to the NSFRB pipeline
	- `inject_burst_image`
- `offline`: Service routines to facilitate copying fast visibilities for the offline NSFRB system.
	- `getfastvisservice`
	- `clearvis`
- `scripts': Test scripts and jupyter notebooks
	- `get_status`
	- `ms_vis_extraction`
	- `SearchAlgorithmTuning_V1.ipynb`
	- `socket_client_test_POST/PUT`
	- `run_search` [DEPRECATED]
	- `run_classifier` [DEPRECATED]
	- `run_NSFRB` [DEPRECATED]
	- `kill_NSFRB` [DEPRECATED]
External folders containing log files (`dsa110-nsfrb-logfiles`), candidates (`dsa110-nsfrb-candidates`), image frames (`dsa110-nsfrb-frames`), and fast visibilities (`dsa110-nsfrb-fast-visibilities) are created upon installation.

Other Files:

- `DSA110_Station_Coordinates.csv`: Locations of DSA-110 antennas
- `metadata.txt`: Metadata (including working directory) for internal use
- `casa38nsfrb_env.yml`: yaml file to create NSFRB Python 3.8 environment (deprecated)
- `casa310nsfrb_env.yml`: yaml file to create NSFRB Python 3.10 environment. Run with:

```bash
conda env create -f casa10nsfrb_env.yml
```

This effort is conducted by Myles Sherman, Nikita Kosogorov, Casey Law, Vikram Ravi, Liam Connor, and the DSA-110 Team.
