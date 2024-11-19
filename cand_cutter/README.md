# Cand Cutter

This folder contains scripts to execute the candidate clusterer and classifier, which checks for new raw candidate files and processes them offline.

## Structure

- `run.py`: Python implementation of candcutter; uses Python `multiprocessing.Queue` to receive new raw candidate files
- `run_cand_cutter`: Bash executable script that calls` run.py` so it can be run in the background
- `kill_candcutter`: Bash executable script that kills candcutter while in the background

## Usage

To run the candcutter:

```bash
./run_candcutter arguments
```

Optional arguments are defined below:

- `--SNRthresh`: SNR threshold, default = 3000
- `--port`: Port number for receiving data from subclient, default = 8843
- `--gridsize`: Expected length in pixels for each sub-band image, default=300
- `--nsamps`: Expected number of time samples (integrations) for each sub-band image, default=25
- `--nchans`: Expected number of sub-band images for each full image, default=16
- `--datasize`: Expected size of each element in sub-band image in bytes,default=8
- `--subimgpix`: Length of image cutouts in pixels, default=11
- `-T`,`--testh23`: Set if using h23 as a test server
- `--maxconnect`: Maximum number of connections accepted by the server, default=16
- `--timeout`: Max time in seconds to wait for more data to be ready to receive, default = 10
- `--model_weights`: Path to the model weights file
- `--verbose`: Enable verbose output
- `--maxProcesses`: Maximum number of images that can be searched at once, default = 5, maximum is 40
- `--headersize`: Number of bytes representing the header; note this varies depending on the data shape, default = 128
- `--usefft`: Implement PSF spatial matched filter as a 2D FFT
- `--cluster`: Enable clustering with HDBSCAN
- `--multithreading`: Enable multithreading in search
- `--nrows`: Number of rows to break image into if multithreading, default = 4 
- `--ncols`: Number of columns to break image into if multithreading, default = 2 
- `--threadDM` : Break DM trials among multiple threads
- `--samenoise`: Assume the noise in each pixel is the same 
- `--cuda`: Uses CUDA to accelerate computation with GPUs via Pytorch and/or JAX. The cuda flag overrides the multithreading option
- `--toslack`: Send candidate summary plots to Slack
- `--PyTorchDedispersion`: Uses GPU-accelerated dedispersion code from https://github.com/nkosogor/PyTorchDedispersion
- `--exportmaps`: Output noise maps for each DM and width trial to the noise directory
- `--initframes`: Initializes previous frames for dedispersion
- `--initnoise`: Initializes noise statistics for S/N estimates
- `--savesearch`: Saves the searched image as a numpy array
- `--appendframe`: Use the previous image to fill in dedispersion search
- `--DMbatches`: Number of pixel batches to submit dedispersion to the GPUs , default=1
- `--usejax`: Use JAX Just-In-Time compilation for GPU acceleration


- `--cutout`: Get image cutouts around each candidate
- `--subimgpix`: Length of image cutouts in pixels, default=11
- `--cluster`: Enable clustering with HDBSCAN
- `--plotclusters`: Plot intermediate plots from HDBSCAN clustering
- `--mincluster`: Minimum number of candidates required to be made a separate HDBSCAN cluster,default=5
- `--minsamples`: Minimum number of candidates to be core point,default=2
- `--verbose`: Enable verbose output
- `--classify`: Classify candidates with a machine learning convolutional neural network
- `--model_weights`: Path to the model weights file
- `--toslack`: Sends Candidate Summary Plots to Slack
- `--sleep`: Time in seconds to sleep between successive cand\_cutter runs; default=0
- `--runtime`: Minimum time in seconds to run before sleep cycle; default=60
- `--maxProcesses`: Maximum number of threads for thread pool; default=5
- `--archive`: Archive candidates on dsastorage
- `--maxcands`: Maximum number of candidates searchable in one iteration. Default is full image, 300x300x5x16=7.2e6
- `--percentile`: Percentile above which to take candidates, e.g. if 90, candidates with s/n in 90th percentile will be clustered. Default 0
- `--SNRthresh`: SNR threshold, default = 10
- `--train`: Save candidate cutouts to the training set for the ML classifier
- `--useTOA`: Include TOAs in clustering algorithm
- `--psfcluster`: PSF-based spatial clustering

To kill candcutter:

```bash
./kill_candcutter
```


