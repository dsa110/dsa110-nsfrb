# Process Server

This folder contains scripts to execute the persistent process server, which receives images over HTTP connections from the image clients on each DSA-110 correlator node, and starts the search and classification pipelines.

## Structure

- `run.py`: Python implementation of process server; uses HTTP to receive data and `concurrent.futures` module to multithread.
- `run_proc_server`: Bash executable script that calls `run.py` so it can be run in the background
- `kill_proc_server`: Bash executable script that kills process server while in the background
- `process_flags.txt`: 8-bit flag storing the receipt status of each HTTP request; this is sent back to the client to prompt a re-send if needed

## Usage

To run the process server:

```bash
./run_proc_server arguments
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

To kill process server:

```bash
./kill_proc_server
```


