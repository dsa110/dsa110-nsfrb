# Inject

This folder contains scripts to inject NSFRBs into the pipeline via the TX client. It uses a PSF generated by the `simulations_and_classifications` pipeline.

## Structure

- `inject_burst_image.py`: This script creates a specified number of NSFRBs and sends them to the process server.

## Usage

To run the burst injector:

```bash
python inject_burst_image.py arguments
```

Optional arguments are defined below:

- `--SNR`: SNR of injected burst, default = 100
- `--port`: Port number for sending injected burst, default = 8080
- `--gridsize`: Length in pixels for each sub-band image, default=300
- `--nsamps`: Number of time samples (integrations) for each sub-band image, default=25
- `--nchans`: Number of sub-band images for each full image, default=16
- `--width`: Width of the injected burst in samples, default = 4
- `--DM`: Dispersion measure of injected burst in pc/cc, default = 0
- `--verbose`: Enable verbose output
- `--nbursts`: Number of injected bursts; default = 1; if > 1, the SNR, width, and DM are drawn from normal distributions centered on the provided values, default=1

