# NSFRB


This folder defines sub-modules for the NSFRB pipeline, which consists of forming images, sending them to the T4 process server, searching each image, and classifying with a convolutional neural network.

## Structure

- `imaging`: This file contains helper functions for imaging of fast visibility data. Radio images will be formed on each of 16 corr nodes that handle 15.6 MHz each. These will be sent to T4 (implemented on corr20) for combination and searching
- `searching`: This file contains helper functions to search for NSFRBs in RA, Declination, DM, and pulse-width space. DM is searched up to 4000 pc/cc and pulse width up to 3.25 s (25 samples). Identified candidates are clustered in position then used to obtain 10x10 pixel cutouts (for all DM, pulse width trials) around the candidate.
- `pipeline[DEPRECATED]`: This module contains helper functions for managing data streaming among the imaging, searching and classification subsystems.
- `simulating`: This module contains helper functions for RFI and source simulation.
- `classifying`: This module contains functions to run and train the Convolutional Neural Network (CNN) used to identify RFI
	- `NumpyImageCubeDataset`: A dataset class for loading and processing image cube batches. It supports dynamic transformations and preprocessing, including image resizing to a uniform dimension.
    	- `EnhancedCNN`: Model architecture (see more details in `simulations_and_classifications`).
    	- `classify_images`: A function designed for classifying images. It returns both binary predictions and their corresponding probabilities.
- `TXclient`: This module contains functions send image data over http from the correlator nodes to the T4 system
- `plotting`: This module contains plotting functions for UV baselines and dirty images
- `config.py`: This contains relevant parameters and constants for the NSFRB search
- `outputlogging.py`: This manages logging of output from each subsystem. Most log files (aside from the process server logs) are in the `dsa110-nsfrb/tmpoutput` directory.

## Usage

- Each sub-module can be imported for use in a script with:

```python
from nsfrb import *module_name*
```
