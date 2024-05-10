# dsa110-nsfrb

This repository contains code for the DSA-110 Not-So-Fast Radio Burst (NSFRB) search pipeline, which uses low time resolution (130 ms) radio images to search for transient radio emission on 1-10 second timescales. The pipeline is currently being developed as part of DSA-110 Completion efforts.

Software requirements are listed in `requirements.txt`, and the module can be installed by runnning:

```bash
python setup.py install
```

from the bash command line. dsa110-nsfrb contains four primary modules:
- `nsfrb.imaging`: This file contains helper functions for imaging of fast visibility data. Radio images will be formed on each of 16 corr nodes that handle 15.6 MHz each. These will be sent to T4 (implemented on corr20) for combination and searching
- `nsfrb.searching`: This file contains helper functions to search for NSFRBs in RA, Declination, DM, and pulse-width space. DM is searched up to 4000 pc/cc and pulse width up to 3.25 s (25 samples). Identified candidates are clustered in position then used to obtain 10x10 pixel cutouts (for all DM, pulse width trials) around the candidate.
- `nsfrb.pipeline`: This module contains helper functions for managing data streaming among the imaging, searching and classification subsystems.
- `nsfrb.simulating`: This module contains helper functions for RFI and source simulation.
- `nsfrb.classifying`: This module contains the following scripts:
    - `NumpyImageCubeDataset`: A dataset class for loading and processing image cube batches. It supports dynamic transformations and preprocessing, including image resizing to a uniform dimension.
    - `EnhancedCNN`: Model architecture (see more details in `simulations_and_classifications`).
    - `classify_images`: A function designed for classifying images. It returns both binary predictions and their corresponding probabilities.

`config.py` contains relevant parameters and `logging.py` manages logging of output from each subsystem. 

User-facing scripts are contained in the `scripts` directory, including commands to start the T4 server, send data via corr node clients, and run the search and classification systems. See `scripts/README.md`, `simulations_and_classifications/README.md` and `src/README.md` for details on exection. The simplest usage is to start the T4 search pipeline, which is done using the following commands:

```bash
cd dsa110-nsfrb/scripts
./run_search.sh | ./run_classifier.sh
```
This effort is conducted by Myles Sherman, Nikita Kosogorov, Casey Law, Vikram Ravi, Liam Connor, and the DSA-110 Team.
