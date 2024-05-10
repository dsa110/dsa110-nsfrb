# Scripts 

This folder contains scripts to start and monitor the DSA-110 NSFRB search pipeline.

## Structure

The folder is structured as follows:

- `run_search.sh`: This script clears log files and opens a socket in listening mode to accept http `PUT` or `POST` commands from corr nodes carrying image data. Received data is piped directly to the `run_search.py` script which executes the de-dispersion search pipeline.
- `run_search.py`: This script receives image data piped from the server and converts them to numpy image tesseract arrays. It calls the `nsfrb.searching.run_search` function to start the de-dispersion search, which outputs a list of candidates' RA, DEC, DM, pulse width, and SNR. These are passed to `nsfrb.searching.get_subimage` to get image cutouts which are converted to a hex string and piped stdout.

- `run_classifier.sh`: This script starts the ML classifier script which data is piped to from stdin.
- `run_clssifier.py`: This script receives image cutouts for NSFRB candidates piped from `run_search.py` and converts them to numpy arrays. [DETAILS TBD]

[TODO: ADD ARGUMENT OPTIONS]

- `script_warnings`
    - `searchlog_warnings.txt`: This records stderr output piped from the `run_search.py` script
    - `classlog_warnings.txt`: This records stderr output piped from the `run_classifier.py` script

- `script_flags`
    - `searchlog_flags.txt`: This records data info about data piped from `run_search.py` to `run_classifier.py` needed to convert raw hex to numpy array. This includes `datasize` (int specifying the total number of bytes output from `run_search.py`), `outputshape` (tuple specifying the shape of the numpy array), and `size` (int specifying the number of bits for each entry in the numpy array); each argument is delimited by a `;`.

- `get_status.sh`: This function reads log files currently in the `dsa110-nsfrb/tmpoutput` folder and outputs them to terminal for periodic monitoring, taking the wait time between updates in seconds as the argument

- `socket_client_test_PUT.sh`: This script uses curl (https://curl.se/) to take an input datafile and send using an http PUT command to the T4 server. 
- `socket_client_test_POST.sh` [deprecated] : This script uses curl (https://curl.se/) to take an input datafile and send using an http POST command to the T4 server. This script has been replaced by the more reliable and efficient `socket_client_test_PUT.sh`.


## Usage

To initiate the search and classification pipeline, run the following commands:

```bash
cd /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts
./run_search | ./run_classifier
```

To monitor the output, open a separated terminal and run:
```bash
cd /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts
./get_status.sh [time]
```
where [time] is the time between status updates in seconds (recommend 10-15 seconds). Stop the monitor with Ctrl+c

To send data using the client script, run:

```bash
cd /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts
./socket_client_test_PUT.sh [datafile.npy]
```
where [datafile.npy] contains image data as a numpy array saved to file using the python command `numpy.save(img,filename)` (see https://numpy.org/doc/stable/reference/generated/numpy.save.html).
