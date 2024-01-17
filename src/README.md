# Client-Server Source Code

This folder contains the source code (in C) and executable for the T4 server. This receives image data from each corr node and pipes them to stdout.

## Structure

The folder is structured as follows:

- `socket_server_test_V3.c': This script opens a socket in listening mode and accepts http `PUT' or `POST' commands from corr nodes carrying image data.

- `socket_server_test_V3.out': This is the executable compiled from `socket_server_test_V3.c'

- `server_log.txt': This records output from the server so that stdout is not contaminated with status output.

- `server_tests'
    - `test_server.py': This script (in Python) tests the `nsfrb.pipeline' module by reading data output from the server to stdout and converting to a numpy array.
    - `test_server.sh': This script (in C) calls `socket_server_test_V3.out' with necessary piping commands to test the server. To send test data, run the corresponding client script in `dsa110-nsfrb/scripts/socket_client_test_PUT.sh'

- `.pipestatus.txt': This is a specialized log file that reports when a given stage of the pipeline has failed, and allows the pipeline to abort without failure.

The folder also contains the relevant CMake files and directories required for compilation:

- CMakeCache.txt  
- CMakeFiles  
- cmake_install.cmake  
- CMakeLists.txt  
- Makefile

## Usage

To compile the server code, run:

```bash
cd /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/server_tests
cmake

To test the server, run:

```bash
cd /home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/server_tests
./test_server.sh

Send data using the `dsa110-nsfrb/scripts/socket_client_test_PUT.sh' script from another corr node. A copy of the script is available on h23 with test data, and can be accessed and run as followed:

```bash
ssh h23
cd /dataz/dsa110/imaging/NSFRB_client
./socket_client_test_PUT.sh simulated1_noisy.npy




