# Realtime

This folder contains the main script for the realtime imager and C helper libraries for reading and writing fast visibilities with MMMIO.

## Structure

-`realtime_imager`: This defines the main realtime imager routine which images memory-mapped visibilities on a given sub-band and sends them to the process server for offline searching.
-`realtime_injector`: This defines the realtime injector service; this will be run on h24 and produce injections that are copied to the corr nodes to be injected into fast visibilities by the realtime imager
-`rtreader/`: Custom Python C Extension used to read memory-mapped data into a Python bytes structure.
-`rtwriter/`: Custom C library used to write fast visibilities to a memory mapped byte stream.


## Usage

The realtime imager should only be deployed on a single corr node. To run, use e.g.:

```bash
python realtime_imager.py 1 --verbose --num_time_samples 25 --nchans_per_node 8 --gridsize 301 --briggs --robust 2 --save
```

Use the --help flag to see all available command line arguments.

The realtime injector will be run on h24 as:

```bash
python realtime_injector
```

To import the `rtreader` module in Python:

```bash
import rtreader
rtreader.read(shmid,datasize)
```

where `shmid` and `datasize` can be pulled from the etcd key `/mon/nsfrb/fastvis`.

To use the `rtwriter` library in C:

```bash
#include "rtwriter.h"

//create an rtobj object
struct rtwriter_obj *rtobj = malloc(sizeof *rtobj);
//example data
char *buffer = "helloworld";
size_t buffersize = 11;
//write to memory mapped stream
rtwrite(buffer,buffersize,0,rtobj);
//close stream
rtwrite(NULL,0,1,rtobj);
free(rtobj)

```

