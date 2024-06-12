"""A service to preprcocess hdf5 files before calibration.
"""
import datetime
import sys
import warnings
from multiprocessing import Process, Queue
import time

import pandas
import h5py
import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u

import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
from dsautils import cnf

import dsacalib.constants as ct
from dsacalib.preprocess import rsync_file, first_true
from dsacalib.preprocess import update_caltable
from dsacalib.utils import exception_logger

# make sure warnings do not spam syslog
warnings.filterwarnings("ignore")

# TODO: Get these parameters from a function, rather tahn as defaults
CONF = cnf.Conf(use_etcd=True)
CORR_CONF = CONF.get('corr')
CAL_CONF = CONF.get('cal')
MFS_CONF = CONF.get('fringe')
CORRLIST = list(CORR_CONF['ch0'].keys())
CORRLIST = [CORRLIST[i][6:] for i in range(len(CORRLIST))]
print(CORRLIST)
NCORR = len(CORRLIST)
CALTIME = CAL_CONF['caltime_minutes']*u.min
FILELENGTH = MFS_CONF['filelength_minutes']*u.min
HDF5DIR = CAL_CONF['hdf5_dir']
NSFRBHDF5DIR = "/home/ubuntu/msherman_nsfrb/dsa110-nsfrb-fast-visibilities/"

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# ETCD interface
ETCD = ds.DsaStore()

# FIFO Queues for rsync, freq scrunching, calibration
FSCRUNCH_Q = Queue()
RSYNC_Q = Queue()
GATHER_Q = Queue()
ASSESS_Q = Queue()
CALIB_Q = Queue()

# Maximum number of files per correlator that can be assessed for calibration
# needs at one time.
MAX_ASSESS = 4

# Maximum amount of time that gather_files will wait for all correlator files
# to be gathered, in seconds
MAX_WAIT = 5*60

# Time to sleep if a queue is empty before trying to get an item
TSLEEP = 10

def _update_caltable_callback(etcd_dict):
    """When the antennas are moved, make and read a new calibration table.
    """
    if etcd_dict['cmd'] == 'move':
        pt_el = etcd_dict['val']*u.deg
        update_caltable(pt_el)

def populate_queue(etcd_dict, queue=RSYNC_Q, hdf5dir=NSFRBHDF5DIR):
    """Populates the fscrunch and rsync queues using etcd.

    Etcd watch callback function.
    """
    print("populating queue:",etcd_dict)
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']
    if cmd == 'rsync':
        rsync_string = (
            f"{val['hostname']}.pro.pvt:{val['filename']} "
            f"{hdf5dir}/{val['hostname']}/"
        )
        queue.put(rsync_string)

def task_handler(task_fn, inqueue, outqueue=None):
    """Handles in and out queues of preprocessing tasks.

    Parameters
    ----------
    task_fn : function
        The function to execute, with a single argument.
    inqueue : multiprocessing.Queue instance
        The queue containing the arguments to `task_fn`.
    outqueue : multiprocessing.Queue instance
        The queue to write the otuput of `task_fn` to.
    """
    while True:
        if not inqueue.empty():
            fname = inqueue.get()
            try:
                fname = task_fn(fname)
                if outqueue is not None:
                    outqueue.put(fname)
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    f"preprocessing of file {fname}",
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)

def gather_worker(inqueue, outqueue, corrlist=None):
    """Gather all files that match a filename.

    Will wait for a maximum of 15 minutes from the time the first file is
    received.

    Parameters
    ----------
    inqueue : multiprocessing.Queue instance
        The queue containing the filenames, max size of 16 (i.e. one file per
        corr node).
    outqueue : multiprocessing.Queue instance
        The queue in which to place the gathered files (as a list).
    """
    if not corrlist:
        corrlist = CORRLIST
    ncorr = len(corrlist)
    filelist = [None]*ncorr
    nfiles = 0
    # Times out after 15 minutes
    end = time.time() + 60*15
    while nfiles < ncorr and time.time() < end:
        if not inqueue.empty():
            fname = inqueue.get()
            corrid = fname.replace('//', '/').split('/')[5]
            filelist[corrlist.index(corrid)] = fname
            nfiles += 1
        time.sleep(1)
    outqueue.put(filelist)

def gather_files(inqueue, outqueue, ncorr=NCORR, max_assess=MAX_ASSESS, tsleep=TSLEEP):
    """Gather files from all correlators.

    Will wait for a maximum of 15 minutes from the time the first file is
    received.

    Parameters
    ----------
    inqueue : multiprocessing.Queue instance
        The queue containing the ungathered filenames .
    outqueue : multiprocessing.Queue instance
        The queue in which to place the gathered files (as a list).
    """
    gather_queues = [Queue(ncorr) for idx in range(max_assess)]
    gather_names = [None]*max_assess
    gather_processes = [None]*max_assess
    nfiles_assessed = 0
    while True:
        if not inqueue.empty():
            try:
                fname = inqueue.get()
                print(fname)
                if not fname.split('/')[-1][:-7] in gather_names:
                    gather_names[nfiles_assessed%max_assess] = \
                        fname.split('/')[-1][:-7]
                    gather_processes[nfiles_assessed%max_assess] = \
                        Process(
                            target=gather_worker,
                            args=(
                                gather_queues[nfiles_assessed%max_assess],
                                outqueue
                            ),
                        daemon=True
                        )
                    gather_processes[nfiles_assessed%max_assess].start()
                    nfiles_assessed += 1
                gather_queues[
                    gather_names.index(fname.split('/')[-1][:-7])
                ].put(fname)
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    f"preprocessing of file {fname}",
                    exc,
                    throw=False
                )
        else:
            time.sleep(tsleep)

def assess_file(inqueue, outqueue, caltime=CALTIME, filelength=FILELENGTH):
    """Decides whether calibration is necessary.

    Sends a command to etcd using the monitor point /cmd/cal if the file should
    be calibrated.

    Parameters
    ----------
    inqueue : multiprocessing.Queue instance
        The queue containing the gathered filenames.
    outqueue : multiprocessing.Queue instance
        The queue to which the calname and gathered filenames (as a tuple) if
        the file is appropriate for calibration.
    caltime : astropy quantity
        The amount of time around the calibrator to be converted to
        a measurement set for calibration. Used to assess whether any part of
        the desired calibrator pass is in a given file.
    """
    while True:
        if not inqueue.empty():
            try:
                flist = inqueue.get()
                fname = first_true(flist)
                datet = fname.split('/')[-1][:19]
                tstart = Time(datet).sidereal_time(
                    'apparent',
                    longitude=ct.OVRO_LON*u.rad
                )
                tend = (Time(datet)+filelength).sidereal_time(
                    'apparent',
                    longitude=ct.OVRO_LON*u.rad
                )
                a0 = (caltime*np.pi*u.rad/
                      (ct.SECONDS_PER_SIDEREAL_DAY*u.s)).to_value(u.rad)
                with h5py.File(fname, mode='r') as h5file:
                    pt_dec = h5file['Header']['extra_keywords']['phase_center_dec'].value*u.rad
                caltable = update_caltable(pt_dec)
                calsources = pandas.read_csv(caltable, header=0)
                for _index, row in calsources.iterrows():
                    if isinstance(row['ra'], str):
                        rowra = row['ra']
                    else:
                        rowra = row['ra']*u.deg
                    delta_lst_start = (
                        tstart-Angle(rowra)
                    ).to_value(u.rad)%(2*np.pi)
                    if delta_lst_start > np.pi:
                        delta_lst_start -= 2*np.pi
                    delta_lst_end = (
                        tend-Angle(rowra)
                    ).to_value(u.rad)%(2*np.pi)
                    if delta_lst_end > np.pi:
                        delta_lst_end -= 2*np.pi
                    if delta_lst_start < a0 < delta_lst_end:
                        calname = row['source']
                        print(f"Calibrating {calname}")
                        outqueue.put((calname, flist))
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    f"preprocessing of file {fname}",
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)

if __name__=="__main__":
    processes = {
        'rsync': {
            'nthreads': 1,
            'task_fn': rsync_file,
            'queue': RSYNC_Q,
            'outqueue': GATHER_Q, #FSCRUNCH_Q,
            'processes': []
        },
        # 'fscrunch': {
        #     'nthreads': 4,
        #     'task_fn': fscrunch_file,
        #     'queue': FSCRUNCH_Q,
        #     'outqueue': GATHER_Q,
        #     'processes': []
        # },
    }
    

    # Start etcd watch
    #ETCD.add_watch('/cmd/cal', populate_queue)

    #Add a new watch for nsfrbvis key?
    ETCD.add_watch('/cmd/nsfrbvis', populate_queue)#lambda etcd_dict: populate_queue(etcd_dict, hdf5dir=NSFRBHDF5DIR))



    ETCD.put_dict(f"/cmd/nsfrbvis",
            {
                "cmd":"rsync",
                "val":{"hostname":'h04',
                        "filename":"/home/ubuntu/nsfrb/*hdf5"}
            }
        )

    # Start all threads
    for name, pinfo in processes.items():
        for i in range(pinfo['nthreads']):
            pinfo['processes'] += [Process(
                target=task_handler,
                args=(
                    pinfo['task_fn'],
                    pinfo['queue'],
                    pinfo['outqueue'],
                ),
                daemon=True
            )]
        for pinst in pinfo['processes']:
            pinst.start()

    try:
        processes['gather'] = {
            'nthreads': 1,
            'task_fn': gather_files,
            'queue': GATHER_Q,
            'outqueue': ASSESS_Q,
            'processes': []
        }
        processes['gather']['processes'] += [Process(
            target=gather_files,
            args=(
                GATHER_Q,
                ASSESS_Q
                )
        )]
        processes['gather']['processes'][0].start()

        processes['assess'] = {
            'nthreads': 1,
            'task_fn': assess_file,
            'queue': ASSESS_Q,
            'outqueue': CALIB_Q,
             'processes': []
        }
        processes['assess']['processes'] += [Process(
            target=assess_file,
            args=(
                ASSESS_Q,
                CALIB_Q
            ),
            daemon=True
        )]
        processes['assess']['processes'][0].start()

        while True:
            
            for name, pinfo in processes.items():
                print(name,pinfo,[pinst.is_alive() for pinst in
                            pinfo['processes']])
                ETCD.put_dict(
                    f'/mon/nsfrb/{name}_process',
                    {
                        "queue_size": pinfo['queue'].qsize(),
                        "ntasks_alive": sum([
                            pinst.is_alive() for pinst in
                            pinfo['processes']
                        ]),
                        "ntasks_total": pinfo['nthreads']
                    }
                )
            
            
            ETCD.put_dict(
                '/mon/service/nsfrbpreprocess',
                {
                    "cadence": 60,
                    "time": Time(datetime.datetime.utcnow()).mjd
                }
            )
            print("/mon/service/nsfrbpreprocess")
            """
            while not CALIB_Q.empty():
                (calname_fromq, flist_fromq) = CALIB_Q.get()
                ETCD.put_dict(
                    '/cmd/cal',
                    {
                        'cmd': 'calibrate',
                        'val': {
                            'calname': calname_fromq,
                            'flist': flist_fromq
                        }
                    }
                )

            """
            print("ASSESS_Q",ASSESS_Q.empty())
            print("GATHER_Q",GATHER_Q.empty())
            print("CALIB_Q",CALIB_Q.empty())
            while not CALIB_Q.empty():
                print("CALIB_Q NOT EMPTY:" ,CALIB_Q)
                (calname_fromq, flist_fromq) = CALIB_Q.get()
                ETCD.put_dict(
                    '/cmd/nsfrbvis',
                    {
                        'cmd': 'copy',
                        'val': {
                            'calname': calname_fromq,
                            'flist': flist_fromq
                        }
                    }
                )
            
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        processes['gather']['processes'][0].terminate()
        processes['gather']['processes'][0].join()
        sys.exit()
