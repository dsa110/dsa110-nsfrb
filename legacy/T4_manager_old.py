from dsautils import dsa_store
import json
import os
import sys
import numpy as np
from astropy.time import Time
from nsfrb.config import tsamp,baseband_tsamp,tsamp_slow,tsamp_imgdiff
from nsfrb.outputlogging import printlog
from event import event
from dsaT4 import data_manager
from dask.distributed import Client, Lock
from copy import deepcopy
from itertools import chain
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import MappingProxyType
from typing import Union

from astropy.time import Time
import astropy.units as u
from dsautils import cnf
from dsautils import dsa_syslog as dsl
"""
Functions for converting cand cutter output into T4 triggers and json files
compatible with DSA-110 event scheduler
"""

#client = Client('10.41.0.254:8781')
client = Client('10.42.0.232:8786')
LOCK = Lock('update_json')
ds = dsa_store.DsaStore()
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsaT4")
LOGGER.function("T4_manager")
#dc = alert_client.AlertClient('dsa')

TIMEOUT_FIL = 600
FILPATH = os.environ["DSA110DIR"] + "operations/T1/"
OUTPUT_PATH = os.environ["DSA110DIR"] + "operations/T4/"
IP_GUANO = '3.13.26.235'

final_cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/final_cands/candidates/"
def nsfrb_to_json(cand_isot,mjds,snr,width,dm,ra,dec,trigname,final_cand_dir=final_cand_dir,slow=False,imgdiff=False):
    """
    Takes the following arguments and saves to a json file in the specified cand dir
    cand_isot: str
    snr: float
    width: int
    dm: float
    ra: float
    dec: float
    trigname: str
    """
    #mjds = Time(cand_isot,format='isot').mjd
    if slow:
        ibox = int(np.ceil(width*tsamp_slow/baseband_tsamp))
    elif imgdiff:
        ibox = int(np.ceil(width*tsamp_imgdiff/baseband_tsamp))
    else:
        ibox = int(np.ceil(width*tsamp/baseband_tsamp))
    f = open(final_cand_dir + "/" + trigname + ".json","w")
    json.dump({"mjds":mjds,
               "isot":cand_isot,
               "snr":snr,
               "ibox":ibox,
               "dm":dm,
               "ibeam":-1,
               "cntb":-1,
               "cntc":-1,
               "specnum":-1,
               "ra":ra,
               "dec":dec,
               "trigname":trigname},f)
    f.close()

    return final_cand_dir + "/" + trigname + ".json"


LOCK = Lock('update_json')
def submit_cand_nsfrb(fl, logfile,lock=LOCK):
    """
    Modelled from dsa110-T3/dsaT3/T3_manager.submit_cand(); Given filename of trigger json,
    create DSACand and submit to scheduler for T3 processing
    """
    
    d = event.create_event(fl)
    printlog(f"Submitting task for trigname {d.trigname}",logfile)


    d_cs = client.submit(run_createstructure_nsfrb, d, key=f"run_createstructure_nsfrb-{d.trigname}", lock=lock, priority=1)  # create directory structure
#    d_bf = client.submit(run_burstfit, d_fp, key=f"run_burstfit-{d.trigname}", lock=lock, priority=1)  # burstfit model fit
    d_vc = client.submit(run_voltagecopy_nsfrb, d_cs, key=f"run_voltagecopy_nsfrb-{d.trigname}", lock=lock)  # copy voltages
    d_h5 = client.submit(run_hdf5copy_nsfrb, d_cs, key=f"run_hdf5copy_nsfrb-{d.trigname}", lock=lock)  # copy hdf5
    d_fm = client.submit(run_fieldmscopy_nsfrb, d_cs, key=f"run_fieldmscopy_nsfrb-{d.trigname}", lock=lock)  # copy field image MS
#    d_hr = client.submit(run_hires, (d_bf, d_vc), key=f"run_hires-{d.trigname}", lock=lock)  # create high resolution filterbank
#    d_cm = client.submit(run_candidatems, (d_bf, d_vc), key=f"run_candidatems-{d.trigname}", lock=lock)  # make candidate image MS
#    d_po = client.submit(run_pol, d_hr, key=f"run_pol-{d.trigname}", lock=lock)  # run pol analysis on hires filterbank
#    d_hb = client.submit(run_hiresburstfit, d_hr, key=f"run_hiresburstfit-{d.trigname}", lock=lock)  # run burstfit on hires filterbank
#    d_il = client.submit(run_imloc, d_cm, key=f"run_imloc-{d.trigname}", lock=lock)  # run image localization on candidate image MS
#    d_as = client.submit(run_astrometry, (d_fm, d_cm), key=f"run_astrometry-{d.trigname}", lock=lock)  # astrometric burst image
#    fut = client.submit(run_final, (d_h5, d_po, d_hb, d_il, d_as), key=f"run_final-{d.trigname}", lock=lock)
    fut = client.submit(run_final_nsfrb, (d_h5, d_fm, d_vc), key=f"run_final_nsfrb-{d.trigname}", lock=lock)

    return fut


def run_createstructure_nsfrb(d, lock=None):
    """ Use DSACand (after filplot) to decide on creating/copying files to candidate data area.
    """

    if d.real and not d.injected:
        print("Running createstructure for real/non-injection candidate.")

        # TODO: have DataManager parse DSACand
        dm = data_manager.NSFRBDataManager(d.__dict__)
        # TODO: have update method accept dict or DSACand
        d.update(dm())

    else:
        print("Not running createstructure for non-astrophysical candidate.")

    d.writejson(outpath=OUTPUT_PATH, lock=lock)
    return d

def run_voltagecopy(d, lock=None):
    """ Given DSACand (after filplot), copy voltage files.
    """

    if d.real and not d.injected:
        print('Running voltagecopy on {0}'.format(d.trigname))
        LOGGER.info('Running voltagecopy on {0}'.format(d.trigname))
        dm = data_manager.NSFRBDataManager(d.__dict__)
        dm.copy_voltages()

        # TODO: have update method accept dict or DSACand
        d.update(dm.candparams)
        d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d

def run_hires(ds, lock=None):
    """ Given DSACand objects from burstfit and voltage, generate hires filterbank files.
    """

    d, d_vc = ds
    d.update(d_vc)

    print('placeholder run_hires on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_hires on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_pol(d, lock=None):
    """ Given DSACand (after hires), run polarization analysis.
    Returns updated DSACand with new file locations?
    """

    print('placeholder nrun_pol on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_pol on {0}'.format(d.trigname))

#    if d_hr['real'] and not d_hr['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_fieldmscopy(d, lock=None):
    """ Given DSACand (after filplot), copy field MS file.
    Returns updated DSACand with new file locations.
    """

    print('placeholder run_fieldmscopy on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_fieldmscopy on {0}'.format(d.trigname))

#    if d_fp['real'] and not d_fp['injected']:
#        dm = data_manager.DataManager(d_fp)
#        dm.link_field_ms()
#        update_json(dm.candparams, lock=lock)
#        return dm.candparams
#    else:
    return d


def run_candidatems(ds, lock=None):
    """ Given DSACands from filplot and voltage copy, make candidate MS image.
    Returns updated DSACand with new file locations.
    """

    d, d_vc = ds
    d.update(d_vc)

    print('placeholder run_candidatems on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_candidatems on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_hiresburstfit(d, lock=None):
    """ Given DSACand, run highres burstfit analysis.
    Returns updated DSACand with new file locations.
    """

    print('placeholder run_hiresburstfit on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_hiresburstfit on {0}'.format(d.trigname))

#    if d_hr['real'] and not d_hr['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_imloc(d, lock=None):
    """ Given DSACand (after candidate image MS), run image localization.
    """

    print(f'Running localization on {d.trigname}')
    LOGGER.info(f'Running localization on {d.trigname}')

# TODO: is this the first sent or an update with good position?
#    if d.real and not d.injected:
#        dc.set('observation', args=asdict(d))

    d.writejson(outpath=OUTPUT_PATH, lock=lock)
    return d


def run_astrometry(ds, lock=None):
    """ Given field image MS and candidate image MS, run astrometric localization analysis.
    """

    d, d_cm = ds
    d.update(d_cm)

    print('placeholder run_astrometry on {0}'.format(d.trigname))
    LOGGER.info('placeholder run_astrometry on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_final(ds, lock=None):
    """ Reduction task to handle all final tasks in graph.
    May also update etcd to notify of completion.
    """

#    d, d_po, d_hb, d_il, d_as = ds
    d, d_fm, d_vc = ds
    d.update(d_fm)
    d.update(d_vc)
#    d.update(d_il)
#    d.update(d_as)

    print('Final merge of results for {0}'.format(d.trigname))
    LOGGER.info('Final merge of results for {0}'.format(d.trigname))

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def wait_for_local_file(fl, timeout, allbeams=False):
    """ Wait for file named fl to be written. fl can be string filename of list of filenames.
    If timeout (in seconds) exceeded, then return None.
    allbeams will parse input (str) file name to get list of all beam file names.
    """

    if allbeams:
        assert isinstance(fl, str), 'Input should be detection beam fil file'
        loc = os.path.dirname(fl)
        fl0 = os.path.basename(fl.rstrip('.fil'))
        fl1 = "_".join(fl0.split("_")[:-1])
        fl = [f"{os.path.join(loc, fl1 + '_' + str(i) + '.fil')}" for i in range(512)]

    if isinstance(fl, str):
        fl = [fl]
    assert isinstance(fl, list), "name or list of fil files expected"

    elapsed = 0
    while not all([os.path.exists(ff) for ff in fl]):
        time.sleep(5)
        elapsed += 5
        if elapsed > timeout:
            return None
        elif elapsed <= 5:
            print(f"Waiting for {len(fl)} files, like {fl[0]}...")

    return fl
