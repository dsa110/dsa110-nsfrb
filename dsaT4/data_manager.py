"""Data manager and related functions for managing Level 1 data in T4."""
from copy import deepcopy
from itertools import chain
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import MappingProxyType
from typing import Union
import os
from astropy.time import Time
import astropy.units as u
from dsautils import cnf
from dsautils import dsa_syslog as dsl

OPS_PATH = os.environ["DSA110DIR"] + "operations/"
VOLTAGE_PATH = os.environ["DSA110DIR"] + "T3/"
CANDS_PATH = os.environ["NSFRBDATA"] + "dsa110-nsfrb-candidates/"
class NSFRBDataManager:
    """Manage Level 1 data for confirmed candidates."""

    operations_dir = Path(OPS_PATH)
    candidates_dir = Path(CANDS_PATH + "final_candidates/candidates")
    candidates_subdirs = (
        "Level3", "Level2/voltages", "Level2/filterbank", "Level2/calibration", "other")
    try:
        subband_corrnames = tuple(cnf.Conf().get('corr')['ch0'].keys())
    except:
        subband_corrnames = None
    nbeams = 512
    voltage_dir = Path(VOLTAGE_PATH) 

    # Ensure read only since shared between instances
    directory_structure = MappingProxyType({
        'voltages':
            {
                'target': "{voltage_dir}/{hostname}/{candname}_data.out",
                'target_scp': "ubuntu@{hostname}.sas.pvt:/home/ubuntu/data/{candname}_data.out",
                'destination': (
                    "{candidates_dir}/{isot}/{candname}/Level2/voltages/{candname}_{subband}_data.out"),
            },
        'voltages_headers':
            {
                'target': "{voltage_dir}/{hostname}/{candname}_header.json",
                'target_scp': "ubuntu@{hostname}.sas.pvt:/home/ubuntu/data/{candname}_header.json",
                'destination': (
                    "{candidates_dir}/{isot}/{candname}/Level2/voltages/{candname}_{subband}_header.json"),
            },
        'filterbank':
            {
                'target': "{operations_dir}/T1/{candname}/{candname}_{beamnumber}.fil",
                'destination': (
                    "{candidates_dir}/{isot}/{candname}/Level2/filterbank/{candname}_{beamnumber}.fil"),
            },
        'beamformer_weights':
            {
                'target': "{operations_dir}/beamformer_weights/applied/",
                'destination': "{candidates_dir}/{isot}/{candname}/Level2/calibration/"
            },
        'hdf5_files':
            {
                'target': "{operations_dir}/correlator/{hdf5_name}*.hdf5",
                'destination': "{candidates_dir}/{isot}/{candname}/Level2/calibration/"
            },
    })

    def __init__(
            self, candparams: dict, logger: dsl.DsaSyslogger = None) -> None:
        """Initialize info from candidate.

        Parameters
        ----------
        candparams : dict
            Dictionary of candidate parameters.
        logger : dsl.DsaSyslogger
            Logger object.
        """
        if logger is None:
            self.logger = dsl.DsaSyslogger()
        else:
            self.logger = logger

        self.candname = candparams['trigname']
        self.candparams = deepcopy(candparams)
        self.candtime = Time(
            self.candparams['mjds'], format='mjd', precision=0)
        
        self.candisot = self.candparams['isot']
    def __call__(self) -> dict:
        """Create the candidate directory structure and hardlink pre-existing files.

        Returns
        -------
        dict
            Dictionary of candidate parameters.
        """
        self.create_directory_structure()
        self.link_filterbank()
        self.link_beamformer_weights()
        self.copy_T2_csv()
        self.copy_T2_json()
        self.link_filplot_and_json()

        return self.candparams

    def create_directory_structure(self) -> None:
        """Create directory structure for candidate."""

        self.logger.info(
            f"Creating directory structure for candidate {self.candname}.")

        cand_dir = self.candidates_dir / self.candisot / self.candname
        for subdir in self.candidates_subdirs:
            newdir = cand_dir / subdir
            if not newdir.exists():
                newdir.mkdir(parents=True)

        self.logger.info(
            f"Directory structure at {cand_dir} created for {self.candname}.")

    def copy_voltages(self, timeout_s: int = 3*60*60) -> None:
        """Link voltages to candidate directory."""

        end_time = Time.now() + timeout_s * u.s
        tsleep = timeout_s / 100

        self.logger.info(
            f"Linking voltages to candidate directory for {self.candname}.")

        found = [False] * len(self.subband_corrnames)

        while not all(found):
            if Time.now() > end_time:
                raise FileNotFoundError(
                    "Timeout waiting for voltage files to be written.")

            for subband, corrname in enumerate(self.subband_corrnames):
                if found[subband]:
                    continue

                destpath = Path(
                    self.directory_structure['voltages']['destination'].format(
                        candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname,
                        subband=f"sb{subband:02d}"))
                sourcepath = Path(
                    self.directory_structure['voltages']['target'].format(
                        voltage_dir=self.voltage_dir, hostname=corrname, candname=self.candname))
                sourcepath_scp = self.directory_structure['voltages']['target_scp'].format(
                    hostname=corrname, candname=self.candname)

                if sourcepath.exists():
                    self.copy_file(sourcepath, destpath, remote=False)
                    found[subband] = True
                else:
                    try:
                        self.copy_file(sourcepath_scp, destpath, remote=True)
                    except subprocess.CalledProcessError as exc:
                        self.logger.error(
                            f"scp returned non-zero error code copying {sourcepath_scp} to "
                            f"{destpath} with output: {exc}")
                    else:
                        found[subband] = True

                if found[subband]:
                    self.candparams[f'voltage_sb{subband:02d}'] = str(destpath)

            time.sleep(tsleep)

        self.logger.info(f"Voltages linked for {self.candname}.")

    def copy_voltages_headers(self, timeout_s: int = 60*60) -> None:
        """Link voltage headers to candidate directory."""

        end_time = Time.now() + timeout_s * u.s
        tsleep = timeout_s / 100

        self.logger.info(
            f"Linking voltage headers to candidate directory for {self.candname}.")

        found = [False] * len(self.subband_corrnames)

        while not all(found):
            if Time.now() > end_time:
                raise FileNotFoundError(
                    "Timeout waiting for voltage files to be written.")

            for subband, corrname in enumerate(self.subband_corrnames):
                if found[subband]:
                    continue

                destpath = Path(
                    self.directory_structure['voltages_headers']['destination'].format(
                        candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname,
                        subband=f"sb{subband:02d}"))
                sourcepath = Path(
                    self.directory_structure['voltages_headers']['target'].format(
                        voltage_dir=self.voltage_dir, hostname=corrname, candname=self.candname))
                sourcepath_scp = self.directory_structure['voltages_headers']['target_scp'].format(
                    hostname=corrname, candname=self.candname)

                if sourcepath.exists():
                    self.copy_file(sourcepath, destpath, remote=False)
                    found[subband] = True
                else:
                    try:
                        self.copy_file(sourcepath_scp, destpath, remote=True)
                    except subprocess.CalledProcessError as exc:
                        self.logger.error(
                            f"scp returned non-zero error code copying {sourcepath_scp} to "
                            f"{destpath} with output: {exc}")
                    else:
                        found[subband] = True

            time.sleep(tsleep)

        self.logger.info(f"Voltage headers linked for {self.candname}.")

    def link_filterbank(self) -> None:
        """Link filterbank to candidate directory."""

        self.logger.info(
            f"Linking filterbank to candidate directory for {self.candname}.")

        for beamnumber in range(self.nbeams):
            sourcepath = Path(
                self.directory_structure['filterbank']['target'].format(
                    operations_dir=self.operations_dir, candname=self.candname,
                    beamnumber=f"{beamnumber:d}"))
            destpath = Path(
                self.directory_structure['filterbank']['destination'].format(
                    candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname,
                    beamnumber=f"{beamnumber:d}"))
            self.link_file(sourcepath, destpath)

        self.candparams['filterbank'] = str(destpath.parent)
        self.candparams['filfile_cand'] = (
            self.directory_structure['filterbank']['destination'].format(
                candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname,
                beamnumber=f"{self.candparams['ibeam']+1:03d}"))
        self.logger.info(f"Filterbank linked for {self.candname}.")

    def link_beamformer_weights(self) -> None:
        """Link beamformer weights to candidate directory.

        Links the weights applied in the real-time system at the candidate
        time.
        """
        self.logger.info(
            f"Linking beamformer weights to candidate directory for "
            f"{self.candname}.")

        beamformer_dir = Path(
            self.directory_structure['beamformer_weights']['target'].format(
                operations_dir=self.operations_dir))
        destdir = Path(
            self.directory_structure['beamformer_weights']['destination'].format(
                candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname))
        beamformer_name = find_beamformer_weights(
            self.candtime, beamformer_dir)

        self.logger.info(f"Found beamformerweights: {beamformer_name}")

        sourcepaths = beamformer_dir.glob(
            f"beamformer_weights_*_{beamformer_name}.dat")

        subband_pattern = re.compile(r'sb\d\d')
        for sourcepath in sourcepaths:
            subband = subband_pattern.findall(sourcepath.name)[0]
            destpath = destdir / sourcepath.name
            self.link_file(sourcepath, destpath)
            self.candparams[f'beamformer_weights_{subband}'] = str(destpath)

        sourcepath = beamformer_dir / f"beamformer_weights_{beamformer_name}.yaml"
        destpath = destdir / sourcepath.name
        self.link_file(sourcepath, destpath)
        self.candparams['beamformer_weights'] = str(destpath)

        self.logger.info(f"Beamformer weights linked for {self.candname}.")

    def link_hdf5_files(self, hours_to_save: float = 2., filelength_min: float = 5.) -> None:
        """Link hdf5 correlated data files to the candidates directory.

        Links all files within `hours_to_save`/2 hours of the candidate time.

        Parameters
        ----------
        hours_to_save : int
            Number of hours to save around the candidate.
        timeout_s : int
            Timeout for files to appear in source directory in s.

        Raises
        ------
        FileNotFoundError
            If less than 93.75% of the expected number of files are found in the source directory.
        """
        # Wait until the time when the hdf5 files for all of `hours_to_save` should be there
        target_time = self.candtime + hours_to_save / 2 * u.h + 10 * u.min
        wait_until(target_time)

        # Determine the dates and times for which to find files
        date_format = '%Y-%m-%d'
        today = self.candtime.strftime(date_format)
        yesterday = (self.candtime - 1 * u.d).strftime(date_format)
        tomorrow = (self.candtime + 1 * u.d).strftime(date_format)
        start = self.candtime - hours_to_save / 2 * u.h
        stop = self.candtime + hours_to_save / 2 * u.h

        self.logger.info(
            f"Linking HDF5 files for {hours_to_save} hours to candidate "
            f"directory for {self.candname}.")

        # Construct an iterator over existing files that match the dates
        source_dir = self.operations_dir / "correlator"
        sourcepaths = chain(
            source_dir.glob(f"{today}*hdf5"),
            source_dir.glob(f"{yesterday}*hdf5"),
            source_dir.glob(f"{tomorrow}*hdf5"))

        # Create a list that also match the times
        tokeep = []
        for sourcepath in sourcepaths:
            filetime = time_from_hdf5_filename(sourcepath)
            if within_times(start, stop, filetime):
                tokeep.append(sourcepath)

        # Hard link the files in `to_keep` to the candidate directory
        destpath = Path(
            self.directory_structure['hdf5_files']['destination'].format(
                candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname))
        for sourcepath in tokeep:
            self.link_file(sourcepath, destpath / sourcepath.name)

        self.logger.info(
            f"{len(tokeep)} hdf5 files linked for {self.candname}.")

        # Check that the correct number of files were hardlinked
        nfiles_expected = (
            (hours_to_save * u.h / (filelength_min * u.min)
             ).to_value(u.dimensionless_unscaled)
            * len(self.subband_corrnames))
        if len(tokeep) < nfiles_expected * 15 / 16:
            raise FileNotFoundError(
                f"Only {len(tokeep)} of {nfiles_expected} hdf5 files found.")

        self.candparams['hdf5_files'] = (
            self.directory_structure['hdf5_files']['destination'].format(
                candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname))

    def link_field_ms(self) -> None:
        """Link the field measurement at the time of the candidate."""
        raise NotImplementedError

    def link_caltables(self):
        """Link delay and bandpass calibration tables to the candidates directory.

        Links tables generated from the most recent calibrator observation
        prior to the candidate.
        """
        raise NotImplementedError

    def copy_T2_csv(self):
        """Copy the T2 csv file to the candidates directory."""

        self.logger.info(
            f"Copying T2 csv to candidate directory for {self.candname}.")

        sourcepath = Path(
            self.directory_structure['T2_csv']['target'].format(
                operations_dir=self.operations_dir))
        destpath = Path(
            self.directory_structure['T2_csv']['destination'].format(
                candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname))

        # TODO: create sourcepath to file of last two days of T2 csv
        self.copy_file(sourcepath, destpath, remote=False)

        self.candparams['T2_csv'] = str(destpath)

        self.logger.info(
            f"Linked T2 csv to candidate directory for {self.candname}")

    def copy_T2_json(self):
        """Copy the T2 json file to the candidates directory."""

        self.logger.info(
            f"Copying T2 json to candidate directory for {self.candname}.")

        sourcepath = Path(
            self.directory_structure['T2_json']['target'].format(
                operations_dir=self.operations_dir, candname=self.candname))
        destpath = Path(
            self.directory_structure['T2_json']['destination'].format(
                candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname))

        # TODO: create sourcepath to file of last two days of T2 json
        self.copy_file(sourcepath, destpath, remote=False)

        self.candparams['T2_json'] = str(destpath)

        self.logger.info(
            f"Linked T2 json to candidate directory for {self.candname}")


    def link_filplot_and_json(self):
        """Link the filplotter json and png files."""
        self.logger.info(
            f"Linking filplotter json and png to candidate directory for "
            f"{self.candname}.")

        for file in ['filplot_json', 'filplot_png']:
            sourcepath = Path(
                self.directory_structure[file]['target'].format(
                    operations_dir=self.operations_dir, candname=self.candname))
            destpath = Path(
                self.directory_structure[file]['destination'].format(
                    candidates_dir=self.candidates_dir, isot=self.candisot, candname=self.candname))
            self.link_file(sourcepath, destpath)

        self.candparams['filplot_cand'] = str(destpath)

        self.logger.info(
            f"Linked filplotter json and png to candidate directory for "
            f"{self.candname}")

    def link_file(self, sourcepath: Path, destpath: Path) -> None:
        """Link `destpath` to `sourcepath` if `sourcepath` does not already exist.

        Parameters
        ----------
        sourcepath : Path
            Path to link from.
        destpath : Path
            Path to link to.
        timeout_s: int
            Timeout in seconds.
        """
        if destpath.exists():
            self.logger.warning(
                f"{destpath} already exists. Skipped linking {sourcepath}.")
            return

        if sourcepath.exists():
            sourcepath.link_to(destpath)
            self.logger.info(f"Linked {sourcepath} to {destpath}.")
        else:
            self.logger.info(f"{sourcepath} not found")

    def copy_file(self, sourcepath: Union[str, Path], destpath: Path, remote: bool) -> None:
        """Copy `sourcepath` to `destpath` if `destpath` does not already exist.

        Parameters
        ----------
        sourcepath : str
            Path to copy from.
        destpath : Path
            Path to copy to.
        """
        if destpath.exists():
            self.logger.warning(
                f"{destpath} already exists. Skipped copying {sourcepath}.")
            return

        if remote:
            subprocess.check_output(
                f"scp {sourcepath} {destpath}", shell=True, stderr=subprocess.STDOUT)
        else:
            shutil.copy(str(sourcepath), str(destpath))

        self.logger.info(f"Copied {sourcepath} to {destpath}.")


def within_times(start_time: Time, end_time: Time, target_time: Time) -> bool:
    """Check if `target_time` is between `start_time` and `end_time`.

    Parameters
    ----------
    start_time : Time
        Start time of the interval.
    end_time : Time
        End time of the interval.
    time : Time
        Time to check if lies within the interval.

    Returns
    -------
    bool
        True if `time` is between `start_time` and `end_time`.
    """
    return start_time <= target_time <= end_time


def time_from_hdf5_filename(sourcepath: Path) -> Time:
    """Get time from hdf5 file name.

    Parameters
    ----------
    sourcepath : Path
        Path to hdf5 file.

    Returns
    -------
    Time
        Approximate start time of the file.
    """
    return Time(sourcepath.stem.split('_')[0])


def find_beamformer_weights(candtime: Time, bfdir: Path) -> str:
    """Find the beamformer weights that were in use at a time `candtime`.

    The times in the beamformer weight names are the times when they were
    uploaded to the correlator nodes. Therefore, we want the most recent
    calibration files that were created before `candtime`.

    Parameters
    ----------
    candtime : Time
        Time of the candidate.
    bfdir : Path
        Path to the beamformer weights directory.

    Returns
    -------
    str
        Name of the beamformer weights applied at `candtime`.
    """
    isot_string = (
        r"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]")
    isot_pattern = re.compile(isot_string)
    avail_calibs = sorted(
        [
            isot_pattern.findall(str(calib_path))[0] for calib_path
            in bfdir.glob(f"beamformer_weights_{isot_string}.yaml")],
        reverse=True)
    for avail_calib in avail_calibs:
        if avail_calib < isot_pattern.findall(candtime.isot)[0]:
            return avail_calib

    raise RuntimeError(f"No beamformer weights found for {candtime.isot}")


def wait_until(target_time: Time) -> None:
    """Sleep until `target_time`."""
    current_time = Time.now()
    seconds_to_wait = (target_time - current_time).to_value(u.s)
    if seconds_to_wait > 0.:
        time.sleep(seconds_to_wait)
