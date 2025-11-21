import numpy as np
from astropy.time import Time
import json



# need to implement functions to send voltage triggers, model from dump_cluster_results_json()https://github.com/dsa110/dsa110-T2/blob/main/T2/cluster_heimdall.py#L406

import dsautils.dsa_store as ds
ETCD = ds.DsaStore()
def nsfrb_dump_cluster_results_json(
        tab,
        outputfile=None,
        output_cols=["mjds", "snr", "ibox", "dm", "ibeam", "cntb", "cntc"], 
        trigger=False,
        lastname=None,
        gulp=None,
        cat=None,
        beam_model=None,
        coords=None,
        snrs=None,
        outroot="",
        nbeams=0,
        max_nbeams=40,
        frac_wide=0.0,
        injectionfile='/home/ubuntu/data/injections/injection_list.txt',
        prev_trig_time=None,
        min_timedelt=1.
    ):
    """
    *adapted for NSFRB triggers*
    Takes tab from parse_candsfile and clsnr from get_peak,
    json file will be named with generated name, unless outputfile is set
    TODO: make cleaner, as it currently assumes NSNR at compile time.
    candidate name and specnum is calculated. name is unique.
    trigger is bool to update DsaStore to trigger data dump.
    cat is path to source catalog (default None)
    beam_model is pre-calculated beam model (default None)
    coords and snrs are parsed source file input
    injectionfile is path to info on injects and controls whether trigger is compared to that
    returns row of table that triggered, along with name generated for candidate.
    """

    print("START TRIGGER")


    itime = tab["itime"]
    maxsnr = tab["snr"]#.max()
    #imaxsnr = np.where(tab["snr"] == maxsnr)[0][0]
    #itime = str(itimes[imaxsnr])
    specnum = tab["specnum"]#(int(itime) - offset) * downsample
    mjd = tab["mjds"]#[imaxsnr]
    snr = tab["snr"]#[imaxsnr]
    maxsnr=snr
    dm = tab["dm"]#[imaxsnr]
    #ibeam = tab["ibeam"]#[imaxsnr]
    isinjection = False #-->because injections are after the split to fast visibilities, we don't need ibeam and always know its not an injection b/c of nsfrb flags
    candname = tab["trigname"] #-->already have candname


    output_dict = {candname: {}}
    if outputfile is None:
        outputfile = f"{outroot}cluster_output{candname}.json"

    #row = tab[imaxsnr]
    #red_tab = tab[imaxsnr : imaxsnr + 1]
    for col in output_cols:
        if col in tab.keys():
            if type(tab[col]) == np.int64:
                output_dict[candname][col] = int(tab[col])
            else:
                output_dict[candname][col] = tab[col]
        else:
            output_dict[candname][col] = -1

    output_dict[candname]["specnum"] = specnum
    (
        output_dict[candname]["ra"],
        output_dict[candname]["dec"],
    ) = (tab['ra'],tab['dec'])#get_radec(output_dict)

    if gulp is not None:
        output_dict[candname]["gulp"] = gulp

    output_dict[candname]['injected'] = isinjection

    #* don't need to worry about nbeams condition or len(tab)==0
    # cat and red_tab not used
    if prev_trig_time is not None:
        if Time.now()-prev_trig_time < min_timedelt*units.s:
            print(f"Not triggering because of short wait time")
            #logger.info(f"Not triggering because of short wait time")
            return None, candname, None
        else:
            trigtime = None

    with open(outputfile, "w") as f:  # encoding='utf-8'
        print(
            f"Writing trigger file with SNR={maxsnr}"
        )
        #logger.info(
        #    f"Writing trigger file with SNR={maxsnr}"
        #)
        json.dump(output_dict, f, ensure_ascii=False, indent=4)

    if trigger and Time.now().mjd - mjd < 13:
        nsfrb_send_trigger(output_dict=output_dict,outputfile=outputfile)
        trigtime = Time.now()
    else:
        trigtime = None
    print("END TRIGGER")
    return output_dict, candname, trigtime

NSFRB_TRIGGER_KEY = "/testkey1"#"/mon/corr/1/nsfrbtrigger"
NSFRB_CORR0CMD_KEY = "/testkey2"#"/cmd/corr/0"
def nsfrb_send_trigger(output_dict=None, outputfile=None):
    """Use either json file or dict to send trigger for voltage dumps via etcd."""

    if outputfile is not None:
        print("Overloading output_dict trigger info with that from outputfile")
        #logger.info(
        #    "Overloading output_dict trigger info with that from outputfile"
        #)
        with open(outputfile, "r") as f:
            output_dict = json.load(f)

    candname = list(output_dict)[0]
    if outputfile is None:
        outputfile = "/tmp/cluster_output{candname}.json"
    val = output_dict.get(candname)
    isinjection = False #output_dict[candname]['injected']

    print(
        f"Sending trigger for candidate {candname} with specnum {val['specnum']}"
    )
    #logger.info(
    #    f"Sending trigger for candidate {candname} with specnum {val['specnum']}"
    #)

    with open(outputfile, "w") as f:  # encoding='utf-8'
        print(
            f"Writing dump dict"
        )
        json.dump({"cmd": "nsfrbtrigger", "val": f'{val["specnum"]}-{candname}-'}, f, ensure_ascii=False, indent=4)

    ETCD.put_dict(
        NSFRB_CORR0CMD_KEY,
        {"cmd": "nsfrbtrigger", "val": f'{val["specnum"]}-{candname}-'},
    )  # triggers voltage dump in corr.py
    ETCD.put_dict(
        NSFRB_TRIGGER_KEY, output_dict
    )  # tells look_after_dumps.py to manage data
    #else:
    #    print(f"Candidate {candname} was detected as an injection. Not triggering voltage recording.")
    #    slack_client.chat_postMessage(channel='candidates', text=f'Injection detected as {candname} with DM={val["dm"]} and SNR={val["snr"]}.')

    return



