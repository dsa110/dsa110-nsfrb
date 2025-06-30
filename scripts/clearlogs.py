
import os
import time
while True:
    logdir = os.environ["NSFRBDIR"]+"-logfiles/"
    logfiles = ["error_log.txt",
            "inject_error_log.txt",
            "pipe_log.txt",
            "run_log.txt",
            "search_log.txt",
            "process_log.txt",
            "candcutter_log.txt",
            "candcuttertask_log.txt",
            "candcutter_error_log.txt",
            "inject_log.txt",
            "time_log.txt",
            "rttimes_log.txt",
            "rttx_log.txt",
            "srchtx_log.txt",
            "srchtime_log.txt",
            "candmem_log.txt",
            "candtime_log.txt",
            "journalctl.txt",
            "srchstartstoptime_log.txt"]
    for i in range(len(logfiles)):
        l = logfiles[i]
        print(l)
        os.system("> "+logdir + l)
    print("sleeping for 1 hour")
    time.sleep(3600)
