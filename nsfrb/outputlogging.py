#simple function to print to logging file during execution
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
import sys
import os
from astropy.io import fits
"""
cwd = os.environ['NSFRBDIR']

output_file = cwd + "-logfiles/run_log.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
"""
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file

def printlog(txt,output_file=run_file,end='\n'):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    print(txt,file=fout,end=end,flush=True)
    if output_file != "":
        fout.close()
    return

#functions to send candidates to slack; adapted from https://api.slack.com/tutorials/tracks/uploading-files-python
import logging, os
from slack_sdk import WebClient
error_file = cwd + "-logfiles/error_log.txt"
cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/"#cwd + "-candidates/"
final_cand_dir = cand_dir + "final_cands/"

#initialize Web API client
client = WebClient(os.environ["SLACK_TOKEN_DSA"])
candidates_channel_ID = "C07PVF82FQX"
def send_candidate_slack(filename,filedir=final_cand_dir,error_file=error_file):
    #upload file to bot
    try:
        client.files_upload_v2(channel=candidates_channel_ID,title=filename[:-4],file=filedir + filename,initial_comment=filename,)
        return 0
    except Exception as e:
        printlog(e,output_file=error_file)
        return 1

#slack alternative: pushover
import requests
import os
def send_candidate_pushover(filename,filedir=final_cand_dir,error_file=error_file):
    try:
        r = requests.post("https://api.pushover.net/1/messages.json", data = {
            "token": os.environ["PSHOVR_TOKEN_DSA"],
            "user": os.environ["PSHOVR_GROUP_DSA"],
            "message": filename[:-4]
            },

        files = {
            "attachment": (filename, open(filedir + filename, "rb"), "image/jpeg")
            })
        return 0
    except Exception as e:
        printlog(e,output_file=error_file)
        return 1

import websocket
import asyncio
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import sys
import json
from nsfrb.config import frame_dir

async def handler(websocket,filename):
    event = {"content":filename}
    #await websocket.send(json.dumps(event))
    await websocket.send(json.dumps(event))
    m = await websocket.recv()
    print("success:",m)
    await asyncio.sleep(0)
    loop = asyncio.get_running_loop()
    loop.stop()

async def runserver(filename,port=9087):
    async with serve(lambda websocket: handler(websocket,filename), "localhost", port) as server:
        await server.serve_forever()
        #server.close()
        #await server.wait_closed()
        print("done")
    return

def send_candidate_custom(filename,port=9087,filedir=final_cand_dir,error_file=error_file):
    try:
        asyncio.run(runserver(filename,port))
    except Exception as exc:
        print("done")
    return




#save image to fits file
def numpy_to_fits(img,fname):
    hdu = fits.PrimaryHDU(img)
    hdu.writeto(fname,overwrite=True)
    return


# GRAFANA FUNCTIONS
"""
/mon/nsfrb/outputs


tag? -- number in payload -- e.g. different key per cor node

payload: {'time' -- float
          'processed' -- bool
          'detected' -- bool
          'injected' -- bool
          'injected_params' -- list
          'recovered_params' -- list
          'ncands' -- int}


/mon/nsfrb/imagers/1-16


-- issue to request key/dashboard
"""

