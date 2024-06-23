#simple function to print to logging file during execution
f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()

output_file = cwd + "-logfiles/run_log.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"

def printlog(txt,output_file=output_file,end='\n'):
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
cand_dir = cwd + "-candidates/"

#initialize Web API client
client = WebClient(os.environ["SLACK_TOKEN_DSA"])
candidates_channel_ID = "C01NUV2M0HM"
def send_candidate_slack(filename,filedir=cand_dir,error_file=error_file):
    #upload file to bot
    try:
        client.files_upload_v2(channel=candidates_channel_ID,title=filename[:-4],file=cand_dir + filename,initial_comment=filename,)
        return 0
    except Exception as e:
        printlog(e,output_file=error_file)
        return 1
