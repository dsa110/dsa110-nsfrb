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


