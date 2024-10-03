import csv

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()

NSFRBVISFILE=cwd+"-fast-visibilities/vis_file.csv"



def main():

    """
    This script checks for visibilities that have been on-disk longer than a specified timeframe and
    deletes them. Backing up candidates to
    """

