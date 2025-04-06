import dsautils.dsa_store as ds
import argparse
ETCD = ds.DsaStore()

def main(args):

    ETCD.put_dict(f"/mon/nsfrb/fastvis", {"shmid":args.shmid,
                                        "datasize":args.datasize,
                                        "mjd":args.mjd,
                                        "sb":args.sb,
                                        "dec":args.dec})
    return 0

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Update the /mon/nsfrb/fastvis key in etcd.')
    parser.add_argument('shmid',type=int)
    parser.add_argument('datasize',type=int)
    parser.add_argument('mjd',type=float)
    parser.add_argument('sb',type=int)
    parser.add_argument('dec',type=float)
    args = parser.parse_args()
    main(args)
    
