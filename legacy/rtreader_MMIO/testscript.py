import rtreader
import sys

#print(rtreader.read(int(sys.argv[1]),11).hex())
import dsautils.dsa_store as ds
ETCD = ds.DsaStore()
d = ETCD.get_dict("/mon/nsfrb/fastvis")
print(rtreader.read(d['shmid'],d['datasize']).hex())
