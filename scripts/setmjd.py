from astropy.time import Time


f = open("/home/ubuntu/tmp/mjd.dat","w")
f.write(str(Time.now().mjd))
f.close()
