import h5py

with h5py.File("test.he5", mode="r") as f:
    print(f.keys())
    
    dset = f["HDFEOS"]
    print(dset.keys())
    
    dset = dset["SWATHS"]
    print(dset.keys())
    
    dset = dset["PRS_L1_HCO"]
    print(dset.keys())
    
    dset = dset["Geolocation Fields"]
    print(dset.keys())
    
    dset = dset["Latitude_SWIR"]
    data = dset[:]
    print(data[:,:3,:])