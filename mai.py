import h5py
import numpy as np

with h5py.File("data/test.he5", mode="r") as f:
    print(f.keys())
    
    dset = f["HDFEOS"]
    print(dset.keys())
    
    dset = dset["SWATHS"]
    print(dset.keys())
    
    dset = dset["PRS_L1_HCO"]
    print(dset.keys())
    
    dset = dset["Geolocation Fields"]
    print(dset.keys())
    
    lat = dset["Latitude_SWIR"][:]
    print(lat.shape)  
    
    lon = dset["Longitude_SWIR"][:]
    print(lon.shape)      
 

if __name__ == "__main__":
    pass