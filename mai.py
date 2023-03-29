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
    
    dset = dset["Data Fields"]
    print(dset.keys())
    
    
    data = dset["SWIR_Cube"]
    swir = data[:]
    data = dset["VNIR_Cube"]
    vnir = data[:]
    # mami_swir = []
    # for i in range(swir.shape[1]):
    #     maximum, minimum = np.max(swir[:, i, :]), np.min(swir[:, i, :])
    #     if maximum == 0: continue
    #     mami_swir.append((maximum, minimum))
    #     swir[:, i, :] = (swir[:, i, :] - minimum) / (maximum - minimum)
        
    # mami_vnir = []
    # for i in range(vnir.shape[1]):
    #     maximum, minimum = np.max(vnir[:, i, :]), np.min(vnir[:, i, :])
    #     if maximum == 0: continue
    #     mami_vnir.append((maximum, minimum))
    #     vnir[:, i, :] = (vnir[:, i, :] - minimum) / (maximum - minimum)
        
    # FDI = vnir[:, 15:17,:].mean(axis=1) - (vnir[:, 24:27, :].mean(axis=1) + (swir[:, 106:108, :].mean(axis=1) - vnir[:, 24:27, :].mean(axis=1)) * 10 * ((np.mean([849.21, 838.5272]) - 664.8941) / (np.mean([1616.8336, 1606.4913]) - 644.8941)))
    # print(FDI)
    # print(FDI.max())
 

red_index = slice(33, 35)
red_lambda = np.mean([664.8941, 655.41876])
re2_index = slice(24,27)
re2_lambda = np.mean([754.4696, 744.14954, 733.9552])
NIR_index = slice(15, 17)
NIR_lambda = np.mean([849.21, 838.5272])
SWIR_index = slice(106, 108)
SWIR_lambda = np.mean([1616.8336, 1606.4913])
def getFDI():
    return vnir[:, NIR_index, :].mean(axis=1) - (vnir[:, re2_index,:].mean(axis=1) + (swir[:, SWIR_index, :].mean(axis=1) - vnir[:, re2_index, :].mean(axis=1)) * 10 * ((NIR_lambda - red_lambda) / (SWIR_lambda - red_lambda)))

def getNDVI():
    return (vnir[:, NIR_index, :].mean(axis=1) - vnir[:, red_index, :].mean(axis=1)) / (vnir[:, NIR_index, :].mean(axis=1) + vnir[:, red_index,:].mean(axis=1))
    
def normalize(a):
    ma, mi = np.max(a), np.min(a)
    t = []
    b = a.copy()

if __name__ == "__main__":
    fdi = getFDI()
    nonzero_0, nonzero_1 = fdi.nonzero()
    print(fdi[nonzero_0, nonzero_1])
    print(fdi[nonzero_0, nonzero_1].shape)
    ndvi = getNDVI()
    nonzero_0, nonzero_1 = ndvi.nonzero()
    print((ndvi[nonzero_0, nonzero_1])[~np.isnan(ndvi[nonzero_0, nonzero_1])])
    print((ndvi[nonzero_0, nonzero_1])[~np.isnan(ndvi[nonzero_0, nonzero_1])].shape)