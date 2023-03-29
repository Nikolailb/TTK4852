import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.font_manager
from geopy.geocoders import Nominatim
matplotlib.rcParams['font.family'] = ['sans-serif', 'BIZ UDGothic']

def get_frequency_cubes(filename):
    with h5py.File(f"data/{filename}.he5", mode="r") as file:
        data = file["HDFEOS"]["SWATHS"]["PRS_L1_HCO"]["Data Fields"]
        loc = file["HDFEOS"]["SWATHS"]["PRS_L1_HCO"]["Geolocation Fields"]
        return data["VNIR_Cube"][:], data["SWIR_Cube"][:], loc["Latitude_SWIR"][:], loc["Longitude_SWIR"][:]

def normalize(cube):
    """Normalizes the provided cube. Expected shape is (n, features, n)

    Args:
        cube (np.ndarray): Cube of image bands

    Returns:
        np.ndarray, np.ndarray, np.ndarray: Minimum cube, maximum cube, normalized cube
    """
    max_pp = np.max(cube, axis=1, keepdims=True)
    min_pp = np.min(cube, axis=1, keepdims=True)
    return min_pp, max_pp, (cube - min_pp) / (max_pp - min_pp)

def get_fdi(vnir, swir):
    """Floating Debris Index (FDI)"""
    return vnir[:, NIR_INDEX, :].mean(axis=1) - (vnir[:, RE2_INDEX, :].mean(axis=1) + (swir[:, SWIR_INDEX, :].mean(axis=1) - vnir[:, RE2_INDEX, :].mean(axis=1)) * ((NIR_LAMBDA - RED_LAMBDA) / (SWIR_LAMBDA - RED_LAMBDA)) * 10)

def get_ndvi(vnir):
    """Normalised Vegetation Difference Index (NDVI)"""
    return (vnir[:, NIR_INDEX, :].mean(axis=1) - vnir[:, RED_INDEX, :].mean(axis=1)) / (vnir[:, NIR_INDEX, :].mean(axis=1) + vnir[:, RED_INDEX, :].mean(axis=1))

def get_plastic_index(vnir):
    return vnir[:, NIR_INDEX, :].mean(axis=1) / (vnir[:, NIR_INDEX, :].mean(axis=1) + vnir[:, RED_INDEX, :].mean(axis=1))

def get_kndvi(vnir):
    return np.tanh(np.square(get_ndvi(vnir)))

def show_image(arr, title, coords=None, show=False):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(arr)
    if type(coords) is tuple:
        geolocator = Nominatim(user_agent="geoapiExercises")
        address = geolocator.reverse(f"{coords[0][0, 0]},{coords[1][0, 0]}", language="en").raw["address"]
        title += f" - Country: {address.get('country', 'Unknwn')} - Municipality: {address.get('municipality', 'Unknwn')}"
    ax.set_title(title)
    if show:
        plt.show()

def get_plastic(swir):
    slices = np.array([[j for j in range(i - PLASTIC_STEP, i + PLASTIC_STEP)] for i in PLASTIC_DIP_INDEX])
    slices = slices.flatten()
    return np.sum(swir[:, slices, :], axis=1)


NIR_INDEX = slice(15, 17)
RED_INDEX = slice(33, 35)
RE2_INDEX = slice(24, 27)
SWIR_INDEX = slice(106, 108)

NIR_LAMBDA = np.mean([849.21, 838.5272])
RED_LAMBDA = np.mean([664.8941, 655.41876])
SWIR_LAMBDA = np.mean([1616.8336, 1606.4913])

PLASTIC_DIP_LAMBDA = np.array([1250, 1375, 1740, 2250])
PLASTIC_DIP_INDEX = np.array([140, 129, 93, 34])
PLASTIC_STEP = 1


swir_f = np.array([
 2497.1155  ,2490.2192  ,2483.793   ,2477.055   ,2469.6272  ,2463.0303,
 2456.5857  ,2449.1423  ,2442.403   ,2435.5442  ,2428.6677  ,2421.2373,
 2414.3567  ,2407.6045  ,2400.036   ,2393.0388  ,2386.0618  ,2378.771,
 2371.5522  ,2364.5945  ,2357.2937  ,2349.7915  ,2342.8228  ,2335.5264,
 2327.8242  ,2320.8955  ,2313.2007  ,2305.7227  ,2298.6094  ,2290.8267,
 2283.4934  ,2276.0537  ,2268.2883  ,2260.8665  ,2253.1104  ,2245.4485,
 2237.904   ,2230.0076  ,2222.4263  ,2214.625   ,2206.843   ,2199.1353,
 2191.1003  ,2183.4202  ,2175.3442  ,2167.4849  ,2159.564   ,2151.3862,
 2143.4656  ,2135.5103  ,2127.3372  ,2119.2314  ,2111.039   ,2102.8213,
 2094.6252  ,2086.3823  ,2077.9915  ,2069.7957  ,2061.3787  ,2053.0078,
 2044.6809  ,2036.2607  ,2027.7267  ,2019.3214  ,2010.6614  ,2002.1106,
 1993.5482  ,1984.853   ,1976.013   ,1967.3418  ,1958.6244  ,1949.9008,
 1941.1107  ,1932.26    ,1923.3857  ,1914.3014  ,1904.9347  ,1896.0913,
 1887.0809  ,1878.7426  ,1868.1733  ,1859.5587  ,1850.5543  ,1841.3256,
 1832.0272  ,1822.4413  ,1813.0513  ,1803.5902  ,1793.9531  ,1784.7173,
 1775.1178  ,1765.5126  ,1755.833   ,1746.2192  ,1736.4884  ,1726.6516,
 1716.8589  ,1707.0945  ,1697.2943  ,1687.4269  ,1677.3193  ,1667.1852,
 1656.933   ,1647.2316  ,1637.0919  ,1627.021   ,1616.8336  ,1606.4913,
 1596.2454  ,1585.8599  ,1575.6273  ,1565.3688  ,1554.8168  ,1544.2262,
 1533.7764  ,1523.2223  ,1512.6333  ,1502.0234  ,1491.4292  ,1480.8422,
 1469.9308  ,1459.3157  ,1449.1888  ,1438.466   ,1427.3746  ,1416.5374,
 1405.627   ,1394.754   ,1383.2798  ,1372.9117  ,1361.0531  ,1349.7877,
 1339.1294  ,1328.2993  ,1317.2566  ,1306.218   ,1295.4219  ,1284.4878,
 1273.4963  ,1262.5322  ,1250.9799  ,1240.2145  ,1229.1852  ,1217.8635,
 1207.2737  ,1196.3394  ,1185.5884  ,1174.7142  ,1163.6761  ,1152.6501,
 1142.0703  ,1131.3048  ,1120.6759  ,1109.8894  ,1099.2776  ,1088.761,
 1078.2161  ,1067.7948  ,1057.5737  ,1047.675   ,1037.9878  ,1029.344,
 1018.5357  ,1008.6443   ,998.9082   ,988.9179   ,979.224    ,969.8449,
  959.97394,  951.4014,   943.3579    , 0.  ,       0.     ])

def main():
    vnir, swir, lat, lon = get_frequency_cubes("PRS_L1_STD_OFFL_20220902105906_20220902105910_0001")
    _, _, n_vnir = normalize(vnir)
    _, _, n_swir = normalize(swir)
    fdi = get_fdi(n_vnir, n_swir)
    ndvi = get_ndvi(n_vnir)
    kndvi = get_kndvi(n_vnir)
    pi = get_plastic_index(n_vnir)
    test = get_plastic(n_swir) # 772, 278
    
    # show_image(test, "PLASTIC", coords=(lat, lon))
    # show_image(ndvi, "NDVI", coords=(lat, lon))
    # show_image(fdi, "FDI", coords=(lat, lon))
    # show_image(kndvi, "kNDVI", coords=(lat, lon))
    # show_image(pi, "PI", coords=(lat, lon))
    step = 5
    xs = [i for i in range(772 - step, 772 + step + 1)]
    ys = [i for i in range(278 - step, 278 + step + 1)]
    for x in xs:
        for y in ys:
            plt.plot(swir_f, n_swir[x, :, y], alpha=.5)
        
    plt.show()

if __name__ == "__main__":
    main()