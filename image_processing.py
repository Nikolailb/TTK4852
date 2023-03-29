import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

def get_frequency_cubes(filename):
    with h5py.File(f"data/{filename}.he5", mode="r") as file:
        dset = file
        data = dset["HDFEOS"]["SWATHS"]["PRS_L1_HCO"]["Data Fields"]
        return data["VNIR_Cube"][:], data["SWIR_Cube"][:]

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
    return vnir[:, NIR_INDEX, :].mean(axis=1) - (vnir[:, RE2_INDEX, :].mean(axis=1) + (swir[:, SWIR_INDEX, :].mean(axis=1) - vnir[:, RE2_INDEX, :].mean(axis=1)) * ((NIR_LAMBDA - RED_LAMBDA) / (SWIR_LAMBDA - RED_LAMBDA)) * 10)

def get_ndvi(vnir):
    return (vnir[:, NIR_INDEX, :].mean(axis=1) - vnir[:, RED_INDEX, :].mean(axis=1)) / (vnir[:, NIR_INDEX, :].mean(axis=1) + vnir[:, RED_INDEX, :].mean(axis=1))

def get_plastic_index(vnir):
    return vnir[:, NIR_INDEX, :].mean(axis=1) / (vnir[:, NIR_INDEX, :].mean(axis=1) + vnir[:, RED_INDEX, :].mean(axis=1))

def show_image(arr, title, show=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr)
    ax.set_title(title)
    if show:
        plt.show()

NIR_INDEX = slice(15, 17)
RED_INDEX = slice(33, 35)
RE2_INDEX = slice(24, 27)
SWIR_INDEX = slice(106, 108)

NIR_LAMBDA = np.mean([849.21, 838.5272])
RED_LAMBDA = np.mean([664.8941, 655.41876])
SWIR_LAMBDA = np.mean([1616.8336, 1606.4913])

def main():
    vnir, swir = get_frequency_cubes("PRS_L1_STD_OFFL_20220312105923_20220312105927_0001")
    _, _, n_vnir = normalize(vnir)
    _, _, n_swir = normalize(swir)
    fdi = get_fdi(n_vnir, n_swir)
    ndvi = get_ndvi(n_vnir)
    show_image(ndvi, "NDVI")
    show_image(get_plastic_index(vnir), "Plastic index")

if __name__ == "__main__":
    main()