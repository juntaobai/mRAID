import math
import numpy as np
import numpy.ma as ma
from numpy import linalg as la
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from read_psrfits import read_fits
from rfi_covariance import acm
from concurrent.futures import ThreadPoolExecutor
import h5py  # Import the h5py library

def process_chunk(i, step, infile, freq_start, freq_end, sigma, nsub, hdf5_file):
    sub_start = int(i * step)
    sub_end = int((i + 1) * step)
    if sub_end > nsub:
        sub_end = nsub
    print(sub_start, sub_end)

    mb_acm = acm(infile, sub_start, sub_end, freq_start, freq_end, sigma)
    mb_acm.read_data()
    mb_acm.normalise(base=[2600, 2800])

    mb_acm.cal_acm()
    mb_acm.cal_eigen()

    # Save eigval and eigvec to the HDF5 file
    eigval_dataset = f'eigval_sub{sub_start}_to_{sub_end}'
    eigvec_dataset = f'eigvec_sub{sub_start}_to_{sub_end}'

    hdf5_file.create_dataset(eigval_dataset, data=mb_acm.eigval)
    hdf5_file.create_dataset(eigvec_dataset, data=mb_acm.eigvec)

def main():
    parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
    parser.add_argument('-f',  '--input_file',  metavar='Input file name',  nargs='+', required=True, help='Input file name')
    parser.add_argument('-step',  '--subint_step', metavar='Step of subint', default=32, type=int, help='Subint step')
    parser.add_argument('-freq',  '--freq_range', metavar='Freq range (MHz)', nargs='+', default=[0, 0], type=int, help='Frequency range (MHz)')
    parser.add_argument('-sig',  '--sigma', metavar='Threshold', default=3, type=float, help='Masking threshold')
    parser.add_argument('-nsub',  '--num_subint', metavar='Total number of subint', default=512, type=int, help='Total number of subint in the search mode file')
    parser.add_argument('-cpu',  '--cpu_workers', metavar='Number of CPU cores', default=5, type=int, help='Number of CPU cores to use')
    parser.add_argument('-o', '--output_file', metavar='Output HDF5 file', required=True, help='Output HDF5 file name')

    args = parser.parse_args()
    step = int(args.subint_step)
    freq_start = int(args.freq_range[0])
    freq_end = int(args.freq_range[1])
    sigma = float(args.sigma)
    infile = args.input_file
    nsub = int(args.num_subint)
    cpu_workers = args.cpu_workers
    output_file = args.output_file

    npart = int(np.ceil(nsub / step))

    # Open HDF5 file
    with h5py.File(output_file, 'w') as hdf5_file:
        with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            futures = [executor.submit(process_chunk, i, step, infile, freq_start, freq_end, sigma, nsub, hdf5_file) for i in range(npart)]
            for future in futures:
                future.result()

if __name__ == "__main__":
    main()
