#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File      : run_rfi.py
# Author    : Shi Dai and Juntao Bai
# Created   : 2025-11-10
# License   : GPL v3 License
# -----------------------------------------------------------------------------
# Description :
#   Main script for mRAID (Multi-beam RFI Identification) data processing.
#   This program reads PSRFITS search-mode data, constructs covariance matrices,
#   performs eigen-decomposition.
# =============================================================================
import math
import numpy as np
import h5py
import argparse
from astropy.io import fits
from read_psrfits import read_fits
from rfi_covariance import ccm
from multiprocessing import Pool


class mRAID (ccm):
    def __init__(self, filenames, step=20, freq_start=0, freq_end=0, sigma_val=3, sigma_vec=1, nsub=256, downsamp=1, normal_base_start=2600, normal_base_end=2800, lam=1e3, ratio=0.005, itermax=35, cpu_workers=5, output_file="test.h5"):
        self.infile = filenames
        self.step = step
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.sig_val = sigma_val
        self.sig_vec = sigma_vec
        self.nsub = nsub
        self.downsamp = downsamp
        self.sig_val = sigma_val
        self.sig_vec = sigma_vec
        self.normal_base_start = normal_base_start
        self.normal_base_end =normal_base_end
        self.lam = lam
        self.ratio = ratio
        self.itermax = itermax
        #super().__init__(filenames, sub_start, sub_end, freq_start, freq_end, sigma_val, sigma_vec, downsamp, normal_base_start, normal_base_end)
        self.cpu_workers = cpu_workers
        self.output_file = output_file

        self.npart = int(np.ceil(self.nsub / self.step))

    def process_chunk(self, i):
        sub_start = int(i * self.step)
        sub_end = min(int((i + 1) * self.step), self.nsub)
        print(f"Processing subint {sub_start} to {sub_end}")

        mb_ccm = ccm(self.infile, sub_start, sub_end, self.freq_start, self.freq_end, self.sig_val, self.sig_vec, self.downsamp, self.normal_base_start, self.normal_base_end, self.lam, self.ratio, self.itermax)
        mb_ccm.read_data()
        mb_ccm.normalise()    
        mb_ccm.cal_ccm()
        mb_ccm.cal_eigen()

        return (i, sub_start, sub_end, mb_ccm.eigval[:, 0], mb_ccm.eigvec[:, :, 0])

    def run(self):
        with Pool(processes=self.cpu_workers) as pool:
            results = pool.map(self.process_chunk, range(self.npart))

        with h5py.File(self.output_file, 'w') as hdf5_file:
            eigval_ds = hdf5_file.create_dataset('eigval', shape=(self.npart, self.use_nchan), dtype='float32')
            eigvec_ds = hdf5_file.create_dataset('eigvec', shape=(self.npart, self.use_nchan, self.nbeam), dtype='float32')

            for i, sub_start, sub_end, eigval, eigvec in results:
                eigval_ds[i, :] = eigval
                eigvec_ds[i, :, :] = eigvec

        print(f"Results saved to {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
    parser.add_argument('-f', '--input_file', metavar='Input file name', nargs='+', required=True, help='Input file name')
    parser.add_argument('-step', '--subint_step', metavar='Step of subint', default=20, type=int, help='Subint step')
    parser.add_argument('-freq', '--freq_range', metavar='Freq range (MHz)', nargs='+', default=[0, 0], type=int, help='Frequency range (MHz)')
    parser.add_argument('-sig_val',  '--sigma_val', metavar='Eigenvalue threshold', default = 3, type = int, help='Masking eigenvalue threshold')
    parser.add_argument('-sig_vec',  '--sigma_vec', metavar='Eigenvector threshold', default = 1, type = int, help='Masking eigenvector threshold')
    parser.add_argument('-nsub', '--num_subint', metavar='Total number of subint', default=256, type=int, help='Total number of subint in the search mode file')
    parser.add_argument('-cpu', '--cpu_workers', metavar='Number of CPU cores', default=5, type=int, help='Number of CPU cores to use')
    parser.add_argument('-o', '--output_file', metavar='Output HDF5 file', required=True, help='Output HDF5 file name')
    parser.add_argument('-downsamp', '--down_sample', metavar='Down sample', default=1, type=int, help='Down sample')
    parser.add_argument('-normal_base',  '--normalise_base', metavar='Normalise base', nargs=2, default = [2600, 2800], type = int, help='Normalise base')
    parser.add_argument('-arpls', '--arpls_par', metavar='ArPLS parameters (lam ratio itermax)', nargs=3, default=[1e3, 0.005, 35], type=float, help='ArPLS parameters (lam ratio itermax)')

    args = parser.parse_args()
    infile = args.input_file
    step = int(args.subint_step)
    freq_start = int(args.freq_range[0])
    freq_end = int(args.freq_range[1])
    sigma_val = int(args.sigma_val)
    sigma_vec = int(args.sigma_vec)
    nsub = int(args.num_subint)
    downsamp = int(args.down_sample)
    cpu_workers = int(args.cpu_workers)
    output_file = args.output_file
    normal_base_start = int(args.normalise_base[0])
    normal_base_end = int(args.normalise_base[1])
    lam, ratio, itermax = args.arpls_par

    runner = mRAID(filenames=infile, step=step, freq_start=freq_start, freq_end=freq_end, sigma_val=sigma_val, sigma_vec=sigma_vec, downsamp=downsamp, normal_base_start=normal_base_start, normal_base_end=normal_base_end, lam=lam, ratio=ratio, itermax=int(itermax), cpu_workers=cpu_workers, output_file=output_file )
    runner.run()

