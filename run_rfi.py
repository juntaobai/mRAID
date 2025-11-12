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

class mRAID(ccm):
    def __init__(self, filenames, step=20, freq_start=0, freq_end=0, sigma_val=3, sigma_vec=1,
                 nsub=256, downsamp=1, normal_base_start=2600, normal_base_end=2800,
                 lam=1e3, ratio=0.005, itermax=35, task_index=0, output_prefix="mRAID_test"):
        self.infile = filenames
        self.step = step
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.sig_val = sigma_val
        self.sig_vec = sigma_vec
        self.nsub = nsub
        self.downsamp = downsamp
        self.normal_base_start = normal_base_start
        self.normal_base_end = normal_base_end
        self.lam = lam
        self.ratio = ratio
        self.itermax = itermax
        self.task_index = task_index
        self.output_prefix = output_prefix

        self.npart = int(np.ceil(self.nsub / self.step))

    def run_single_chunk(self):
        """Run one chunk corresponding to task_index."""
        i = self.task_index
        if i >= self.npart:
            print(f"Task index {i} exceeds number of parts ({self.npart}). Exiting.")
            return

        sub_start = int(i * self.step)
        sub_end = min(int((i + 1) * self.step), self.nsub)
        print(f"[Task {i}] Processing subint {sub_start} to {sub_end}")

        mb_ccm = ccm(self.infile, sub_start, sub_end,
                     self.freq_start, self.freq_end,
                     self.sig_val, self.sig_vec,
                     self.downsamp, self.normal_base_start, self.normal_base_end,
                     self.lam, self.ratio, self.itermax)

        mb_ccm.read_data()
        mb_ccm.normalise()
        mb_ccm.cal_ccm()
        mb_ccm.cal_eigen()

        out_file = f"{self.output_prefix}_part{i:03d}.h5"
        with h5py.File(out_file, 'w') as hdf5_file:
            hdf5_file.create_dataset('eigval', data=mb_ccm.eigval[:, 0], dtype='float32')
            hdf5_file.create_dataset('eigvec', data=mb_ccm.eigvec[:, :, 0], dtype='float32')
            hdf5_file.attrs['sub_start'] = sub_start
            hdf5_file.attrs['sub_end'] = sub_end

        print(f"[Task {i}] Saved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mRAID: Multi-beam RFI detection tool')
    parser.add_argument('-f', '--input_file', metavar='Input file name', nargs='+', required=True)
    parser.add_argument('-step', '--subint_step', default=20, type=int, help='Subint step per task')
    parser.add_argument('-freq', '--freq_range', nargs=2, default=[0, 0], type=int, help='Frequency range (MHz)')
    parser.add_argument('-sig_val', '--sigma_val', default=3, type=int, help='Masking eigenvalue threshold')
    parser.add_argument('-sig_vec', '--sigma_vec', default=1, type=int, help='Masking eigenvector threshold')
    parser.add_argument('-nsub', '--num_subint', default=256, type=int, help='Total number of subints')
    parser.add_argument('-downsamp', '--down_sample', default=1, type=int, help='Down sample factor')
    parser.add_argument('-normal_base', '--normalise_base', nargs=2, default=[2600, 2800], type=int)
    parser.add_argument('-arpls', '--arpls_par', nargs=3, default=[1e3, 0.005, 35], type=float)
    parser.add_argument('-i', '--task_index', default=0, type=int, help='Task index (for PBS array job)')
    parser.add_argument('-o', '--output_prefix', required=True, help='Output prefix for HDF5 files')

    args = parser.parse_args()

    infile = args.input_file
    step = args.subint_step
    freq_start, freq_end = args.freq_range
    sigma_val = args.sigma_val
    sigma_vec = args.sigma_vec
    nsub = args.num_subint
    downsamp = args.down_sample
    normal_base_start, normal_base_end = args.normalise_base
    lam, ratio, itermax = args.arpls_par
    task_index = args.task_index
    output_prefix = args.output_prefix

    runner = mRAID(
        filenames=infile,
        step=step,
        freq_start=freq_start,
        freq_end=freq_end,
        sigma_val=sigma_val,
        sigma_vec=sigma_vec,
        nsub=nsub,
        downsamp=downsamp,
        normal_base_start=normal_base_start,
        normal_base_end=normal_base_end,
        lam=lam, ratio=ratio, itermax=int(itermax),
        task_index=task_index, output_prefix=output_prefix
    )

    runner.run_single_chunk()
