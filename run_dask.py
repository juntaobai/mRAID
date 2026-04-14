#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File      : run_dask.py
# Author    : Shi Dai and Juntao Bai
# Created   : 2026-04-07
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
from rfi_covariance_dask import ccm

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - [PID %(process)d] - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mRAID.log", mode='w'), # Save to file
        logging.StreamHandler()           # Also print to console
    ]
)

logging.info("A fresh start! This file was just overwritten.")


def run_mRAID (filenames, out_file, sub_start=0, sub_end=0, freq_start=0, freq_end=0, sigma_val=3, sigma_vec=1,
               nsub=256, downsamp=1, normal_base_start=1400.0, normal_base_end=1500.0, no_arpls=False,
               lam=1e3, ratio=0.005, itermax=35):
        """Run mRAID on one chunk of data according to sub_start and sub_end """

        #print(f"[Task {i}] Processing subint {sub_start} to {sub_end}")
        logger.info(f"Processing subint {sub_start} to {sub_end}")

        mb_ccm = ccm(filenames, sub_start, sub_end,
                     freq_start, freq_end,
                     sigma_val, sigma_vec,
                     downsamp, normal_base_start, normal_base_end,
                     no_arpls, lam, ratio, itermax)

        mb_ccm.read_data()
        logger.info('Finished reading data...')

        mb_ccm.normalise()
        #mb_ccm.cal_ccm_einsum()
        mb_ccm.cal_ccm()
        logger.info('Finished calculating CCM...')
        #print(np.amax(mb_ccm.ccm), np.amin(mb_ccm.ccm))
        #mb_ccm.plot_ccm(100)

        mb_ccm.cal_eigen()
        logger.info('Finished SVD ...')

        with h5py.File(out_file, 'w') as hdf5_file:
            hdf5_file.create_dataset('eigval', data=mb_ccm.eigval[:, 0], dtype='float32')
            hdf5_file.create_dataset('eigvec', data=mb_ccm.eigvec[:, :, 0], dtype='float32')
            hdf5_file.attrs['sub_start'] = sub_start
            hdf5_file.attrs['sub_end'] = sub_end

        #print(f"[Task {i}] Saved results to {out_file}")
        logger.info(f"Saved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mRAID: Multi-beam RFI detection tool')
    parser.add_argument('-f',           '--input_file',     metavar='Input file name', nargs='+', required=True)
    parser.add_argument('-step',        '--subint_step',    default=20, type=int,                          help='Subint step per task')
    parser.add_argument('-freq',        '--freq_range',     nargs=2, default=[0, 0], type=int,             help='Frequency range (MHz)')
    parser.add_argument('-sig_val',     '--sigma_val',      default=3, type=int,                           help='Masking eigenvalue threshold')
    parser.add_argument('-sig_vec',     '--sigma_vec',      default=1, type=int,                           help='Masking eigenvector threshold')
    parser.add_argument('-nsub',        '--num_subint',     default=256, type=int,                         help='Total number of subints')
    parser.add_argument('-downsamp',    '--down_sample',    default=1, type=int,                           help='Down sample factor')
    parser.add_argument('-normal_base', '--normalise_base', nargs=2, default=[1400.0, 1500.0], type=float, help='Frequency range used for normalisation')
    parser.add_argument('-arpls',       '--arpls_par',      nargs=3, default=[1e3, 0.005, 35], type=float)
    parser.add_argument('-o',           '--output_prefix',  required=True,                                 help='Output prefix for HDF5 files')
    parser.add_argument('-no_arpls',    '--no_arpls_par',   action='store_true',                           help='Turn off ArPLS')
    parser.add_argument('-ncpus',       '--num_cpus',       default=5, type=int,                           help='Number of CPUs/jobs')
    #parser.add_argument('-i',           '--task_index',     default=0, type=int,                           help='Task index (for PBS array job)')

    args = parser.parse_args()

    filenames            = args.input_file
    step                 = args.subint_step
    freq_start, freq_end = args.freq_range
    sigma_val            = args.sigma_val
    sigma_vec            = args.sigma_vec
    nsub                 = args.num_subint
    downsamp             = args.down_sample
    lam, ratio, itermax  = args.arpls_par
    output_prefix        = args.output_prefix
    no_arpls             = args.no_arpls_par
    ncpus                = args.num_cpus
    normal_base_start, normal_base_end = args.normalise_base
    #task_index           = args.task_index
    #######################################

    # 1. Initialize a Dask cluster with exactly ncpus workers
    # Using threads_per_worker=1 is usually safest for heavy numpy/HDF5 I/O tasks 
    # to avoid the Python Global Interpreter Lock (GIL) and HDF5 concurrency issues.
    cluster = LocalCluster(n_workers=ncpus, threads_per_worker=1, processes=True)
    client = Client(cluster)

    #print(f"Dask Dashboard accessible at: {client.dashboard_link}")
    logger.debug(f"Dask Dashboard accessible at: {client.dashboard_link}")

    # 2. Create the lazy tasks
    tasks = []
    for start in range(0, nsub, step):
        # Calculate the end index. Since you specified "0 to 4" for a chunk of 5, 
        # we subtract 1. (If your 'ccm' function expects exclusive Python slicing 
        # like 0:5, simply remove the '- 1').
        if step == 1:
            end = start + 1
        else:
            end = min(start + step - 1, nsub - 1)
        
        # Give each chunk a unique output file to prevent write collisions
        out_file = f"{output_prefix}_{start}_{end}.h5"

        # Wrap the function call in dask.delayed instead of running it immediately
        task = delayed(run_mRAID)(
            filenames=filenames,
            out_file=out_file,
            sub_start=start,
            sub_end=end,
            no_arpls=no_arpls,
            downsamp=downsamp
        )
        tasks.append(task)

    # 3. Execute the computation graph
    #print(f"Submitting {len(tasks)} tasks to Dask...")
    logger.debug(f"Submitting {len(tasks)} tasks to Dask...")
    results = dask.compute(*tasks)
    
    #print("All tasks completed successfully!")
    logger.debug("All tasks completed successfully!")
    
    # Cleanly shut down the cluster
    client.close()
    cluster.close()

    #for start in range(0, nsub, step):
    #    if step > 1:
    #        end = min(start + step - 1, nsub - 1)
    #    else:
    #        end = start + 1
    #    
    #    # Give each chunk a unique output file to prevent write collisions
    #    out_file = f"{output_prefix}_{start}_{end}.h5"

    #    # Wrap the function call in dask.delayed instead of running it immediately
    #    run_mRAID(
    #        filenames=filenames,
    #        out_file=out_file,
    #        sub_start=start,
    #        sub_end=end,
    #        no_arpls=no_arpls,
    #        downsamp=downsamp
    #    )
