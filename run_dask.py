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
from dask_jobqueue import SLURMCluster

import logging
import sys

#logger = logging.getLogger(__name__)
#logging.basicConfig(
#    level=logging.DEBUG,
#    format="%(asctime)s - [PID %(process)d] - %(name)s - %(levelname)s - %(message)s",
#    handlers=[
#        logging.FileHandler("mRAID.log", mode='w'), # Save to file
#        logging.StreamHandler()           # Also print to console
#    ]
#)
#
#logging.info("A fresh start! This file was just overwritten.")


def run_mRAID (filenames, out_file, sub_start=0, sub_end=0, freq_start=0, freq_end=0, sigma_val=3, sigma_vec=1,
               nsub=256, downsamp=1, normal_base_start=1400.0, normal_base_end=1500.0, no_arpls=False,
               nchunks=64, lam=1e3, ratio=0.005, itermax=35):
        """Run mRAID on one chunk of data according to sub_start and sub_end """

        # 1. Get the Root Logger
        root_logger = logging.getLogger()
        
        # 2. Clear any existing handlers (important for Dask worker reuse)
        # This prevents the same log lines from printing 5 times if a worker runs 5 tasks.
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # 3. Define the format
        formatter = logging.Formatter(
            f"%(asctime)s - [CHUNK {sub_start}-{sub_end}] - [PID %(process)d] - %(levelname)s - %(message)s"
        )

        # 4. Add a StreamHandler (to feed Slurm --output)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)

        # 5. Optional: Add a FileHandler for this specific chunk
        fh = logging.FileHandler(f"mRAID_{sub_start}_{sub_end}.log", mode='w')
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

        # Set the level
        root_logger.setLevel(logging.DEBUG)

        # --- Now all calls below this will log correctly ---
        logger.info("Task initialized.") # This works

        #print(f"[Task {i}] Processing subint {sub_start} to {sub_end}")
        #logger.info(f"Processing subint {sub_start} to {sub_end}")

        mb_ccm = ccm(filenames, sub_start, sub_end,
                     freq_start, freq_end,
                     sigma_val, sigma_vec,
                     downsamp, normal_base_start, normal_base_end,
                     no_arpls, nchunks, lam, ratio, itermax)

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

##########################################################

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
    parser.add_argument('-ncpus',       '--num_cpus',       default=5,     type=int,                       help='Number of CPUs/jobs')
    parser.add_argument('-nchunks',     '--num_chunks',     default=64,    type=int,                       help='Number of chunks for DASK unpacking and CCM calculation')
    parser.add_argument('-time',        '--wall_time',      default='06',  type=str,                       help='Slurm wall time for each job')

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
    nchunks              = args.num_chunks             # num_chunks is used for Dask within functions such as unpacking and cal_ccm; this is not used when calling Dask at higher level to avoid dask-within-dask
    normal_base_start, normal_base_end = args.normalise_base
    wall_time            = args.wall_time

    # 1. Setup SLURMCluster
    # This configuration asks Slurm for nodes. Adjust 'queue', 'cores', and 'memory' 
    # to match your HPC's specific partition rules.
    cluster = SLURMCluster(
        account='od-207757',
        cores=1,                        # One task per Slurm job
        memory='100GB',                 # Match your 100GB+ array needs
        walltime='{0}:00:00'.format(args.wall_time),
        # Crucial for NumPy/HDF5: ensure each worker is a separate process
        job_extra_directives=['--ntasks=1', '--cpus-per-task=1']
    )

    # 2. Scale the cluster
    # This tells Slurm to start 'ncpus' number of jobs.
    cluster.scale(jobs=args.num_cpus)
    client = Client(cluster)
    
    # 3. Create the lazy task list
    tasks = []
    for start in range(0, nsub, step):
        end = start + 1 if step == 1 else min(start + step - 1, nsub - 1)
        out_file = f"{output_prefix}_{start}_{end}.h5"

        # Wrap run_mRAID in delayed
        task = delayed(run_mRAID)(
            filenames=filenames,
            out_file=out_file,
            sub_start=start,
            sub_end=end,
            no_arpls=no_arpls,
            downsamp=downsamp,
            nchunks=nchunks
        )
        tasks.append(task)

    # 4. Execute everything across the HPC nodes
    #logger.info(f"Submitting {len(tasks)} tasks to Slurm via Dask...")
    print (f"Submitting {len(tasks)} tasks to Slurm via Dask...")
    results = dask.compute(*tasks)
    
    #logger.info("All tasks completed. Closing cluster.")
    print ("All tasks completed. Closing cluster.")
    client.close()
    cluster.close()
