#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File      : read_psrfits.py
# Author    : Shi Dai and Juntao Bai
# Created   : 2025-11-10
# License   : GPL v3 License
# -----------------------------------------------------------------------------
# Description :
#   This program reads PSRFITS search-mode data.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import glob
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
from scipy import sparse
from scipy.sparse.linalg import spsolve
import math
import dask.array as da
from dask.distributed import client

import logging
logger = logging.getLogger(__name__)

# Assuming 2-bit sampling
# Create the Lookup Table (Pre-compute once)
# This maps every value from 0-255 to its four 2-bit components
LUT = np.array([
    [(i >> 6) & 0x03, (i >> 4) & 0x03, (i >> 2) & 0x03, i & 0x03]
    for i in range(256)
], dtype=np.uint8)

def unpack_nchan_axis(data_3d):
    """
    Unpacks (nsamp, npol, nchan) -> (nsamp, npol, nchan * 4)
    """
    nsamp, npol, nchan = data_3d.shape
    
    # 1. Map the bytes to the LUT. 
    # New shape becomes (nbeam, nsamp, npol, nchan, 4)
    unpacked = LUT[data_3d]
    
    # 2. Reshape to merge the new '4' dimension into the 'nchan' dimension.
    # We use reshape to effectively 'flatten' the last two dimensions.
    return unpacked.reshape(nsamp, npol, nchan * 4)

###############################

class read_fits ():
        def __init__ (self, filenames, sub0=0, sub1=40, freq0=0, freq1=0, downsamp=1, no_arpls=False, nchunks=64, lam=1e3, ratio=0.005, itermax=35):
                self.filenames = filenames
                self.sub0 = sub0
                self.sub1 = sub1
                self.downsamp = downsamp
                self.lam = lam
                self.ratio = ratio
                self.itermax = itermax
                self.no_arpls = no_arpls
                self.nchunks = nchunks
                logger.debug ('Are we fitting the baseline? {0}'.format(not self.no_arpls))

                self.nbeam = len(filenames)

                # here I assume that all beams use the same observational setup
                hdulist = fits.open(self.filenames[0], mode='readonly', memmap=True)
                self.obsbw = float(hdulist['PRIMARY'].header['OBSBW'])
                #self.ibeam = hdulist['PRIMARY'].header['IBEAM']
                self.nsub =  int(hdulist['SUBINT'].header['NAXIS2'])
                self.nbits = int(hdulist['SUBINT'].header['NBITS'])
                self.nchan = int(hdulist['SUBINT'].header['NCHAN'])
                self.nsblk = int(hdulist['SUBINT'].header['NSBLK'])
                self.npol =  int(hdulist['SUBINT'].header['NPOL'])
                #self.nstot = int(hdulist['SUBINT'].header['NSTOT'])  

                tbdata = hdulist['SUBINT'].data
                self.dat_freq = tbdata['DAT_FREQ'][0]
                #print("dat_freq:", self.dat_freq)
                #self.dat_scls = tbdata['DAT_SCL']
                #self.dat_offs = tbdata['DAT_OFFS']

                hdulist.close()

                #self.nbarray = np.empty((self.nbeam, self.nsblk*int(sub1-sub0), self.nchan))
                #self.freq_mask = self.dat_freq < 1520
                if freq0 == 0 and freq1 == 0:
                        #self.freq_mask = self.dat_freq > freq0
                        self.freq_mask = np.ones(self.nchan, dtype=bool)
                        self.chn_mask = np.ones(int(self.nchan/(8/self.nbits)), dtype=bool)
                else:
                        #self.freq_mask = (self.dat_freq > freq0) & (self.dat_freq < freq1)
                        #print(self.freq_mask, len(self.freq_mask), type(self.freq_mask))
                        freq_low =  math.floor(self.dat_freq[0])
                        chn0 = int(self.nchan * (freq0 - freq_low)/self.obsbw)
                        chn1 = int(self.nchan * (freq1 - freq_low)/self.obsbw)
                        self.chn_mask = np.zeros(int(self.nchan/(8/self.nbits)), dtype=bool)
                        self.chn_mask[int(chn0/(8/self.nbits)):int(chn1/(8/self.nbits))] = True

                        self.freq_mask = np.zeros(self.nchan, dtype=bool)
                        self.freq_mask[chn0:chn1] = True

                #np.save('freq_mask.npy', self.freq_mask )
                self.use_nchan = np.sum(self.freq_mask)
                self.dat_freq = self.dat_freq[self.freq_mask]
                #np.save('dat_freq.npy', self.dat_freq)
                np.save('freq_mask.npy', self.freq_mask)

                if self.sub0 == 0 and self.sub1 == 0:
                        self.nsamp = self.nsub*self.nsblk
                else:
                        self.nsamp = (self.sub1-self.sub0)*self.nsblk

                self.nbarray = np.empty((self.nbeam, int(self.nsamp/self.downsamp), self.use_nchan), dtype=np.float16)

                size_gb = self.nbarray.nbytes / (1024**3)
                logger.debug ('Size of initial nbarray: {0}GB'.format(size_gb))

        def ArPLS(self, y):
            N = len(y)
            D = sparse.eye(N, format='csc')
            D = D[1:] - D[:-1]
            D = D[1:] - D[:-1]
            D = D.T
            w = np.ones(N)
            self.lam = self.lam * np.ones(N)
            for i in range(self.itermax):
                    W = sparse.diags([w], [0], shape=(N, N))
                    LAM = sparse.diags([self.lam], [0], shape=(N, N))
                    Z = W + LAM * D.dot(D.T)
                    z = spsolve(Z, w * y)
                    d = y - z
                    dn = d[d < 0]
                    m = np.mean(dn)
                    s = np.std(dn)
                    wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
                    #self.lam = self.lam * (1-wt)
                    if np.linalg.norm(w - wt) / np.linalg.norm(w) < self.ratio:
                        break
                    w = wt
            return z

        def baseline_ArPLS(self, data):
            #self.data = data
            #baseline = self.ArPLS(self.data.mean(axis=0))

            #self.data -= baseline
            #return self.data

            baseline = self.ArPLS(data.mean(axis=0))
            return data-baseline

        def read_data (self):
                for i in range(self.nbeam):
                        hdulist = fits.open(self.filenames[i], mode='readonly', memmap=True)
                        tbdata = hdulist['SUBINT'].data
                        hdulist.close()
                        #print ('Read in data...')
                        #print ('{0} nsub:{1} nsblk:{2} nbits:{3} nchan:{4}'.format(self.filenames[i], self.nsub, self.nsblk, self.nbits, self.use_nchan))

                        if self.sub0 == 0 and self.sub1 == 0:
                                data = tbdata['DATA']
                                data = np.reshape(data, (self.nsub*self.nsblk, self.npol, int(self.nchan/(8/self.nbits))))
                        else:
                                data = tbdata['DATA'][self.sub0:self.sub1, :, :, :]
                                data = np.reshape(data, ((self.sub1-self.sub0)*self.nsblk, self.npol, int(self.nchan/(8/self.nbits))))   

                        data = data[:, :, self.chn_mask]

                        # unpack data if it's not 8-bit
                        if self.nbits != 8:
                                #temp = np.reshape(np.unpackbits(data, axis=-1), (self.nsamp, self.npol, self.use_nchan, self.nbits))
                                #unpack_data = np.squeeze(np.packbits(np.insert(temp, [0,0,0,0,0,0], 0, axis=-1), axis=-1))

                                # accelerating unpacking using a look-up table
                                #unpack_data = LUT[data]
                                #unpack_data.reshape(self.nsamp, self.npol, self.use_nchan)

				# divide the data array into 64 chunks
                                client = Client(n_workers=4, threads_per_worker=1, processes=True)
                                raw_data_dask = da.from_array(data, chunks=(int(self.nsamp/self.nchunks), self.npol, int(self.use_nchan/(8/self.nbits))))
                                new_chunks = list(raw_data_dask.chunks)
                                new_chunks[2] = tuple(c * 4 for c in raw_data_dask.chunks[2])
                                #unpack_data = raw_data_dask.map_blocks(unpack_nchan_axis, dtype=np.uint8, chunks=new_chunks)
                                unpack_data = raw_data_dask.map_blocks(unpack_nchan_axis, dtype=np.uint8, chunks=new_chunks)
                                client.close()
                        else:
                                unpack_data = data

                        size_gb = unpack_data.nbytes / (1024**3)
                        logger.debug ('Unpack array shape of beam{0}: {1}'.format(i, unpack_data.shape))
                        logger.debug ('Unpack array size of beam{0}: {1}GB'.format(i, size_gb))

                        # only use total intensity
                        if self.npol != 1:
                                output_data = unpack_data[:,0,:] + unpack_data[:,1,:]
                        else:
                                output_data = np.squeeze(unpack_data)

                        output_data = output_data.reshape(int(self.nsamp/self.downsamp), self.downsamp, self.use_nchan).mean(axis=1)
                        output_data = output_data.astype(np.float16)

                        size_gb = output_data.nbytes / (1024**3)
                        logger.debug ('Output array shape of beam{0}: {1}'.format(i, output_data.shape))
                        logger.debug ('Output array size {0}: {1}GB'.format(i, size_gb))

                        if self.no_arpls is False:
                                logger.info ('Fitting baseline with ArPLS...')
                                output_data = self.baseline_ArPLS(output_data)
                                output_data = output_data.astype(np.float16)
                        #print(data.dtype)

                        #############################################################
                        self.nbarray[i] = output_data
                        size_gb = self.nbarray.nbytes / (1024**3)
                        #logger.debug ('Size of nbarray {0}GB'.format(size_gb))

                        #print ('Shape of the unpacked data: {0}\n'.format(self.nbarray[i].shape))
                        #np.save('raw_data_ArPLS.npy', self.nbarray )

        def plot_bandpass (self, beam=np.nan):
                plt.clf()
                ax1 = plt.subplot(111)
                ax1.set_xlabel('Channel')
                ax1.set_xlim(0, self.nchan)
                ax1.set_xticks(np.arange(0,self.nchan,100))

                ax2 = ax1.twiny()
                ax2.set_xlabel('Frequency (MHz)')
                ax2.set_xlim(0, self.nchan)
                ax2.set_xticks(np.arange(0,self.nchan,100))
                ax2.set_xticklabels(np.round(self.dat_freq[0] + np.arange(0,self.nchan,100)*self.obsbw/self.nchan,1))

                if self.npol == 1:
                        if np.isnan(beam):
                                bp = np.mean(self.nbarray, axis=1)
                        else:
                                bp = np.mean(self.nbarray[beam], axis=0)
                else:
                        if np.isnan(beam):
                                bp = np.mean(self.nbarray[:,:,0,:], axis=1)
                        else:
                                bp = np.mean(self.nbarray[beam,:,0,:], axis=0)

                if np.isnan(beam):
                        bp_name = 'allbeams_bandpass.png'
                        for i in range(self.nbeam):
                                #ax1.plot(self.dat_freq, bp[i]+i, label='beam{0}'.format(i))
                                ax1.plot(np.arange(self.nchan), bp[i]+i, label='beam{0}'.format(i))
                else:
                        #ax1.plot(self.dat_freq, bp)
                        ax1.plot(np.arange(self.nchan), bp)
                        bp_name = self.filenames[beam] + '.bandpass.png'

                #ax.legend(loc='upper right')
                plt.savefig(bp_name)

######################
if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
        parser.add_argument('-f',         '--input_file',    metavar='Input file name', nargs='+',  required=True, help='Input file name')
        parser.add_argument('-sub0',      '--subband_start', metavar='Starting subband',            required=True, help='Starting subband')
        parser.add_argument('-sub1',      '--subband_end',   metavar='Ending subband',              required=True, help='Ending subband')
        parser.add_argument('-downsamp',  '--down_sample',   metavar='Down sample', default = 1,    type = int,    help='Down sample')
        parser.add_argument('-no_arpls',  '--no_arpls_par',  action='store_true',                                  help='Turn off ArPLS')
        parser.add_argument('-arpls',     '--arpls_par',     metavar='ArPLS parameters (lam ratio itermax)', nargs=3, default=[1e3, 0.005, 35], type=float, help='ArPLS parameters (lam ratio itermax)')
        parser.add_argument('-nchunk',    '--num_chunks',    metavar='N chunks for DASK', default=64,  type = int, help='Number of chunks in the data array for DASK acceleration')

        args = parser.parse_args()
        infile    = args.input_file
        sub_start = int(args.subband_start[0])
        sub_end   = int(args.subband_end[0])
        downsamp  = int(args.down_sample)
        no_arpls  = args.no_arpls_par
        lam, ratio, itermax = args.arpls_par
        nchunks    = args.num_chunks

        srch = read_fits(filenames=infile, sub0=sub_start, sub1=sub_end, downsamp=downsamp, no_arpls=no_arpls, nchunks=nchunks, lam=lam, ratio=ratio, itermax=int(itermax))
        srch.read_data()

        #srch.plot_bandpass()
        srch.plot_bandpass(0)

