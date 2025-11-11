#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File      : run_rfi.py
# Author    : Shi Dai and Juntao Bai
# Created   : 2025-11-10
# License   : MIT License
# -----------------------------------------------------------------------------
# Description :
#   This program generates RFI masks for each beam.
# =============================================================================
import math
import numpy as np
import numpy.ma as ma
from numpy import linalg as la
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from decimal import Decimal
from scipy.optimize import curve_fit
import h5py

##########################

class eig():
        def __init__(self, filenames, hdf5_filename, nsub, sub_step, nbeam, sigma_eigval=3, sigma_eigvec=1, ignorchan_start=0, ignorchan_end=0):
                self.filenames = filenames
                self.hdf5_filename = hdf5_filename
                self.sig_val = sigma_eigval
                self.sig_vec = sigma_eigvec
                self.nsub = nsub
                self.nbeam = nbeam
                self.sub_step = sub_step
                self.ignorchan_start = ignorchan_start
                self.ignorchan_end = ignorchan_end
                self.npart = int(np.ceil(self.nsub / self.sub_step))

                self.freq_mask = np.load("freq_mask.npy") 
                self.nchan = len(self.freq_mask)
                self.use_nchan = self.freq_mask.sum()

                true_indices = np.where(self.freq_mask)[0]
                if true_indices.size > 0:
                        self.chn_s = true_indices[0]
                        self.chn_e = true_indices[-1] + 1
                else:
                        self.chn_s = 0
                        self.chn_e = self.nchan


                self.eigval_array = ma.masked_array(np.empty((self.npart, self.use_nchan)), mask=np.zeros((self.npart, self.use_nchan)))
                self.chan_array = ma.masked_array(np.empty((self.npart, self.use_nchan, self.nbeam)), mask=np.zeros((self.npart, self.use_nchan, self.nbeam)))
                self.eigvec_array = np.empty((self.npart, self.use_nchan, self.nbeam))

                self.chan_array[:, :, :] = np.arange(self.use_nchan)[None, :, None]

                with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
                        self.eigval_array = ma.masked_array(hdf5_file['eigval'][:]) 
                        self.eigvec_array = ma.masked_array(hdf5_file['eigvec'][:]) 

        def gaussian(self, x, a, mean, sigma):
                return a * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
            
        def plot_eigval (self):
                print('eigval_shape:', (self.eigval_array[:,:]).shape)
                plt.clf()
                fig = plt.figure(figsize=(8,6))
                ax = plt.subplot(111)
                cax = ax.imshow(self.eigval_array[:,:], origin='lower', aspect='auto', interpolation='none')
                cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.02)
                ax.set_xlabel(f"Channel", fontsize=14)
                ax.set_ylabel(f"Subinterval", fontsize=14)
                start_x = int((self.nchan-self.use_nchan)/2)
                ax.set_xticks(np.arange(0, self.use_nchan, 500), np.arange(start_x, self.use_nchan+start_x, 500))
                plt.savefig(f'{hdf5_filename}_plot_eigval.png')
                

        def generate_filter (self):
                nchan_zap0 = np.ma.count_masked(self.eigval_array)
                mid = self.eigval_array[:,:].ravel()
                index = np.where((mid > -0.01) & (mid < 0.01))
                data = mid[index].compressed()
                counts, bin_edges = np.histogram(data, bins=200)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                mea, st = np.abs(np.mean(data)), np.abs(np.std(data))
                cou = max(counts)
                bounds_par = ([cou-0.3*cou, -1, -1], [cou+0.3*cou, 1, 1])
                
                popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=[cou, mea, st], bounds=bounds_par, maxfev = 100000)
                print('sigma:',popt[2])
                fit_sigma =  popt[2]
                self.eigval_array[:,:] = ma.masked_greater(self.eigval_array[:,:], popt[1]+self.sig_val*popt[2] )
                self.eigval_array[:,:] = ma.masked_less(self.eigval_array[:,:], popt[1]-self.sig_val*popt[2] )
                nchan_zap = np.ma.count_masked(self.eigval_array)
                print("nchan_zap:", nchan_zap)
                
                nchan_zap = np.ma.count_masked(self.eigval_array)

                count = 0
                while nchan_zap != nchan_zap0:
                    nchan_zap0 = nchan_zap
                    index = np.where((mid > -fit_sigma) & (mid < fit_sigma))
                    data = mid[index].compressed()
                    counts, bin_edges = np.histogram(data, bins=200)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    mea, st = np.abs(np.mean(data)), np.abs(np.std(data))
                    cou = max(counts)
                    bounds_par = ([cou-0.3*cou, -1, -1], [cou+0.3*cou, 1, 1])
                    
                    popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=[cou, mea, st], bounds=bounds_par, maxfev = 100000)
                    print(count, 'sigma:',popt[2])
                    fit_sigma =  popt[2]
                    self.eigval_array[:,:] = ma.masked_greater(self.eigval_array[:,:], popt[1]+self.sig_val*popt[2] )
                    self.eigval_array[:,:] = ma.masked_less(self.eigval_array[:,:], popt[1]-self.sig_val*popt[2] )
                    nchan_zap = np.ma.count_masked(self.eigval_array)
                    print("nchan_zap:", nchan_zap)
                    count += 1


        def cal_evc_Threshold(self):
                mean_list = []
                std_list = []
                for i in range(0, self.nbeam, 1): 
                        mid = self.eigvec_array[:,:, i]
                        masked = self.eigval_array[:,:].mask
                        mid = np.ma.array(mid, mask = masked)
                        
                        index = np.where((mid > -0.2) & (mid < 0.2))
                        data = mid[index].compressed() 

                        counts, bin_edges = np.histogram(data, bins=200)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        mea, st = np.abs(np.mean(data)), np.abs(np.std(data))
                        cou = max(counts)
                        bounds_par = ([cou-0.3*cou, -1, -1], [cou+0.3*cou, 1, 1])
                        popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=[cou, mea, st], bounds=bounds_par, maxfev = 100000)
                        mean_list.append(popt[1])
                        std_list.append(np.abs(popt[2]))
                evc_mean = np.array(mean_list)
                evc_sigma = np.array(std_list)
                print("Mean and Sigma per beam:", evc_mean, evc_sigma)
                return evc_mean, evc_sigma
            

        def generate_mask (self):
                evc_mean, evc_sigma = self.cal_evc_Threshold()
                self.rfi_mask = np.ones((self.npart, self.use_nchan, self.nbeam))
                eig_mask = self.eigval_array.mask[:,:]
                plt.clf()
                fig = plt.figure(figsize=(8,6))
                ax = plt.subplot(111)
                cax = plt.imshow(self.eigval_array[:,:], origin='lower', aspect='auto', rasterized=True, interpolation='none')
                cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.02)
                ax.set_xlabel(f"Channel", fontsize=14)
                ax.set_ylabel(f"Subinterval", fontsize=14)
                start_x = int((self.nchan-self.use_nchan)/2)
                ax.set_xticks(np.arange(0, self.use_nchan, 500), np.arange(start_x, self.use_nchan+start_x, 500))
                plt.savefig(f'{hdf5_filename}_plot_eigval_masked.png')
                
                init_mask = np.moveaxis(np.array(self.nbeam*[eig_mask]), 0, -1)  
                for j in range(self.npart):
                        for i in range(self.use_nchan):
                                if self.eigval_array.mask[j,i] == 1:
                                        vec = self.eigvec_array[j, i, :]
                                        beam_mask = (vec < (evc_mean - evc_sigma* self.sig_vec)) | (vec > (evc_mean + evc_sigma* self.sig_vec))
                                        self.rfi_mask[j, i, beam_mask] = 0
                #print("self.rfi_mask.dtype:", self.rfi_mask.dtype)
                self.all_rfi_mask = np.zeros((self.npart, self.nchan, self.nbeam))
                self.all_rfi_mask[:, self.chn_s:self.chn_e, :] = self.rfi_mask
                threshold_chan = 0.8
                for j in range(self.nbeam):
                        for i in range(self.nchan):  
                                rfi_fraction = np.sum(self.all_rfi_mask[:, i, j] == 0) / self.npart
                                
                                if rfi_fraction > threshold_chan:
                                        self.all_rfi_mask[:, i, j] = 0  
                if (self.ignorchan_start != 0 and self.ignorchan_end != 0 and self.ignorchan_start < self.ignorchan_end):
                        print(f"Ignoring channel range [{self.ignorchan_start}:{self.ignorchan_end}]")
                        self.all_rfi_mask[:, self.ignorchan_start:self.ignorchan_end, :] = 1
                self.all_rfi_mask = np.array(self.all_rfi_mask, dtype=bool)
                np.save('rfi_mask.npy', self.all_rfi_mask )
                print ("4:", np.array_equal(self.all_rfi_mask[:,:,0], self.all_rfi_mask[:,:,-1]))


        def plot_result (self, xtick):
                plt.clf()
                beam=np.arange(self.nbeam)
                xr=[0, self.nchan]

                plt.figure(figsize=(18,18))
                plt.subplots_adjust(hspace=0.1, wspace=0.1)
                i = 0
                for beam_id in beam:
                        plot_idx = i+1
                        ax = plt.subplot(5, 4, plot_idx)
                        ax.set_title('beam{0}'.format(beam_id+1))
                        ax.set_ylabel('Time', fontsize=14)

                        ax.set_xticks([])
                        ax.set_yticks([])
                        if plot_idx == 17 or plot_idx == 18 or plot_idx == 19:
                                ax.set_xticks(np.arange(0, self.use_nchan, xtick))
                                ax.set_xlabel('Channel number', fontsize=14)

                        ax.set_xlim(xr[0], xr[1])
                        ax.imshow(self.all_rfi_mask[:,:,beam_id], origin='lower', aspect='auto', interpolation='none')#, cmap='gray')
                        num_zeros = np.count_nonzero(self.all_rfi_mask[:,:,beam_id] == 0)
                        num_ones = np.count_nonzero(self.all_rfi_mask[:,:,beam_id] == 1)
                        countt = (self.nchan-self.use_nchan)*self.npart
                        print(beam_id, num_zeros, num_ones, num_zeros + num_ones, (num_zeros - countt)/ (num_zeros + num_ones - countt) )
                        i += 1

                plt.savefig(f'{hdf5_filename}_plot_eigvec_masked.png')

        ### generate the mask file
        def write_mask(self):
                basename = self.filenames
                self.nbeam = len(basename)
                mask = self.all_rfi_mask
                mask = np.where(mask == 0, 1, 0)

                for beam_id in range(self.nbeam):
                        print('basename:', basename[beam_id])
                        hdu = fits.open(basename[beam_id])
                        time_sig=np.float64(10.0)
                        freq_sig=np.float64(4.0)
                        tsamp = hdu[1].header['TBIN']
                        secperday = 3600 * 24
                        samppersubint = int(hdu[1].header['NSBLK'])
                        subintoffset = hdu[1].header['NSUBOFFS']
                        MJD = "%.11f" % (Decimal(hdu[0].header['STT_IMJD']) + Decimal(hdu[0].header['STT_SMJD'] + tsamp * samppersubint * subintoffset )/secperday )
                        MJD = np.float64(MJD)
                        lofreq = hdu[0].header['obsfreq'] - hdu[0].header['obsbw']/2
                        lofreq = np.float64(lofreq)
                        df = np.float64(hdu[1].header['chan_bw'])
                        nchan = np.int32(hdu[1].header['nchan'])
                        nint = np.int32(self.npart)
                        ptsperint = samppersubint * self.nsub / self.sub_step
                        ptsperint = np.int32(ptsperint)
                        dtint = np.float64(ptsperint * tsamp)
                        nzap_f=0
                        mask_zap_chans=[]
                        for i in range(nchan):
                            if mask[:,i,beam_id].sum(axis=0)==nint:
                                nzap_f = nzap_f+1
                                mask_zap_chans.append(i)
                        nzap_f=np.int32(nzap_f)
                        mask_zap_chans = np.array(mask_zap_chans).astype(np.int32)
                        nzap_t = 0
                        mask_zap_ints = []
                        for i in range(nint):
                                if mask[i,:,beam_id].sum(axis=-1)==nchan:
                                        nzap_t = nzap_t+1
                                        mask_zap_ints.append(i)
                        nzap_t=np.int32(nzap_t)
                        mask_zap_ints = np.array(mask_zap_ints).astype(np.int32)
                        nzap_per_int = []
                        tozap = []
                        for i in range(nint):
                                spec = mask[i,:,beam_id]
                                nzap = spec.sum()
                                index_rfi = np.where(spec==1)
                                nzap_per_int.append(nzap)
                                tozap.append(index_rfi)
                        nzap_per_int = np.array(nzap_per_int).astype(np.int32)
                        maskfile = basename[beam_id] +'_mRAID.mask'      ### 
                        f = open(maskfile,'wb+')
                        f.write(time_sig)
                        f.write(freq_sig)
                        f.write(MJD)
                        f.write(dtint)
                        f.write(lofreq)
                        f.write(df)
                        f.write(nchan)
                        f.write(nint)
                        f.write(ptsperint)
                        f.write(nzap_f)
                        f.write(mask_zap_chans)
                        f.write(nzap_t)
                        f.write(mask_zap_ints)
                        f.write(nzap_per_int)
                        for i in range(len(tozap)):
                                f.write(np.array(tozap[i]).astype(np.int32))
                        f.close()



        ### generate the stats file
        def write_stats(self):
                basename = self.filenames
                for beam_id in range(self.nbeam):
                        #print('basename:', basename[beam_id])
                        hdu = fits.open(basename[beam_id])
                        nchan = np.int32(hdu[1].header['nchan'])
                        samppersubint = int(hdu[1].header['NSBLK'])
                        nint = np.int32(self.npart)
                        ptsperint = samppersubint * self.nsub / self.sub_step
                        ptsperint = np.int32(ptsperint)
                        sf = self.all_rfi_mask[:,:,beam_id]
                        sf = sf.astype('float64')
                        statsfile = basename[beam_id] +'_mRAID.stats'
                        f = open(statsfile,'wb+')
                        f.write(nchan)
                        f.write(nint)
                        f.write(ptsperint)
                        f.write(sf)
                        f.close()

                #################

##########################
if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')

        parser.add_argument('-nsub', '--nsubint', metavar='Total num of subint', default=8192, type=int, help='Total number of subint')
        parser.add_argument('-nbeam', '--nbeam', metavar='Total num of beam', default=19, type=int, help='Total number of beam')
        parser.add_argument('-step', '--sub_step', metavar='Step in subintegration', default=40, type=int, help='Step in subintegration')
        parser.add_argument('-sig_val', '--sigma_eigval', metavar='Eigenvalue threshold', default=3.0, type=int, help='Masking threshold')
        parser.add_argument('-sig_vec', '--sigma_eigvec', metavar='Eigenvector threshold', default=1.0, type=int, help='Masking threshold')
        parser.add_argument('-fh5', '--input_hdf5_file', metavar='Input HDF5 file name', default='./source_10.h5', help='Input HDF5 file name')
        parser.add_argument('-f',  '--input_file',  metavar='Input file name',  nargs='+', default = [], help='Input file name')
        parser.add_argument('-ignorechan', '--ignorechan', metavar='Ignore channel range', nargs='+', default = [0, 0], type = int, help='Ignore channel range')

        args = parser.parse_args()
        sigma_eigval = int(args.sigma_eigval)
        sigma_eigvec = int(args.sigma_eigvec)
        nbeam = int(args.nbeam)
        nsub = int(args.nsubint)
        sub_step = int(args.sub_step)
        hdf5_filename = args.input_hdf5_file
        filenames = args.input_file
        ignorchan_start = int(args.ignorechan[0])
        ignorchan_end = int(args.ignorechan[1])

        eig = eig( filenames=filenames, hdf5_filename=hdf5_filename, sub_step=sub_step, nsub=nsub, nbeam=nbeam, sigma_eigval=sigma_eigval, sigma_eigvec=sigma_eigvec, ignorchan_start=ignorchan_start, ignorchan_end=ignorchan_end)
        eig.plot_eigval()
        eig.generate_filter()
        eig.generate_mask()
        eig.plot_result(xtick=1000)
        eig.write_mask()
        eig.write_stats()
