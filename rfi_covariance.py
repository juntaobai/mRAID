#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File      : rfi_covariance.py
# Author    : Shi Dai and Juntao Bai
# Created   : 2025-11-10
# License   : GPL v3 License
# -----------------------------------------------------------------------------
# Description :
#   This program constructs covariance matrices, performs eigen-decomposition.
# =============================================================================
import math
import numpy as np
import numpy.ma as ma
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
import argparse
from astropy.io import fits
from matplotlib.gridspec import GridSpec
import time
from scipy.signal import correlate

from read_psrfits import read_fits

##########################


class ccm (read_fits):
        def __init__(self, filenames, sub0=0, sub1=0, freq0=0, freq1=0, sigma_val=3, sigma_vec=1, 
            downsamp=1, normal_base_start=2600, normal_base_end=2800, lam=1e3, ratio=0.005, itermax=35):
                print ('Initialising a multi-beam object\n')
                super().__init__(filenames=filenames, sub0=sub0, sub1=sub1, freq0=freq0, freq1=freq1, downsamp=downsamp, lam=lam, ratio=ratio, itermax=itermax)
                self.sig_val = sigma_val
                self.sig_vec = sigma_vec
                self.normal_base_start = normal_base_start
                self.normal_base_end =normal_base_end

                #self.read_data()
                for i in range(self.nbeam):
                        print ('{0} nsub:{1} nsblk:{2} nbits:{3} nchan:{4}'.format(self.filenames[i], self.nsub, self.nsblk, self.nbits, self.nchan, self.downsamp))

        def normalise (self):
                for i in range(self.nbeam):
                        std = np.std(self.nbarray[i,:,self.normal_base_start:self.normal_base_end])
                        mean = np.mean(self.nbarray[i,:,self.normal_base_start:self.normal_base_end])
                        self.nbarray[i,:,:] = (self.nbarray[i,:,:] - mean)/std

        def cal_ccm (self):
                self.ccm = np.empty((self.use_nchan, self.nbeam, self.nbeam))
                for i in range(self.use_nchan):
                        data_slice = self.nbarray[:, :, i]  
                        self.ccm[i] = np.dot(data_slice, data_slice.T) 

        def sim_ccm (self):
                print ("Simulating....\n")
                x1, x2, x3 = self.nbarray.shape
                for i in range(self.nbeam):
                        A = 3*np.random.randn(1)
                        B = 0.1*np.random.randn(1)
                        self.nbarray[i, :, :] = np.random.randn(x2, x3)*A + B  # assuming the same noise level in each beam


                # adding some RFI
                rfi = np.random.randn(x2,6)*10
                #for i in range(self.nbeam):
                for i in [0,4,12]:
                        self.nbarray[i, :, 904:910] += rfi
                        #self.nbarray[i, :, 100:105] += i*0.1

        def plot_ccm (self, chan):
                plt.clf()
                tmp = np.array(self.ccm[chan], copy=True)
                for i in range(self.nbeam):
                        tmp[i,i] = np.nan

                ax = plt.subplot(111)
                ax.imshow(tmp, origin='lower', aspect='auto')
                plt.savefig('ccm.png')
                #plt.show()

        def plot_data (self, xr, xtick):
                plt.clf()
                plt.figure(figsize=(12,12))
                plt.subplots_adjust(hspace=0.05, wspace=0.05)
                i = 0
                for i in np.arange(self.nbeam):
                        plot_idx = i+1
                        ax = plt.subplot(4, 5, plot_idx)  
                        ax.set_title('beam{0}'.format(plot_idx))

                        if plot_idx == 19:   #jt
                                ax.set_xticks(np.arange(0, self.use_nchan, xtick))
                        else:
                                ax.set_xticks([])
                                ax.set_yticks([])

                        ax.set_xlim(xr[0], xr[1])
                        lc = np.sum(self.nbarray[i], axis=1)
                        ax.imshow(self.nbarray[i], origin='lower', aspect='auto', interpolation='none')
                        i += 1
                plt.savefig('rfi_data.png')


        def cal_eigen (self):
                self.eigval = ma.masked_array(np.empty((self.use_nchan, self.nbeam)), mask=np.zeros((self.use_nchan, self.nbeam)))
                self.freq_array = ma.masked_array(np.empty((self.use_nchan, self.nbeam)), mask=np.zeros((self.use_nchan, self.nbeam)))
                self.chan_array = ma.masked_array(np.empty((self.use_nchan, self.nbeam)), mask=np.zeros((self.use_nchan, self.nbeam)))
                self.eigvec = np.empty((self.use_nchan, self.nbeam, self.nbeam))
                #print("dat_freq:", self.dat_freq)

                for i in range(self.use_nchan):
                        u, s, vh = la.svd(self.ccm[i])
                        self.eigval[i] = np.power(s, 2)/np.sum(np.power(s, 2))    # normalise eig vec
                        self.eigvec[i] = u
                        self.freq_array[i,:] = self.dat_freq[i]
                        self.chan_array[i,:] = i

                for i in range(self.nbeam):
                        self.eigval[:,i] = self.eigval[:,i] - self.ArPLS(self.eigval[:,i])
                #np.save('eigval_sub%d'%self.sub0, ma.getdata(self.eigval))
                #np.save('eigvec_sub%d'%self.sub0, ma.getdata(self.eigvec))
                #np.save('CCM_sub%d'%self.sub0, ma.getdata(self.ccm))

        def gaussian(self, x, a, mean, sigma):
                return a * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

        def generate_filter (self, xlim, xtick):
                nchan_zap0 = np.ma.count_masked(self.eigval)
                mid = self.eigval[:,0].ravel()
                index = np.where((mid > -0.01) & (mid < 0.01))
                data = mid[index].compressed()
                counts, bin_edges = np.histogram(data, bins=50)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                mea, st = np.abs(np.mean(data)), np.abs(np.std(data))
                cou = max(counts)
                bounds_par = ([cou-0.3*cou, -1, -1], [cou+0.3*cou, 1, 1])
                
                popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=[cou, mea, st], bounds=bounds_par, maxfev = 100000)
                print('sigma:',popt[2])
                fit_sigma =  popt[2]
                self.eigval[:,0] = ma.masked_greater(self.eigval[:,0], popt[1]+self.sig_val*popt[2] )
                self.eigval[:,0] = ma.masked_less(self.eigval[:,0], popt[1]-self.sig_val*popt[2] )
                nchan_zap = np.ma.count_masked(self.eigval)
                print("nchan_zap:", nchan_zap)
                
                nchan_zap = np.ma.count_masked(self.eigval)

                count = 0
                while nchan_zap != nchan_zap0:
                    nchan_zap0 = nchan_zap
                    index = np.where((mid > -fit_sigma) & (mid < fit_sigma))
                    data = mid[index].compressed()
                    counts, bin_edges = np.histogram(data, bins=50)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    mea, st = np.abs(np.mean(data)), np.abs(np.std(data))
                    cou = max(counts)
                    bounds_par = ([cou-0.3*cou, -1, -1], [cou+0.3*cou, 1, 1])
                    
                    popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=[cou, mea, st], bounds=bounds_par, maxfev = 100000)
                    print(count, 'sigma:',popt[2])
                    fit_sigma =  popt[2]
                    self.eigval[:,0] = ma.masked_greater(self.eigval[:,0], popt[1]+self.sig_val*popt[2] )
                    self.eigval[:,0] = ma.masked_less(self.eigval[:,0], popt[1]-self.sig_val*popt[2] )
                    nchan_zap = np.ma.count_masked(self.eigval)
                    print("nchan_zap:", nchan_zap)
                    count += 1

                # combine all the masked channel together
                self.freq_mask = np.any(self.eigval.mask, axis=1)
                print (np.arange(self.use_nchan)[self.freq_mask])

                #print (self.eigvec[:,:,0].shape)
                plt.clf()
                plt.figure(figsize=(9,9))
                ax = plt.subplot(211)
                ax.set_title('Spectrum')
                ax.set_xticks(np.arange(0,self.use_nchan,xtick))
                ax.set_xlim(xlim[0], xlim[1])

                spec = np.mean(self.nbarray[:,:,:], axis=1)
                for i in range(self.nbeam):
                        ax.scatter(np.arange(self.use_nchan)[self.freq_mask], spec[i, self.freq_mask]+10*i, label='beam{0}'.format(i), alpha=1, marker='+', s=60)
                        ax.plot(np.arange(self.use_nchan)[0:], spec[i, 0:]+10*i, color='k')
                ax.legend(loc='upper right', fontsize=6)

                ax.grid()
                ax = plt.subplot(212)
                ax.set_title('Mask')
                ax.set_xticks(np.arange(0,self.use_nchan,xtick))
                ax.set_xlim(xlim[0], xlim[1])

                mask = np.empty((self.nbeam, self.use_nchan))
                for i in range(self.nbeam):
                        mask[i,:] = self.freq_mask

                spec = ma.masked_array(spec, mask=mask)
                for i in range(self.nbeam):
                        ax.plot(np.arange(self.use_nchan)[0:], spec[i, 0:]+10*i, color='k')

                ax.grid()
                plt.savefig('zapping_spectrum.png')

                ###################
                plt.clf()
                plt.figure(figsize=(9,9))
                ax = plt.subplot(111)
                ax.set_title('Eig value')
                ax.set_xticks(np.arange(0,self.use_nchan,xtick))
                ax.set_xlim(xlim[0], xlim[1])
                for i in range(self.nbeam):
                        plot_eigval = ma.getdata(self.eigval)[:,i]
                        ax.scatter(np.arange(self.use_nchan)[self.eigval.mask[:,i]], plot_eigval[self.eigval.mask[:,i]]+1.0*i, alpha=1, marker='+', s=60)
                        ax.plot(np.arange(self.use_nchan)[0:], plot_eigval[0:]+1.0*i, color='k')

                ax.grid()
                plt.savefig('zapping_eig.png')


        def cal_evc_Threshold(self):
                mean_list = []
                std_list = []
                for i in range(0, self.nbeam, 1): 
                        mid = self.eigvec[:, i, 0]
                        masked = self.eigval[:,0].mask
                        mid = np.ma.array(mid, mask = masked)
                        
                        index = np.where((mid > -0.2) & (mid < 0.2))
                        data = mid[index].compressed() 

                        counts, bin_edges = np.histogram(data, bins=50)
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
                self.rfi_mask = np.ones((self.use_nchan, self.nbeam))
                for i in range(self.use_nchan):
                        if i <=0:
                                self.rfi_mask[i,:] = 0
                        elif self.eigval.mask[i,0] == 1:
                                vec = self.eigvec_array[i, :, 0]
                                beam_mask = (vec < (evc_mean - evc_sigma* self.sig_vec)) | (vec > (evc_mean + evc_sigma* self.sig_vec))
                                self.rfi_mask[i, beam_mask] = 0

                self.rfi_mask = np.array(self.rfi_mask, dtype=bool)

        def plot_spec (self, xr, xtick):
                plt.clf()
                plt.figure(figsize=(12,12))
                plt.subplots_adjust(hspace=0.05, wspace=0.05)
                i = 0
                for i in np.arange(self.nbeam):
                        plot_idx = i+1
                        ax = plt.subplot(4, 5, plot_idx)
                        ax.set_title('beam{0}'.format(plot_idx))

                        if plot_idx == 19:
                                ax.set_xticks(np.arange(0, self.use_nchan, xtick))
                                #ax.set_xticks(np.arange(0, self.use_nchan, 50))
                        else:
                                ax.set_xticks([])
                                ax.set_yticks([])

                        ax.set_xlim(xr[0], xr[1])
                        spec = np.sum(self.nbarray[i], axis=0)
                        ax.plot(np.arange(self.use_nchan), spec, color='k')

                        ax.scatter(np.arange(self.use_nchan)[self.rfi_mask[:,i]], spec[self.rfi_mask[:,i]], color='r', alpha=0.5)
                        i += 1
                plt.savefig('spectrum.png')


##########################

if __name__ == "__main__":
        ######################
        #rc('text', usetex=True)
        # read in parameters

        parser = argparse.ArgumentParser(description='Constructs covariance matrices, performs eigen-decomposition')
        parser.add_argument('-f',  '--input_file',  metavar='Input file name',  nargs='+', required=True, help='Input file name')
        parser.add_argument('-sub',  '--subband_range', metavar='Subint ragne', nargs='+', default = [0, 0], type = int, help='Subint range')
        parser.add_argument('-freq',  '--freq_range', metavar='Freq range (MHz)', nargs='+', default = [0, 0], type = int, help='Frequency range (MHz)')
        parser.add_argument('-sig_val',  '--sigma_val', metavar='Eigenvalue threshold', default = 3, type = int, help='Masking eigenvalue threshold')
        parser.add_argument('-sig_vec',  '--sigma_vec', metavar='Eigenvector threshold', default = 1, type = int, help='Masking eigenvector threshold')
        parser.add_argument('-downsamp',  '--down_sample', metavar='Down sample', default = 1, type = int, help='Down sample')
        parser.add_argument('-normal_base',  '--normalise_base', metavar='Normalise base', nargs=2, default = [2600, 2800], type = int, help='Normalise base')
        parser.add_argument('-arpls', '--arpls_par', metavar='ArPLS parameters (lam ratio itermax)', nargs=3, default=[1e3, 0.005, 35], type=float, help='ArPLS parameters (lam ratio itermax)')

        args = parser.parse_args()
        sub_start = int(args.subband_range[0])
        sub_end = int(args.subband_range[1])
        freq_start = int(args.freq_range[0])
        freq_end = int(args.freq_range[1])
        sigma_val = float(args.sigma_val)
        sigma_vec = float(args.sigma_vec)
        downsamp = int(args.down_sample)
        normal_base_start = int(args.normalise_base[0])
        normal_base_end = int(args.normalise_base[1])
        lam, ratio, itermax = args.arpls_par
        infile = args.input_file

        #read_data (infile[0], sub_start, sub_end)
        mb_ccm = ccm (
            filenames=infile, 
            sub0=sub_start, sub1=sub_end, 
            freq0=freq_start, freq1=freq_end, 
            sigma_val=sigma_val, sigma_vec=sigma_vec, 
            downsamp=downsamp, 
            normal_base_start=normal_base_start, 
            normal_base_end=normal_base_end, 
            lam=lam, ratio=ratio, itermax=int(itermax))
        mb_ccm.read_data()
        mb_ccm.normalise()

        mb_ccm.plot_data(xr=[0,4096], xtick=400)
        ##mb_ccm.sim_ccm() # instead of reading in data, simulate
        ##print (mb_ccm.ccm.shape)

        ##mb_ccm.plot_bandpass()
        mb_ccm.cal_ccm()

        mb_ccm.plot_ccm(360)
        mb_ccm.cal_eigen()

        #mb_ccm.generate_filter(xlim=[-10,4100],xtick=400)
        #mb_ccm.generate_mask()
        ##mb_ccm.plot_spec (xr=[330,350], xtick=10)
        #mb_ccm.plot_spec (xr=[1000,1600], xtick=200)
