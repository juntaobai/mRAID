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


class acm (read_fits):
        def __init__ (self, filenames, sub0=0, sub1=0, freq0=0, freq1=0, sigma=3.0):
                self.sub0 = sub0
                print ('Initialising a multi-beam object\n')
                super().__init__ (filenames, sub0, sub1, freq0, freq1)
                self.sigma = sigma

                #self.read_data()
                for i in range(self.nbeam):
                        print ('{0} nsub:{1} nsblk:{2} nbits:{3} nchan:{4}'.format(self.filenames[i], self.nsub, self.nsblk, self.nbits, self.nchan))

        def normalise (self, base):
                for i in range(self.nbeam):
                        std = np.std(self.nbarray[i,:,base[0]:base[1]])
                        mean = np.mean(self.nbarray[i,:,base[0]:base[1]])
                        self.nbarray[i,:,:] = (self.nbarray[i,:,:] - mean)/std

        def cal_acm1 (self):
                self.acm = np.empty((self.new_nchan, self.nbeam, self.nbeam))
                for i in range(self.new_nchan):
                        for j in range(self.nbeam):
                                for k in range(self.nbeam):
                                        self.acm[i,j,k] = np.correlate(self.nbarray[j,:,i], self.nbarray[k,:,i])    # use correlation instead of covariance


        def cal_acm2 (self):
                self.acm = np.empty((self.new_nchan, self.nbeam, self.nbeam))
                for i in range(self.new_nchan):
                        data_slice = self.nbarray[:, :, i]
                        self.acm[i] = np.array([[correlate(data_slice[j], data_slice[k], mode='valid')[0] 
                                 for k in range(self.nbeam)] 
                                 for j in range(self.nbeam)])

        def cal_acm (self):
                self.acm = np.empty((self.new_nchan, self.nbeam, self.nbeam))
                for i in range(self.new_nchan):
                        data_slice = self.nbarray[:, :, i]  
                        self.acm[i] = np.dot(data_slice, data_slice.T) 

        def sim_acm (self):
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

        def plot_acm (self, chan):
                plt.clf()
                tmp = np.array(self.acm[chan], copy=True)
                for i in range(self.nbeam):
                        tmp[i,i] = np.nan

                ax = plt.subplot(111)
                ax.imshow(tmp, origin='lower', aspect='auto')
                plt.savefig('acm.png')
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
                                ax.set_xticks(np.arange(0, self.new_nchan, xtick))
                        else:
                                ax.set_xticks([])
                                ax.set_yticks([])

                        ax.set_xlim(xr[0], xr[1])
                        lc = np.sum(self.nbarray[i], axis=1)
                        ax.imshow(self.nbarray[i], origin='lower', aspect='auto', interpolation='none')
                        i += 1
                plt.savefig('rfi_data.png')


        def ArPLS(self, y, lam=1e5, ratio=0.005, itermax=25):
            N = len(y)
            D = sparse.eye(N, format='csc')
            D = D[1:] - D[:-1]
            D = D[1:] - D[:-1]
            D = D.T
            w = np.ones(N)
            lam = lam * np.ones(N)
            for i in range(itermax):
                    W = sparse.diags(w, 0, shape=(N, N))
                    LAM = sparse.diags(lam, 0, shape=(N, N))
                    Z = W + LAM * D.dot(D.T)
                    z = spsolve(Z, w * y)
                    d = y - z
                    dn = d[d < 0]
                    m = np.mean(dn)
                    s = np.std(dn)
                    wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
                    if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                        break
                    w = wt
            return z


        def cal_eigen (self):
                self.eigval = ma.masked_array(np.empty((self.new_nchan, self.nbeam)), mask=np.zeros((self.new_nchan, self.nbeam)))
                self.freq_array = ma.masked_array(np.empty((self.new_nchan, self.nbeam)), mask=np.zeros((self.new_nchan, self.nbeam)))
                self.chan_array = ma.masked_array(np.empty((self.new_nchan, self.nbeam)), mask=np.zeros((self.new_nchan, self.nbeam)))
                self.eigvec = np.empty((self.new_nchan, self.nbeam, self.nbeam))
                #print("dat_freq:", self.dat_freq)

                for i in range(self.new_nchan):
                        u, s, vh = la.svd(self.acm[i])
                        self.eigval[i] = np.power(s, 2)/np.sum(np.power(s, 2))    # normalise eig vec
                        self.eigvec[i] = u
                        self.freq_array[i,:] = self.dat_freq[i]
                        self.chan_array[i,:] = i

                for i in range(self.nbeam):
                        self.eigval[:,i] = self.eigval[:,i] - self.ArPLS(self.eigval[:,i])
                #np.save('eigval_sub%d'%self.sub0, ma.getdata(self.eigval))
                #np.save('eigvec_sub%d'%self.sub0, ma.getdata(self.eigvec))
                #np.save('CCM_sub%d'%self.sub0, ma.getdata(self.acm))

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
                self.eigval[:,0] = ma.masked_greater(self.eigval[:,0], popt[1]+self.sigma*popt[2] )
                self.eigval[:,0] = ma.masked_less(self.eigval[:,0], popt[1]-self.sigma*popt[2] )
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
                    self.eigval[:,0] = ma.masked_greater(self.eigval[:,0], popt[1]+self.sigma*popt[2] )
                    self.eigval[:,0] = ma.masked_less(self.eigval[:,0], popt[1]-self.sigma*popt[2] )
                    nchan_zap = np.ma.count_masked(self.eigval)
                    print("nchan_zap:", nchan_zap)
                    count += 1

                # combine all the masked channel together
                self.freq_mask = np.any(self.eigval.mask, axis=1)
                print (np.arange(self.new_nchan)[self.freq_mask])

                #print (self.eigvec[:,:,0].shape)
                plt.clf()
                plt.figure(figsize=(9,9))
                ax = plt.subplot(211)
                ax.set_title('Spectrum')
                ax.set_xticks(np.arange(0,self.new_nchan,xtick))
                ax.set_xlim(xlim[0], xlim[1])

                spec = np.mean(self.nbarray[:,:,:], axis=1)
                for i in range(self.nbeam):
                        ax.scatter(np.arange(self.new_nchan)[self.freq_mask], spec[i, self.freq_mask]+10*i, label='beam{0}'.format(i), alpha=1, marker='+', s=60)
                        ax.plot(np.arange(self.new_nchan)[0:], spec[i, 0:]+10*i, color='k')
                ax.legend(loc='upper right', fontsize=6)

                ax.grid()
                ax = plt.subplot(212)
                ax.set_title('Mask')
                ax.set_xticks(np.arange(0,self.new_nchan,xtick))
                ax.set_xlim(xlim[0], xlim[1])

                mask = np.empty((self.nbeam, self.new_nchan))
                for i in range(self.nbeam):
                        mask[i,:] = self.freq_mask

                spec = ma.masked_array(spec, mask=mask)
                for i in range(self.nbeam):
                        ax.plot(np.arange(self.new_nchan)[0:], spec[i, 0:]+10*i, color='k')

                ax.grid()
                plt.savefig('zapping_spectrum.png')

                ###################
                plt.clf()
                plt.figure(figsize=(9,9))
                ax = plt.subplot(111)
                ax.set_title('Eig value')
                ax.set_xticks(np.arange(0,self.new_nchan,xtick))
                ax.set_xlim(xlim[0], xlim[1])
                for i in range(self.nbeam):
                        plot_eigval = ma.getdata(self.eigval)[:,i]
                        ax.scatter(np.arange(self.new_nchan)[self.eigval.mask[:,i]], plot_eigval[self.eigval.mask[:,i]]+1.0*i, alpha=1, marker='+', s=60)
                        ax.plot(np.arange(self.new_nchan)[0:], plot_eigval[0:]+1.0*i, color='k')

                ax.grid()
                plt.savefig('zapping_eig.png')


        def cal_evc_Threshold(self):
                Threshold = []
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
                        Threshold.append(np.abs(popt[2]) )
                evc_Threshold = np.array(Threshold)
                print(evc_Threshold)
                return Threshold

        def generate_mask (self, chn, base):
                evc_Threshold = self.cal_evc_Threshold()
                print(evc_Threshold)
                self.rfi_mask = np.ones((self.new_nchan, self.nbeam))
                for i in range(self.new_nchan):
                        if i <=0:
                                self.rfi_mask[i,:] = 0
                        elif self.eigval.mask[i,0] == 1:
                                beam_mask = np.fabs(self.eigvec[i, :, 0]) > evc_Threshold
                                self.rfi_mask[i, beam_mask] = 0
                        #else:
                        #       self.rfi_mask[i,:] = 1

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
                                ax.set_xticks(np.arange(0, self.new_nchan, xtick))
                                #ax.set_xticks(np.arange(0, self.new_nchan, 50))
                        else:
                                ax.set_xticks([])
                                ax.set_yticks([])

                        ax.set_xlim(xr[0], xr[1])
                        spec = np.sum(self.nbarray[i], axis=0)
                        ax.plot(np.arange(self.new_nchan), spec, color='k')

                        ax.scatter(np.arange(self.new_nchan)[self.rfi_mask[:,i]], spec[self.rfi_mask[:,i]], color='r', alpha=0.5)
                        i += 1
                plt.savefig('spectrum.png')


##########################

if __name__ == "__main__":
        ######################
        #rc('text', usetex=True)
        # read in parameters

        parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
        parser.add_argument('-f',  '--input_file',  metavar='Input file name',  nargs='+', required=True, help='Input file name')
        #parser.add_argument('-o',  '--output_file', metavar='Output file name', nargs='+', required=True, help='Output file name')
        parser.add_argument('-sub',  '--subband_range', metavar='Subint ragne', nargs='+', default = [0, 0], type = int, help='Subint range')
        parser.add_argument('-freq',  '--freq_range', metavar='Freq range (MHz)', nargs='+', default = [0, 0], type = int, help='Frequency range (MHz)')
        parser.add_argument('-sig',  '--sigma', metavar='Threshold', default = 3, type = float, help='Masking threshold')
        start_time = time.time()
        args = parser.parse_args()
        sub_start = int(args.subband_range[0])
        sub_end = int(args.subband_range[1])
        freq_start = int(args.freq_range[0])
        freq_end = int(args.freq_range[1])
        sigma = float(args.sigma)
        infile = args.input_file

        #read_data (infile[0], sub_start, sub_end)
        mb_acm = acm (infile, sub_start, sub_end, freq_start, freq_end, sigma)
        mb_acm.read_data()
        #mb_acm.normalise(base=[2600,2800])
        read_time = time.time()
        print('read_time:', read_time - start_time)
        #mb_acm.plot_data(xr=[0,4096], xtick=400)
        ##mb_acm.sim_acm() # instead of reading in data, simulate
        ##print (mb_acm.acm.shape)

        ##mb_acm.plot_bandpass()
        mb_acm.cal_acm()
        cal_ccm_time = time.time()
        print('cal_ccm_time:', cal_ccm_time - read_time, cal_ccm_time - start_time)
        #mb_acm.plot_acm(360)
        mb_acm.cal_eigen()
        cal_eig_time = time.time()
        print('cal_eig_time:', cal_eig_time - cal_ccm_time, cal_eig_time - start_time)
        #mb_acm.generate_filter(xlim=[-10,4100],xtick=400)
        #mb_acm.generate_mask(chn=4096, base=[2600,2800])
        ##mb_acm.plot_spec (xr=[330,350], xtick=10)
        #mb_acm.plot_spec (xr=[1000,1600], xtick=200)
