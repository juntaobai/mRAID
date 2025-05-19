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
        def __init__(self, filenames, hdf5_filename, nsub, sub_step, nchan, new_nchan, nbeam, sigma=3.0):
                self.filenames = filenames
                self.hdf5_filename = hdf5_filename
                self.sigma = sigma
                self.nsub = nsub
                self.nchan = nchan
                self.new_nchan = new_nchan
                self.nbeam = nbeam
                self.sub_step = sub_step
                self.npart = int(np.ceil(self.nsub / self.sub_step))

                self.eigval_array = ma.masked_array(np.empty((self.npart, self.new_nchan, self.nbeam)), mask=np.zeros((self.npart, self.new_nchan, self.nbeam)))
                self.chan_array = ma.masked_array(np.empty((self.npart, self.new_nchan, self.nbeam)), mask=np.zeros((self.npart, self.new_nchan, self.nbeam)))
                self.eigvec_array = np.empty((self.npart, self.new_nchan, self.nbeam, self.nbeam))

                for i in range(self.new_nchan):
                        self.chan_array[:, i, :] = i

                with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
                        for i in range(self.npart):
                                sub_start = int(i * self.sub_step)
                                sub_end = min(sub_start + self.sub_step, self.nsub)

                                eigval_dataset = f'eigval_sub{sub_start}_to_{sub_end}'
                                eigvec_dataset = f'eigvec_sub{sub_start}_to_{sub_end}'

                                self.eigval_array[i, :, :] = hdf5_file[eigval_dataset][:]
                                self.eigvec_array[i, :, :, :] = hdf5_file[eigvec_dataset][:]

        def gaussian(self, x, a, mean, sigma):
                return a * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
            
        def plot_eigval (self, beam=0):
                print('eigval_shape:', (self.eigval_array[:,:,beam]).shape)
                plt.clf()
                fig = plt.figure(figsize=(8,6))
                ax = plt.subplot(111)
                cax = ax.imshow(self.eigval_array[:,:,beam], origin='lower', aspect='auto', interpolation='none')
                cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.02)
                ax.set_xlabel(f"Channel", fontsize=14)
                ax.set_ylabel(f"Subinterval", fontsize=14)
                start_x = int((self.nchan-self.new_nchan)/2)
                ax.set_xticks(np.arange(0, self.new_nchan, 500), np.arange(start_x, self.new_nchan+start_x, 500))
                plt.savefig(f'{hdf5_filename}_masking_plot_eigval.png')
                

        def generate_filter (self, eig_idx):
                nchan_zap0 = np.ma.count_masked(self.eigval_array)
                mid = self.eigval_array[:,:,0].ravel()
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
                self.eigval_array[:,:,0] = ma.masked_greater(self.eigval_array[:,:,0], popt[1]+self.sigma*popt[2] )
                self.eigval_array[:,:,0] = ma.masked_less(self.eigval_array[:,:,0], popt[1]-self.sigma*popt[2] )
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
                    self.eigval_array[:,:,0] = ma.masked_greater(self.eigval_array[:,:,0], popt[1]+self.sigma*popt[2] )
                    self.eigval_array[:,:,0] = ma.masked_less(self.eigval_array[:,:,0], popt[1]-self.sigma*popt[2] )
                    nchan_zap = np.ma.count_masked(self.eigval_array)
                    print("nchan_zap:", nchan_zap)
                    count += 1


        def cal_evc_Threshold(self):
                Threshold = []
                for i in range(0, self.nbeam, 1): 
                        mid = self.eigvec_array[:,:, i, 0]
                        masked = self.eigval_array[:,:,0].mask
                        mid = np.ma.array(mid, mask = masked)
                        
                        index = np.where((mid > -0.2) & (mid < 0.2))
                        data = mid[index].compressed() 

                        counts, bin_edges = np.histogram(data, bins=200)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        mea, st = np.abs(np.mean(data)), np.abs(np.std(data))
                        cou = max(counts)
                        bounds_par = ([cou-0.3*cou, -1, -1], [cou+0.3*cou, 1, 1])
                        popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=[cou, mea, st], bounds=bounds_par, maxfev = 100000)
                        Threshold.append(np.abs(popt[2]) )
                evc_Threshold = np.array(Threshold)
                print(evc_Threshold)
                return Threshold
            

        def generate_mask (self, eig_idx):
                evc_Threshold = self.cal_evc_Threshold()
                self.rfi_mask = np.ones((self.npart, self.new_nchan, self.nbeam))
                eig_mask = self.eigval_array.mask[:,:,eig_idx]
                plt.clf()
                fig = plt.figure(figsize=(8,6))
                ax = plt.subplot(111)
                cax = plt.imshow(self.eigval_array[:,:,0], origin='lower', aspect='auto', rasterized=True, interpolation='none')
                cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.02)
                ax.set_xlabel(f"Channel", fontsize=14)
                ax.set_ylabel(f"Subinterval", fontsize=14)
                start_x = int((self.nchan-self.new_nchan)/2)
                ax.set_xticks(np.arange(0, self.new_nchan, 500), np.arange(start_x, self.new_nchan+start_x, 500))
                plt.savefig(f'{hdf5_filename}_test.png')
                
                init_mask = np.moveaxis(np.array(self.nbeam*[eig_mask]), 0, -1)  
                for j in range(self.npart):
                        for i in range(self.new_nchan):
                                if self.eigval_array.mask[j,i,eig_idx] == 1:
                                        beam_mask = np.fabs(self.eigvec_array[j, i, :, eig_idx]) > evc_Threshold 
                                        self.rfi_mask[j, i, beam_mask] = 0
                #print("self.rfi_mask.dtype:", self.rfi_mask.dtype)
                self.all_rfi_mask = np.zeros((self.npart, self.nchan, self.nbeam))
                freq_s = int((self.nchan-self.new_nchan)/2)+1
                freq_e = -int((self.nchan-self.new_nchan)/2)
                self.all_rfi_mask[:, freq_s:freq_e, :] = self.rfi_mask
                threshold_chan = 0.8
                for j in range(self.nbeam):
                        for i in range(self.nchan):  
                                rfi_fraction = np.sum(self.all_rfi_mask[:, i, j] == 0) / self.npart
                                
                                if rfi_fraction > threshold_chan:
                                        self.all_rfi_mask[:, i, j] = 0  
                self.all_rfi_mask = np.array(self.all_rfi_mask, dtype=bool)
                np.save('rfi_mask.npy', self.all_rfi_mask )
                print ("4:", np.array_equal(self.all_rfi_mask[:,:,0], self.all_rfi_mask[:,:,-1]))


        def plot_result (self, beam, xr, xtick):
                print ("5:", self.eigval_array[:,:,beam].shape)
                plt.clf()              
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
                                ax.set_xticks(np.arange(0, self.new_nchan, xtick))
                                ax.set_xlabel('Channel number', fontsize=14)

                        ax.set_xlim(xr[0], xr[1])
                        ax.imshow(self.all_rfi_mask[:,:,beam_id], origin='lower', aspect='auto', interpolation='none')#, cmap='gray')
                        num_zeros = np.count_nonzero(self.all_rfi_mask[:,:,beam_id] == 0)
                        num_ones = np.count_nonzero(self.all_rfi_mask[:,:,beam_id] == 1)
                        countt = (self.nchan-self.new_nchan)*self.npart
                        print(beam_id, num_zeros, num_ones, num_zeros + num_ones, (num_zeros - countt)/ (num_zeros + num_ones - countt) )
                        i += 1

                plt.savefig(f'{hdf5_filename}_masking_plot_eigval_masked.png')

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
                        maskfile = basename[beam_id] +'.mask'
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
                        print('basename:', basename[beam_id])
                        hdu = fits.open(basename[beam_id])
                        nchan = np.int32(hdu[1].header['nchan'])
                        samppersubint = int(hdu[1].header['NSBLK'])
                        nint = np.int32(self.npart)
                        ptsperint = samppersubint * self.nsub / self.sub_step
                        ptsperint = np.int32(ptsperint)
                        sf = self.all_rfi_mask[:,:,beam_id]
                        sf = sf.astype('float64')
                        statsfile = basename[beam_id] +'.stats'
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
        parser.add_argument('-nchn', '--nchan', metavar='Total num of channel', default=4096, type=int, help='Total number of channel')
        parser.add_argument('-new_nchn', '--new_nchan', metavar='num of channel used', default=3277, type=int, help='number of channel used')
        parser.add_argument('-nbeam', '--nbeam', metavar='Total num of beam', default=19, type=int, help='Total number of beam')
        parser.add_argument('-step', '--sub_step', metavar='Step in subintegration', default=32, type=int, help='Step in subintegration')
        parser.add_argument('-sig', '--sigma', metavar='Threshold', default=3.0, type=float, help='Masking threshold')
        parser.add_argument('-fh5', '--input_hdf5_file', metavar='Input HDF5 file name', default='./source_10.h5', help='Input HDF5 file name')
        parser.add_argument('-f',  '--input_file',  metavar='Input file name',  nargs='+', default = [], help='Input file name')

        args = parser.parse_args()
        sigma = int(args.sigma)
        nbeam = int(args.nbeam)
        nchan = int(args.nchan)
        new_nchan = int(args.new_nchan)
        nsub = int(args.nsubint)
        sub_step = int(args.sub_step)
        hdf5_filename = args.input_hdf5_file
        filenames = args.input_file

        eig = eig( filenames=filenames, hdf5_filename=hdf5_filename, sub_step=sub_step, nsub=nsub, nchan=nchan, new_nchan=new_nchan, nbeam=nbeam, sigma=sigma)
        eig.plot_eigval(beam=0)
        eig.generate_filter(eig_idx=1)
        eig.generate_mask(eig_idx=0)
        eig.plot_result(beam=np.arange(19), xr=[0, 4096], xtick=1000)
        eig.write_mask()
        eig.write_stats()
