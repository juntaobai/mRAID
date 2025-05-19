import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import glob
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
from scipy import sparse
from scipy.sparse.linalg import spsolve


class read_fits ():
        def __init__ (self, filenames, sub0=0, sub1=32, freq0=0, freq1=0):
                self.filenames = filenames
                self.sub0 = sub0
                self.sub1 = sub1
                self.time = int(1)

                self.nbeam = len(filenames)

                # here I assume that all beams use the same observational setup
                hdulist = fits.open(self.filenames[0], mode='readonly', memmap=True)
                self.obsbw = hdulist['PRIMARY'].header['OBSBW']
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
                        self.freq_mask = self.dat_freq > freq0
                else:
                        self.freq_mask = (self.dat_freq > freq0) & (self.dat_freq < freq1)
                        print(self.freq_mask, len(self.freq_mask), type(self.freq_mask))
                #np.save('freq_mask.npy', self.freq_mask )
                self.new_nchan = np.sum(self.freq_mask)
                self.dat_freq = self.dat_freq[self.freq_mask]

                if self.sub0 == 0 and self.sub1 == 0:
                        self.nsamp = self.nsub*self.nsblk
                        self.nbarray = np.empty((self.nbeam, self.nsblk*self.nsub/self.time, self.new_nchan))
                else:
                        self.nsamp = (self.sub1-self.sub0)*self.nsblk
                        self.nbarray = np.empty((self.nbeam, int(self.nsblk*(self.sub1-self.sub0)/self.time), self.new_nchan))

        def ArPLS(self, y, lam=1e5, ratio=0.005, itermax=35):
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
                    #lam = lam * (1-wt)
                    if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                        break
                    w = wt
            return z

        def baseline_ArPLS(self, data):
            self.data = data
            baseline = self.ArPLS(self.data.mean(axis=0))

            self.data -= baseline

            return self.data

        def read_data (self):
                for i in range(self.nbeam):
                        hdulist = fits.open(self.filenames[i], mode='readonly', memmap=True)
                        tbdata = hdulist['SUBINT'].data
                        hdulist.close()
                        #print ('Read in data...')
                        #print ('{0} nsub:{1} nsblk:{2} nbits:{3} nchan:{4}'.format(self.filenames[i], self.nsub, self.nsblk, self.nbits, self.new_nchan))

                        if self.sub0 == 0 and self.sub1 == 0:
                                data = np.squeeze(tbdata['DATA'])
                                data = np.reshape(data, (self.nsub*self.nsblk, int(self.nchan/(8/self.nbits))))
                        else:
                                data = np.squeeze(tbdata['DATA'][self.sub0:self.sub1, :, :, :])
                                data = np.reshape(data, ((self.sub1-self.sub0)*self.nsblk, self.npol, int(self.nchan/(8/self.nbits))))   
                                data = (data[:,0,:]+data[:,1,:]).squeeze()  
                                data = data[:, self.freq_mask]
                                data = data.reshape(int((self.sub1-self.sub0)*self.nsblk/self.time), self.time, self.new_nchan).mean(axis=1)
                                data = data.astype(np.float64)
                                data = self.baseline_ArPLS(data)
                                data = data.astype(np.float64)
                                #print(data.dtype)

                        #############################################################
                        self.nbarray[i] = data
                        #print ('Shape of the unpacked data: {0}\n'.format(self.nbarray[i].shape))
                #np.save('raw_data_ArPLS.npy', self.nbarray )

        def plot_bandpass (self, beam=np.nan):
                plt.clf()
                ax1 = plt.subplot(111)
                ax1.set_xlabel('Channel')
                ax1.set_xlim(0, self.nchan)
                ax1.set_xticks(np.arange(0,self.nchan,200))

                ax2 = ax1.twiny()
                ax2.set_xlabel('Frequency (MHz)')
                ax2.set_xlim(0, self.nchan)
                ax2.set_xticks(np.arange(0,self.nchan,200))
                ax2.set_xticklabels(np.round(self.dat_freq[0] - np.arange(0,self.nchan,200)*self.obsbw/self.nchan,1))

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
        parser.add_argument('-f',  '--input_file',  metavar='Input file name', nargs='+',  required=True, help='Input file name')
        parser.add_argument('-sub0',  '--subband_start', metavar='Starting subband', required=True, help='Starting subband')
        parser.add_argument('-sub1',  '--subband_end', metavar='Ending subband',  required=True, help='Ending subband')
        #parser.add_argument('-t',  '--time_factor', metavar='Time factor',  required=True, help='Time factor')

        args = parser.parse_args()
        sub_start = int(args.subband_start[0])
        sub_end = int(args.subband_end[0])
        #time = int(args.time_factor)
        #dm = float(args.chn_dm[0])
        infile = args.input_file

        srch = read_fits (infile, sub_start, sub_end)
        #print("dat_freq:", self.dat_freq)
        srch.read_data()

        #srch.plot_bandpass(0)
