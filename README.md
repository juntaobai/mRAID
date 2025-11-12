**mRAID (multi-beam RAdio frequency Interference Detector)**



RFI codes for the multibeam system in general (or cryoPAF)

Code written by Shi Dai and Juntao Bai.

See the paper with the description of the concept at: 


Examples:

python run_rfi.py -f *.fits -step 20 -freq 1050 1450 -nsub 256 -normal_base 2600 2800 -i 0 -o mRAID_test

python create_mask.py -nsub 256 -nbeam 19 -step 20 -sig_val 3 -sig_vec 1 -fh5 mRAID_test.h5 -f *.fits
