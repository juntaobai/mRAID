mRAID (multi-beam RAdio frequency Interference Detector)

RFI codes for the multibeam system in general (or cryoPAF)

Code written by Shi Dai and Juntao Bai.

See paper with the description of the concept at: 


Examples:

python run_rfi.py -f *.fits -step 20 -freq 1050 1450 -nsub 1000 -cpu 5 -o test.h5

python create_mask.py -f *.fits -fh5 test.h5 -nsub 1000 -nbeam 19 -step 20 -sig_val 3 -sig_vec 1 
