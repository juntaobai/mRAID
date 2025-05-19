# mRAID_rfi
RFI codes for the multibeam system

Examples:

python run_rfi.py -f *.fits -step 32 -freq 1050 1450 -sig 3.0 -nsub 8192 -cpu 10 -o test.h5

python create_mask.py -nsub 8192 -nchn 4096 -new_nchn 3277 -nbeam 19 -step 32 -sig 3.0 -fh5 test.h5 -f *.fits
