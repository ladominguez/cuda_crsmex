test_correlation: test_correlation.cu
	nvcc -arch=sm_30 -lcufft test_correlation.cu -o test_correlation -L/usr/local/sac/lib  -lsacio -lsac -lm -lcufft -I/usr/local/sac/include -I../include
