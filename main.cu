// To Compile it
// gcc -m32 -o test.out test.c sac.a
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <math.h>
#include "crsmex.h"
#include <cuda.h>
#include <cufft.h>
extern "C"{
#include <sacio.h>
#include <sac.h>
}

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Define the maximum length of the data array */
#define MAX_ARRAY  100000
#define NSAC       20
#define N_FILENAME 100
#define MAX_PATH   100

#define GRID_SIZE  1
#define BLOCK_SIZE 4

//char *strstrip(char *s); // Deletes trailing characters when reading filenames. Similar to .rtrip() in Python.
void usage();            // Show usage
void print_array(float **array, int nsac, int npts, int step);
void print_array(float **array, int M, int N);
void print_fft(  cufftComplex *fft, int batch, int size_fft);
void check_gpu_card_type(void);
void plot_array(float **array, int M, int N);
void plot_fft(int N);
void run_unit_test();
const char CONFIG_FILENAME[]="config.conf";

__global__  void find_repeaters(float *data, int npts);

int main(int argc, char **argv)
{
  /* Define variables to be used in the call to rsac1() */
  float     yarray[MAX_ARRAY];
  float     beg, del;
  int       nlen, nerr, max = MAX_ARRAY, opt = 0;
  float     *data;
  char      kname[ N_FILENAME ] ;
  char      infilename[ N_FILENAME ] ;
  FILE      *fid;
  size_t    len=0;
  int       count=0;
  cufftReal *device_data;

  char      *line;
  size_t    line_size = 100;

  /* Filtering variables */
  struct config_filter configstruct;
  configstruct = get_config(CONFIG_FILENAME); 

  /* CUDA configuration */

  int grdSize   = GRID_SIZE;
  int blockSize = BLOCK_SIZE;

  dim3 dimGrid(grdSize, grdSize, grdSize);
  dim3 dimBlock(blockSize, blockSize, blockSize);
   


 /*
  printf("Low(int)  = %f\n",configstruct.low);
  printf("High(int)  = %f\n",configstruct.high);
  printf("Attenuation(int)  = %f\n",configstruct.attenuation);
  printf("Transition Band(int)  = %f\n",configstruct.transition_band);
  printf("Npoles  = %d\n",configstruct.npoles);
  printf("passes  = %d\n",configstruct.passes);
 */

  if( argc == 1 ) {
	usage();
	exit(-1);
  }

  // Check is a GPU card is available.
  check_gpu_card_type();

  // Retrieve input parameters 
  while((opt = getopt(argc, argv, "f:t")) != -1){
	switch(opt){
	      case 't':
		run_unit_test();
		exit(-1);
		break;
	      case 'f':
		strncpy(infilename, optarg, MAX_PATH);
		break;
	default:
		fprintf(stderr, "Unknown option %c\n\n",opt);
		usage();
		exit(-1);
        }
  }

  line = (char  *)malloc(line_size    * sizeof(char));

//  for (int i=0; i<NSAC; i++)
//  	data[i] = (float *)malloc( MAX_ARRAY  * sizeof(float));  

  data = (float *)malloc(NSAC * MAX_ARRAY * sizeof(float));	

  // Read input filenames.
  fid = fopen(infilename,"r");
  if (fid == NULL){
	fprintf(stderr,"Couldn't open file %s\n",infilename);
	exit(-1);
  } 
 

 // Read sac files into host memory.
 while (getline(&line, &len, fid) != -1)
  {
	line = strstrip(line);
        strcpy ( kname ,line ) ;
        rsac1( kname, yarray, &nlen, &beg, &del, &max, &nerr, strlen( kname ) ) ;
        if ( nerr != 0 ) {
                fprintf(stderr, "Error reading in SAC file: %s\n", kname);
                exit ( nerr ) ;
        }
	else {
    		fprintf(stderr,"Reading SUCCESS: %s\n",kname);
        	fprintf(stderr,"Number of samples read: %d\n\n",nlen);
	}
         /* START - FILTERING */
    /*     Call xapiir ( Apply a IIR Filter ) 
     *        - yarray - Original Data 
     *        - nlen   - Number of points in yarray 
     *        - proto  - Prototype of Filter 
     *                 - SAC_FILTER_BUTTERWORK        - Butterworth 
     *                 - SAC_FILTER_BESSEL            - Bessel 
     *                 - SAC_FILTER_CHEBYSHEV_TYPE_I  - Chebyshev Type I 
     *                 - SAC_FILTER_CHEBYSHEV_TYPE_II - Chebyshev Type II 
     *        - transition_bandwidth (Only for Chebyshev Filter) 
     *                 - Bandwidth as a fraction of the lowpass prototype 
     *                   cutoff frequency 
     *        - attenuation (Only for Chebyshev Filter) 
     *                 - Attenuation factor, equals amplitude reached at 
     *                   stopband egde 
     *        - order  - Number of poles or order of the analog prototype 
     *                   4 - 5 should be ample 
     *                   Cannot exceed 10 
     *        - type   - Type of Filter 
     *                 - SAC_FILTER_BANDPASS 
     *                 - SAC_FILTER_BANDREJECT 
     *                 - SAC_FILTER_LOWPASS 
     *                 - SAC_FILTER_HIGHPASS 
     *        - low    - Low Frequency Cutoff [ Hertz ] 
     *                   Ignored on SAC_FILTER_LOWPASS 
     *        - high   - High Frequency Cutoff [ Hertz ] 
     *                   Ignored on SAC_FILTER_HIGHPASS 
     *        - delta  - Sampling Interval [ seconds ] 
     *        - passes - Number of passes 
     *                 - 1 Forward filter only 
     *                 - 2 Forward and reverse (i.e. zero-phase) filtering 
     */

/*
    xapiir(yarray, nlen, (char *)SAC_BUTTERWORTH, 
           configstruct.transition_band, configstruct.attenuation, 
           configstruct.npoles, 
           (char *)SAC_HIGHPASS, 
           configstruct.low, configstruct.high, 
           del, configstruct.passes);
     /* END */
     memcpy(&data[count*MAX_ARRAY], yarray, nlen*sizeof(float));
     count++;
  }

  /* CUDA FFT */
  nlen          = 32768; // Test value - delete after test.

  cufftHandle   plan;
  cufftComplex *fft_data;
  cufftComplex *hostOutputFFT;
  int rank      = 1;                                 // --- 1D FFTs
  int n[]       = { nlen };                          // --- Size of the Fourier transform
  int istride   = 1, ostride = 1;                    // --- Distance between two successive input/output elements
  int idist     = MAX_ARRAY, odist = (nlen / 2 + 1); // --- Distance between batches
  int inembed[] = { 0 };                             // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };                             // --- Output size with pitch (ignored for 1D transforms)
  int size_fft  = (nlen / 2 + 1);
  int batch = count;                                 // --- Number of batched executions

  printf(" ********** CONFG *********\n");
  printf(" rank     = %d\n", rank       );
  printf(" n[0]     = %d\n", n[0]       );
  printf(" inembed  = %d\n", inembed[0] );
  printf(" istride  = %d\n", istride    );
  printf(" onembed  = %d\n", onembed[0] );
  printf(" ostride  = %d\n", ostride    );
  printf(" odist    = %d\n", odist      );
  printf(" batch    = %d\n", batch      );
  printf(" count    = %d\n", count      );
  printf(" size_fft = %d\n", size_fft   );
  printf(" **************************\n");
 
  // Initiazilizing device data for fft processing
  gpuErrchk(cudaMalloc((void**)&device_data,       MAX_ARRAY * count * sizeof(cufftReal   )));
  gpuErrchk(cudaMalloc((void**)&fft_data,           size_fft * count * sizeof(cufftComplex)));
  hostOutputFFT = (cufftComplex*)malloc(            size_fft * count * sizeof(cufftComplex));
  gpuErrchk(cudaMemcpy(device_data, data,          MAX_ARRAY * count * sizeof(float)         , cudaMemcpyHostToDevice));

  
  
  cufftPlanMany(&plan, rank, n, 
                inembed, istride, idist,
                onembed, ostride, odist, CUFFT_R2C, batch);

  cufftExecR2C(plan, device_data, fft_data);

  gpuErrchk(cudaMemcpy(hostOutputFFT, fft_data, size_fft * count * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

  printf(" %f %f\n",  hostOutputFFT[0].x,hostOutputFFT[0].y );
  print_fft(hostOutputFFT, batch, size_fft);
  plot_fft(batch);
  //print_array(data,count,nlen);
  /*
    cudaMemcpy2DToArray(device_data, 
                    0, 
                    0,  
                    data,
                    MAX_ARRAY * sizeof(float),  
                    nlen      * sizeof(float), 
                    count     * sizeof(float),  cudaMemcpyHostToDevice);
*/

printf("n = %d\n", n[0]);

find_repeaters<<<count, nlen >>> (device_data, nlen);


/* Closing */
gpuErrchk(cudaFree(device_data));
gpuErrchk(cudaFree(fft_data));
cufftDestroy(plan);
free(data);
fclose(fid);
if (line)
        free(line);

cudaDeviceReset();  
return EXIT_SUCCESS;
}

__global__ void find_repeaters(float *data,int npts){
__shared__ float* trace;

trace = (float *)malloc(npts*sizeof(float));

for(int currentBlockOfPoints = 0; currentBlockOfPoints < gridDim.x; currentBlockOfPoints++)
	trace[threadIdx.x] = data[threadIdx.x + currentBlockOfPoints*npts];
	
}
/*
// Strips trailing characters
char *strstrip(char *s)
{
        size_t size;
        char *end;

        size = strlen(s);

        if (!size)
                return s;

        end = s + size - 1;
        while (end >= s && isspace(*end))
                end--;
        *(end + 1) = '\0';

        while (*s && isspace(*s))
                s++;

        return s;
}
*/
void usage(){
fprintf(stderr,"\nCUDA CRSMEX   -  Characteristic Repeating Earthquakes Code \n\n");
fprintf(stderr," This program looks for characteristic repeating earthquakes using GPU/CUDA\n");
fprintf(stderr," Required options:\n");
fprintf(stderr,"                 -f  filenames.dat - filenames.dat must containt a list of all files to be analyzed.\n\n");
fprintf(stderr,"        Author: Luis A. Dominguez - ladominguez@ucla.edu\n\n");

}

void print_fft(cufftComplex *fft, int batch, int size_fft)
{
FILE *fout;
// print out individual files
char filename[] = "outputX.dat";
	for (int i = 0; i < batch; i++){
		filename[6] = i + '0';
		printf("Writting file: %s\n", filename);
		fout = fopen(filename, "w");
		fprintf(stdout, "data size = %d\n", size_fft);
		
		for (int j = 0; j < size_fft; j++){
			fprintf(fout, "%f %f %f \n", fft[i*size_fft + j].x, fft[i*size_fft + j].y, 
                                  sqrt(fft[i*size_fft + j].x*fft[i*size_fft + j].x + fft[i*size_fft + j].y*fft[i*size_fft + j].y));
			
		}
		fclose(fout);
	}

}


//fclose(fout);

void plot_fft(int N)
{
FILE   *gnuplot = NULL;
char filename[] = "outputX.dat";
gnuplot=popen("gnuplot","w");
fprintf(gnuplot,"set term postscript eps enhanced color\n");

        for(int i=0; i<N; i++ ){
		filename[6] = i + '0';
		fprintf(stdout, "Plot array using gnuplot - %s\n", filename);
		fprintf(gnuplot, "set logscale xz\n");
                fprintf(gnuplot, "set output 'graphics_fft_%i.eps'\n", i);
                fprintf(gnuplot, "plot '%s' u 3 with lines\n", filename);
                fprintf(gnuplot, "set output\n");
                fflush(gnuplot);
        }
        pclose(gnuplot);

}

void print_array(float **array, int M, int N)
{
FILE *fout;
fprintf(stdout,"M = %d\n",M);
fprintf(stdout,"N = %d\n",N);

fout = fopen("data.dat","w");
	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++)
			fprintf(fout,"%8.3f ",array[i][j]);
		fprintf(fout,"\n");
	}

fprintf(stdout, "Writing file data.dat\n");
fclose(fout);
}

void print_array(float *array, int nsac, int npts, int step)
{
FILE *fout;
fprintf(stdout,"nsac = %d\n", nsac);
fprintf(stdout,"npts = %d\n", npts);

fout = fopen("data.dat","w");
        for (int i = 0; i < npts; i++){
                for (int j = 0; j < nsac; j++)
                        fprintf(fout,"%8.3f ",array[j*step + i]);
                fprintf(fout,"\n");
        }

fprintf(stdout, "Writing file data.dat\n");
fclose(fout);
}


void run_unit_test(){

float *data;
int   nsac = 0;
int   npts = 0;
int   N    = 3; // Only memory for three waveforms is reserved.
char  filename_test[]="./unit_test/unit_test.dat";
struct config_filter configstruct;
int win_size = 512;

configstruct = get_config(CONFIG_FILENAME);

fprintf(stdout,"\n*** RUNING TEST UNIT ***\n\n");
data = (float *)malloc(N * MAX_ARRAY * sizeof(float));
load_sac_in_host_memory(data, filename_test, &nsac, &npts, win_size, true,  configstruct);
print_array(data, nsac, npts, MAX_ARRAY);
}

void check_gpu_card_type()
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (nDevices == 0){
	fprintf(stderr,"ERROR - No GPU card detected.\n");
	exit(-1);
  }

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("            Device Number: %d\n", i);
    printf("              Device name: %s\n",            prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",            prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",            prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

