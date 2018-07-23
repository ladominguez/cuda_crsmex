// To Compile it
// gcc -m32 -o test.out test.c sac.a
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include "crsmex.h"
#include <cuda.h>
extern "C"{
#include <sacio.h>
#include <sac.h>
}

/* Define the maximum length of the data array */
#define MAX_ARRAY 100000
#define NSAC 100
#define N_FILENAME 100
#define MAX_PATH 100

#define GRID_SIZE  1
#define BLOCK_SIZE 4

char *strstrip(char *s); // Deletes trailing characters when reading filenames. Similar to .rtrip() in Python.
void usage();            // Show usage
void print_array(float **array, int M, int N);
void check_gpu_card_type(void);
const char CONFIG_FILENAME[]="config.conf";

__device__ void initDeviceVectors(int *vecA, int lL);

int
main(int argc, char **argv)
{
  /* Define variables to be used in the call to rsac1() */
  float   yarray[MAX_ARRAY];
  float   beg, del;
  int     nlen, nerr, max = MAX_ARRAY, opt = 0;
  float   *data[NSAC];
  float   *device_data;
  char    kname[ N_FILENAME ] ;
  char    infilename[ N_FILENAME ] ;
  FILE    *fid;
  size_t  len=0;
  int     count=0;

  char  *line;
  size_t line_size = 100;

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

  check_gpu_card_type();
 
  while((opt = getopt(argc, argv, "f:")) != -1){
	switch(opt){
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
  for (int i=0; i<NSAC; i++)
  	data[i] = (float *)malloc( MAX_ARRAY  * sizeof(float));  

  fid = fopen(infilename,"r");
  if (fid == NULL){
	fprintf(stderr,"Couldn't open file %s\n",infilename);
	exit(-1);
  } 
 //while ((read = getline(&line, &len, fid)) != -1)
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
    xapiir(yarray, nlen, (char *)SAC_BUTTERWORTH, 
           configstruct.transition_band, configstruct.attenuation, 
           configstruct.npoles, 
           (char *)SAC_HIGHPASS, 
           configstruct.low, configstruct.high, 
           del, configstruct.passes);
       /* END */
	memcpy(data[count],yarray,nlen*sizeof(float));
	count++;
  }

cudaMalloc(&device_data,count*nlen*sizeof(float));
cudaMemcpy(data,device_data, count*nlen*sizeof(float), cudaMemcpyHostToDevice);
cudaFree(device_data);

print_array(data,count,nlen);
free(*data);
fclose(fid);
if (line)
        free(line);
 
cudaDeviceReset();  
return EXIT_SUCCESS;
}

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

void usage(){
fprintf(stderr,"\nCUDA CRSMEX   -  Characteristic Repeating Earthquakes Code \n\n");
fprintf(stderr," This program looks for characteristic repeating earthquakes using GPU/CUDA\n");
fprintf(stderr," Required options:\n");
fprintf(stderr,"                 -f  filenames.dat - filenames.dat must containt a list of all files to be analyzed.\n\n");
fprintf(stderr,"        Author: Luis A. Dominguez - ladominguez@ucla.edu\n\n");

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
fprintf(stdout, "Writing fie data.dat\n");
fclose(fout);
}

__device__ void initDeviceVectors(int *vecA, int lL){
	
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
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

