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
#define MAX 100000
#define NSAC 100
#define N_FILENAME 100
#define MAX_PATH 100

char *strstrip(char *s); // Deletes trailing characters when reading filenames. Similar to .rtrip() in Python.
void usage();            // Show usage
void print_array(float **array, int M, int N);
const char CONFIG_FILENAME[]="config.conf";

int
main(int argc, char **argv)
{
  /* Define variables to be used in the call to rsac1() */
  float   yarray[MAX];
  float   beg, del;
  int     nlen, nerr, max = MAX, opt = 0;
  float   *data[NSAC];
  char    kname[ N_FILENAME ] ;
  char    infilename[ N_FILENAME ] ;
  FILE    *fid;
  size_t  len=0;
//  ssize_t read;
  int     count=0;

  char  *line;
  size_t line_size = 100;

  struct config_filter configstruct;
  configstruct = get_config(CONFIG_FILENAME); 

  printf("Low(int)  = %f\n",configstruct.low);
  printf("High(int)  = %f\n",configstruct.high);
  printf("Attenuation(int)  = %f\n",configstruct.attenuation);
  printf("Transition Band(int)  = %f\n",configstruct.transition_band);
  printf("Npoles  = %d\n",configstruct.npoles);
  printf("passes  = %d\n",configstruct.passes);
 
  if( argc == 1 ) {
	usage();
	exit(-1);
  } 
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
  	data[i] = (float *)malloc( MAX  * sizeof(float));  

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

print_array(data,count,nlen);
free(*data);
fclose(fid);
if (line)
        free(line);
 
  
  exit(0);
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

