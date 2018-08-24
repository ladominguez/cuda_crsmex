#include "crsmex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
extern "C"{
#include <sacio.h>
#include <sac.h>
}
#define MAX_ARRAY  100000
#define MAXBUF     1024
#define DELIM      "="
#define N_FILENAME 100

struct config_filter get_config(const char *filename)
{
        struct config_filter configstruct;
        FILE *file = fopen (filename, "r");

        if (file != NULL)
        {
                char line[MAXBUF];
                int i = 0;

                while(fgets(line, sizeof(line), file) != NULL)
                {
                        char *cfline;
                        cfline = strstr((char *)line,DELIM);
                        cfline = cfline + strlen(DELIM);

                        if (i == 0){
				configstruct.low = atof(cfline);
                        } else if (i == 1){
				configstruct.high = atof(cfline);
                        } else if (i == 2){
				configstruct.attenuation = atof(cfline);
                        } else if (i == 3){
				configstruct.transition_band = atof(cfline);
                        } else if (i == 4){
				configstruct.npoles = atoi(cfline);
                        } else if (i == 5){
				configstruct.passes = atoi(cfline);
			}

                        i++;
                } // End while
                fclose(file);
        } // End if file
	else{
		fprintf(stderr, "ERROR - Could not open configuration file %s\n",filename);
	}



        return configstruct;

}

void load_sac_in_host_memory(float *data_host, char *infilename, int *nsac, int *npts, bool filter, config_filter configstruct){
char    *line;
size_t  len = 0;
int     nlen, nerr, max = MAX_ARRAY, opt = 0;
int     nzyear, nzday, nzhour, nzmin, nzsec;
int     month, day;
FILE    *fid;
float   yarray[MAX_ARRAY];
char    kname[ N_FILENAME ];
float   beg, del;
int     count = 0;
*nsac = 0;

fid = fopen(infilename,"r");
if (fid == NULL){
        fprintf(stderr,"Couldn't open file %s\n",infilename);
        exit(-1);
}

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
                fprintf(stderr,"Number of samples read: %d\n",nlen);
        }

	getnhv ((char *) "NZYEAR" , & nzyear , &nerr , 6 ) ;		
	getnhv ((char *) "NZJDAY" , & nzday  , & nerr , strlen("NZJDAY") ) ;		
	getnhv ((char *) "NZHOUR" , & nzhour , & nerr , strlen("NZHOUR") ) ;		
	getnhv ((char *) "NZMIN"  , & nzmin  , & nerr , strlen("NZMIN" ) ) ;		
        getnhv ((char *) "NZSEC"  , & nzsec  , & nerr , strlen("NZSEC" ) ) ;		
        julian2mmdd(nzyear, nzday, &month, &day);
        fprintf(stdout, "Julian: %d  %d/%d/%d %d:%d:%d\n", nzday, nzyear, month, day, nzhour, nzmin, nzsec);
	fprintf(stdout, "\n");
   //     if (filtering){
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

    if (filter){
   	 xapiir(yarray, nlen, (char *)SAC_BUTTERWORTH,
        	configstruct.transition_band, configstruct.attenuation,
           	configstruct.npoles,
           	(char *)SAC_HIGHPASS,
           	configstruct.low, configstruct.high,
           	del, configstruct.passes);
    }
     /* END */
//     }
     memcpy(&data_host[*nsac*MAX_ARRAY], yarray, nlen*sizeof(float));
     ++*nsac;
  }

*npts = nlen;
}

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

void julian2mmdd(int year, int julian, int *mm, int *day){
int leap = 0;
int nyear[2][12]={
{0, 31, 59,  90,   120,   151,   181,   212,   243,   273,   304,   334},
{0, 31, 60,  91,   121,   152,   182,   213,   244,   274,   305,   335}
};

if ( year % 4 == 0 ){
	leap = 1;
}

*day = 0;
for (int k=1; k < 12; k++)
	if ( julian <= nyear[leap][k]){
		*day   = julian - nyear[leap][k-1];
		*mm    = k;
		break; 
	}
}

