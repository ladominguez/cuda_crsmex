#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <sac.h>
#include <sacio.h>

#define MAX 1000000


int main(int argc, char *argv[]) {

    double low, high, attenuation, transition_bandwidth;;
    int nlen, nerr, max;

    float beg, delta;
    double delta_d;
    char *kname;
    int order;

    int passes;
    float yarray[MAX];
    float xarray[1];

    max = MAX;

    kname = strdup("20110106064638.IG.CAIG.HHZ.sac");
    printf("%sX\n",kname);
    printf("long = %ld\n",strlen( kname ));
    //rsac1(kname, yarray, &nlen, &beg, &delta, &max, &nerr, SAC_STRING_LENGTH);
    rsac1(kname, yarray, &nlen, &beg, &delta, &max, &nerr, strlen( kname ));
    printf("nerr = %d\n", nerr);
    delta_d = delta;
    if (nerr != 0) {
      fprintf(stderr, "Error reading in file: %s\n", kname);
      exit(-1);
    }

    low    = 0.10;
    high   = 1.00;
    passes = 2;
    order  = 4;
    transition_bandwidth = 0.0;
    attenuation = 0.0;

    xapiir(yarray, nlen, SAC_BUTTERWORTH, 
           transition_bandwidth, attenuation, 
           order, 
           SAC_BANDPASS, 
           low, high, 
           delta_d, passes);

    /*     Do more processing ....  */
    xarray[0] = 0;
    kname = strdup("filterc_out.sac");    
    wsac0(kname, xarray, yarray, &nerr, SAC_STRING_LENGTH);
    if (nerr != 0) {
      fprintf(stderr, "Error writing out file: %s\n", kname);
      exit(-1);
    }
    return 0;
}

