// To Compile it
// gcc -m32 -o test.out test.c sac.a
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sacio.h>
#include <unistd.h>

/* Define the maximum length of the data array */
#define MAX 100000
#define NSAC 100
#define N_FILENAME 100
#define PATH_MAX 100

char *strstrip(char *s); // Deletes trailing characters when reading filenames. Similar to .rtrip() in Python.
void usage();            // Show usage

int
main(int argc, char **argv)
{
  /* Define variables to be used in the call to rsac1() */
  float   yarray[MAX], beg, del;
  int     nlen, nerr, max = MAX, opt = 0;
  float   *data[NSAC];
  char    kname[ N_FILENAME ] ;
  char    infilename[ N_FILENAME ] ;
  FILE    *fid;
  size_t  len=0;
  ssize_t read;
  int     count=0;

  char  *line;
  size_t line_size = 100;

  if( argc == 1 ) {
	usage();
	exit(-1);
  } 
  while((opt = getopt(argc, argv, "f:")) != -1){
	switch(opt){
	      case 'f':
		strncpy(infilename, optarg, PATH_MAX);
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
  while ((read = getline(&line, &len, fid)) != -1)
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
	memcpy(data[count],yarray,nlen*sizeof(float));
	count++;
  }
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

