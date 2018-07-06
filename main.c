// To Compile it
// gcc -m32 -o test.out test.c sac.a
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <sacio.h>

/* Define the maximum length of the data array */
#define MAX 100000
#define NSAC 100
#define N_FILENAME 100

char *strstrip(char *s);

int
main(int argc, char **argv)
{
  /* Define variables to be used in the call to rsac1() */
  float   yarray[MAX], beg, del;
  int     nlen, nerr, max = MAX;
  float   *data;
  char    kname[ N_FILENAME ] ;
  FILE    *fid;
  size_t  len=0;
  ssize_t read;

  char  *line;
  size_t line_size = 100;

  line = (char  *)malloc(line_size    * sizeof(char));
  data = (float *)malloc( MAX  * NSAC * sizeof(float));  

  fid = fopen("filenames.dat","r"); 
  while ((read = getline(&line, &len, fid)) != -1)
  {
        printf("Retrieve line of lenght %zu:\n", read);
        printf("%s", line);
	line = strstrip(line);
        strcpy ( kname ,line ) ;
        printf("%s", kname);
	fprintf(stderr,"strlen(kname) = %lu\n",strlen( kname ));
        rsac1( kname, yarray, &nlen, &beg, &del, &max, &nerr, strlen( kname ) ) ;
        printf("numel %d\n",nlen);
        if ( nerr != 0 ) {
                fprintf(stderr, "Error reading in SAC file: %s\n", kname);
                exit ( nerr ) ;
        }
	else {
    		fprintf(stderr,"Reading SUCCESS: %s\n\n",kname);
	}
  }
free(data);
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
