#include "crsmex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXBUF 1024
#define DELIM "="

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
