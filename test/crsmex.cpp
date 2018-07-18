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
				configstruct.low = atoi(cfline);
                        } else if (i == 1){
				configstruct.high = atoi(cfline);
                        } else if (i == 2){
				configstruct.attenuation = atoi(cfline);
                        }

                        i++;
                } // End while
                fclose(file);
        } // End if file



        return configstruct;

}
