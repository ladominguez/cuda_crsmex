#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char FILENAME[]="config.conf";

extern "C" {
#include "crsmex.h"
}

int main(int argc, char **argv)
{
        struct config_filter configstruct;
        /* Cast port as int */
        float low, high, attenuation;
       
        configstruct = get_config(FILENAME);
       
        printf("Low(int)  = %f\n",configstruct.low);
        printf("High(int)  = %f\n",configstruct.high);
        printf("Attenuation(int)  = %f\n",configstruct.attenuation);

        return 0;
}

