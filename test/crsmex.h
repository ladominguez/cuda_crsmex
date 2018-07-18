#ifndef __CRSMEX__
#define __CRSMEX__

struct config_filter
{
   double low;
   double high;
   double attenuation;
};

struct config_filter get_config(const char *filename);

#endif
