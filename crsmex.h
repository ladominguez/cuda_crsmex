#ifndef __CRSMEX__
#define __CRSMEX__

struct config_filter
{
   double low;
   double high;
   double attenuation;
   double transition_band;
   int    npoles;
   int    passes;
};

struct config_filter get_config(const char *filename);

void load_sac_in_host_memory(float *data_host, char *infilename, int *nsac, int *npts, bool filter, config_filter configstruct);
char *strstrip(char *s);                           // Deletes trailing characters when reading filenames. Similar to .rtrip() in Python.
void julian2mmdd(int year, int julian, int *mm, int *day);
#endif
