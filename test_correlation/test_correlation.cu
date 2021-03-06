//nvcc -arch=sm_30 -lcufft fft_batched.cu
#include <stdio.h>
#include <ctype.h>
#include <cuda.h>
#include <cufft.h>
#include <math.h>
#include <cuComplex.h>
extern "C"{
#include <sacio.h>
#include <sac.h>
}


#define DATASIZE   1024
#define BATCH      2
#define MAX_ARRAY  1024
#define NSAC       2
#define MAX_PATH   100
#define N_FILENAME 2

#define GRID_DIMENSION  2
#define BLOCK_DIMENSION 8



/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
inline void gpuPlanCheck(int error_plan)
{
  if (error_plan != CUFFT_SUCCESS ){
	fprintf(stderr,"GPU: Couldn't create an fft plan.\n");
	exit(error_plan);
  }
}
__global__ void power_func( long int nelem, int npts,  cufftComplex  *fft, cufftComplex *power);
__global__ void correlation_coeff(cufftComplex *correlation, cufftComplex  *fft,  int batch_id);

char *strstrip(char *s);


/********/
/* MAIN */
/********/
int main ()
{
    
    int       grid_size  = GRID_DIMENSION;
    int       block_size = BLOCK_DIMENSION;
    float     *data;
    float     beg, del;
    char      *line;
    FILE      *fid;
    size_t    len=0;
    int       count=0;
    int       nlen, nerr, max = MAX_ARRAY;
    char      infilename[]={"filenames.dat"} ;
    char      kname[ N_FILENAME ] ;
    float     yarray[MAX_ARRAY];
    

    dim3 DimGrid(grid_size, grid_size, grid_size);
    dim3 DimBlock(block_size, block_size, block_size);

    // --- Device side output data allocation
    cufftComplex     *deviceOutputData; 
    cufftComplex     *correlation;
    cufftReal        *correlation_time;
    cufftComplex     *power;
    cufftReal        *power_time;
 

    // --- Host side output data allocation
    int size_fft = DATASIZE / 2 + 1;
    cufftComplex    *hostOutputData     = (   cufftComplex*)malloc((size_fft) * BATCH * sizeof(cufftComplex));
    cufftComplex    *hostOutputPower    = (   cufftComplex*)malloc((size_fft) * BATCH * sizeof(cufftComplex)); 
    cufftReal       *hostOutputPowerT   = (      cufftReal*)malloc((DATASIZE) * BATCH * sizeof(cufftReal)); 
    cufftReal       *correlationHost    = (      cufftReal*)malloc((DATASIZE) * BATCH * sizeof(cufftReal));
    fprintf(stderr, "size_fft = %d\n",size_fft);

    // --- Host side input data allocation and initialization
    cufftReal       *hostInputData = (cufftReal*)malloc(DATASIZE*BATCH*sizeof(cufftReal));
    data = (float *)malloc(NSAC * MAX_ARRAY * sizeof(float));

    // Lee nombres de archio
    fid = fopen(infilename,"r");
    if (fid == NULL){
        fprintf(stderr,"Couldn't open file %s\n",infilename);
        exit(-1);
    }

    // Lee archivos sac
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
      
     	memcpy(&data[count*MAX_ARRAY], yarray, nlen*sizeof(float));
     	count++;
     }


    cufftReal *deviceInputData; 
    gpuErrchk(cudaMalloc((void**)&deviceInputData, DATASIZE * BATCH * sizeof(cufftReal)));
    cudaMemcpy(deviceInputData, data, DATASIZE * BATCH * sizeof(cufftReal), cudaMemcpyHostToDevice);

    // Device allocation data
    gpuErrchk(cudaMalloc((void**)&deviceOutputData, size_fft * BATCH * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void**)&power,            size_fft * BATCH * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void**)&correlation,      size_fft * BATCH * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void**)&power_time,       DATASIZE * BATCH * sizeof(cufftReal)));
    gpuErrchk(cudaMalloc((void**)&correlation_time, DATASIZE * BATCH * sizeof(cufftReal)));

    // --- Batched 1D FFTs
    cufftHandle handle_forward;
    cufftHandle handle_inverse;
    cufftHandle handle_complex;
    int batch = BATCH;                          // --- Number of batched executions

    cufftPlan1d( &handle_forward, DATASIZE, CUFFT_R2C, batch);
    cufftPlan1d( &handle_inverse, DATASIZE, CUFFT_C2R, batch);
    cufftPlan1d( &handle_complex, DATASIZE, CUFFT_C2C, batch);

    // FFT
    gpuPlanCheck( cufftExecR2C(handle_forward,  deviceInputData, deviceOutputData) );

    power_func<<< DimGrid, DimBlock >>> (size_fft * BATCH, size_fft,  deviceOutputData, power );

    gpuPlanCheck( cufftExecC2R(handle_inverse,  power, power_time) );

    correlation_coeff <<< BATCH, size_fft >>> (correlation, deviceOutputData, 0); 

    gpuPlanCheck( cufftExecC2R(handle_inverse,  correlation, correlation_time ) );
    //gpuPlanCheck( cufftExecC2R( handle_inverse, correlation, correlation));

    // --- Device->Host copy of the results
    gpuErrchk(cudaMemcpy(hostOutputData,     deviceOutputData, size_fft * BATCH * sizeof(cufftComplex),    cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hostOutputPower,    power,            size_fft * BATCH * sizeof(cufftComplex),    cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hostOutputPowerT,   power_time,       DATASIZE * BATCH * sizeof(cufftReal),       cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(correlationHost,    correlation_time, DATASIZE * BATCH * sizeof(cufftReal),       cudaMemcpyDeviceToHost));

   
      
    float max_corr[BATCH];
    for (int i=0; i< BATCH; i++){
	for (int j = 0; j < DATASIZE; j++){
		//fprintf(stdout,"%f\n", correlationHost[i*DATASIZE + j]);
		if (correlationHost[i*DATASIZE + j] >= max_corr[i]){
			max_corr[i] = correlationHost[i*DATASIZE + j];
		}
	}
    }



    for (int i=0; i<BATCH; i++){
        fprintf(stderr, "max_corr[%d]         = %f\n", i, max_corr[i]/DATASIZE);
	fprintf(stderr, "hostOutputPowerT[%d] = %f\n", i, hostOutputPowerT[DATASIZE*i]/(2));
        fprintf(stderr, "CC[%d]               = %f\n", i, 2*max_corr[i]/(DATASIZE*sqrt(hostOutputPowerT[0]*hostOutputPowerT[DATASIZE*i])));
    }


    cufftDestroy(handle_forward);
    cufftDestroy(handle_inverse);
    cufftDestroy(handle_complex);
    gpuErrchk(cudaFree(deviceOutputData));
    gpuErrchk(cudaFree(deviceInputData));
    gpuErrchk(cudaFree(power));
    gpuErrchk(cudaFree(power_time));
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return EXIT_SUCCESS;

}

__global__ void correlation_coeff(cufftComplex *correlation, cufftComplex  *fft, int batch_id)
{
int  bx =  blockIdx.x;
int thx = threadIdx.x;

int NumThread = blockDim.x*blockDim.y*blockDim.z;
//int idThread  = thx;
//int BlockId   = bx;

int uniqueid  = thx + NumThread*bx;

correlation[uniqueid].x =      fft[uniqueid].x*fft[thx + batch_id*NumThread].x  + fft[uniqueid].y*fft[thx + batch_id*NumThread].y;
correlation[uniqueid].y =      fft[uniqueid].x*fft[thx + batch_id*NumThread].y  - fft[uniqueid].y*fft[thx + batch_id*NumThread].x;

//printf("%d %f %f\n",uniqueid, correlation[uniqueid].x, correlation[uniqueid].y );

}

__global__ void power_func(long int nelem, int npts, cufftComplex *fft, cufftComplex *power)
{
int bx = blockIdx.x;
int by = blockIdx.y;
int bz = blockIdx.z;

int thx = threadIdx.x;
int thy = threadIdx.y;
int thz = threadIdx.z;

int NumThread = blockDim.x*blockDim.y*blockDim.z;
int idThread  = (thx + thy*blockDim.x) + thz*(blockDim.x*blockDim.y);
int BlockId   =    (bx + by*gridDim.x) + bz*(gridDim.x*gridDim.y);

int uniqueid  = idThread + NumThread*BlockId;

if (uniqueid < nelem){
        power[uniqueid].x = (fft[uniqueid].x*fft[uniqueid].x + fft[uniqueid].y*fft[uniqueid].y)/npts;
	power[uniqueid].y = 0;
        //printf("Unique ID = %d - conj = %f\n",  uniqueid,  conj[uniqueid].y*-1);
}
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
