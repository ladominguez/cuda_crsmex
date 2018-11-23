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
__global__ void conjugate(  long int nelem,            cufftComplex *conj                     );
__global__ void power_func( long int nelem, int npts,  cufftComplex  *fft, cufftComplex *power);
__global__ void correlation_coeff(cufftComplex *power);

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

    // --- Host side output data allocation
    int size_fft = DATASIZE / 2 + 1;
    cufftComplex    *hostOutputData     = (   cufftComplex*)malloc((size_fft) * BATCH * sizeof(cufftComplex));
    cufftComplex    *hostOutputDataConj = (   cufftComplex*)malloc((size_fft) * BATCH * sizeof(cufftComplex));
    cufftComplex    *hostOutputPower    = (   cufftComplex*)malloc((size_fft) * BATCH * sizeof(cufftComplex)); 
    cufftReal       *hostOutputPowerT   = (      cufftReal*)malloc((DATASIZE) * BATCH * sizeof(cufftReal)); 
    fprintf(stderr, "size_fft = %d\n",size_fft);

    // --- Device side output data allocation
    cufftComplex     *deviceOutputData; 
    cufftComplex     *fft_conj; 
    cufftComplex     *power;
    cufftReal        *power_time;
    gpuErrchk(cudaMalloc((void**)&deviceOutputData, size_fft * BATCH * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void**)&fft_conj,         size_fft * BATCH * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void**)&power,            size_fft * BATCH * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void**)&power_time,       DATASIZE * BATCH * sizeof(cufftReal)));

    // --- Batched 1D FFTs
    cufftHandle handle_both;
    cufftHandle handle_forward;
    cufftHandle handle_inverse;
    int rank    = 1;                            // --- 1D FFTs
    int n[]     = { DATASIZE };                 // --- Size of the Fourier transform
//    int n_inv[] = { size_fft };                 // --- Size of the Inverse Fourier transform
    int istride = 1, ostride = 1;               // --- Distance between two successive input/output elements
    int idist   = DATASIZE, odist = (DATASIZE / 2 + 1); // --- Distance between batches
    int inembed[] = { 0 };                      // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                      // --- Output size with pitch (ignored for 1D transforms)
    int batch = BATCH;                          // --- Number of batched executions

    cufftPlan1d( &handle_forward, DATASIZE, CUFFT_R2C, batch);
/*
    cufftPlanMany(&handle_forward, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist, CUFFT_R2C, batch);
*/
    cufftPlan1d( &handle_inverse, DATASIZE, CUFFT_C2R, batch);
/* 
    cufftPlanMany(&handle_inverse, rank, n,
                  onembed, ostride, DATASIZE,
                  inembed, istride, size_fft, CUFFT_C2R, batch);
  */
    // FFT
    //cufftExecC2C(handle_both,  deviceInputData, deviceOutputData, CUFFT_FORWARD); 
    cufftExecR2C(handle_forward,  deviceInputData, deviceOutputData);


//    gpuErrchk(cudaMemcpy(fft_conj,         deviceOutputData, size_fft * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToDevice));
//    conjugate <<< DimGrid, DimBlock >>> (size_fft * BATCH, fft_conj );i
    power_func<<< DimGrid, DimBlock >>> (size_fft * BATCH, size_fft,  deviceOutputData, power );
    correlation_coeff <<< 4, 128 >>> (power); 

    int error_cufft=cufftExecC2R(handle_inverse,  power, power_time);
    fprintf(stderr, "error_cufft = %d\n",error_cufft);

    // --- Device->Host copy of the results
    gpuErrchk(cudaMemcpy(hostOutputData,     deviceOutputData, size_fft * BATCH * sizeof(cufftComplex),    cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hostOutputDataConj, fft_conj,         size_fft * BATCH * sizeof(cufftComplex),    cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hostOutputPower,    power,            size_fft * BATCH * sizeof(cufftComplex),    cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hostOutputPowerT,   power_time,       DATASIZE * BATCH * sizeof(cufftReal),       cudaMemcpyDeviceToHost));
    for (int i=0; i<BATCH; i++){
	printf("hostOutputPowerT[%d] = %f\n",i, hostOutputPowerT[DATASIZE*i]/(2));
    }
/*
    for (int i=0; i<BATCH-1; i++)
	for (int j=0; j<size_fft; j++) 
		printf("%d %f %f %f\n",size_fft*i + j,hostOutputData[size_fft + j].x, hostOutputData[size_fft + j].y, sqrt(hostOutputData[size_fft + j].x*hostOutputData[size_fft + j].x + hostOutputData[size_fft + j].y*hostOutputData[size_fft + j].y));
*/


/*
    for (int i=0; i<BATCH; i++)
        for (int j=0; j<DATASIZE ; j++)
	      printf("%i %f\n",i*DATASIZE + j, hostOutputPowerT[i*DATASIZE + j] );
*/


	    // printf("%i %i %f %f\n",j,i,hostOutputPower[i*size_fft + j].x,hostOutputPower[i*size_fft + j].y );
	    //  printf("%i %i %f %f\n",j,i,hostOutputPower[i*size_fft + j].x,hostOutputPower[i*size_fft + j].y );
	    //printf("%i %i %f\n",i,j,data[i*(DATASIZE)  + j]);		
	    //printf("%i %i %f %f %f\n", i, j, hostOutputDataConj[i*(DATASIZE / 2 + 1) + j].x, hostOutputDataConj[i*(DATASIZE / 2 + 1) + j].y, hostOutputPower[i*(DATASIZE / 2 + 1) + j]);

    cufftDestroy(handle_forward);
    cufftDestroy(handle_inverse);
    gpuErrchk(cudaFree(deviceOutputData));
    gpuErrchk(cudaFree(deviceInputData));
    gpuErrchk(cudaFree(fft_conj));
    gpuErrchk(cudaFree(power));
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return EXIT_SUCCESS;

}

__global__ void correlation_coeff(cufftComplex *power)
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

//printf("%d %d %d\n", uniqueid, idThread, BlockId);

}
__global__ void conjugate(long int nelem, cufftComplex *conj)
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
	conj[uniqueid].y = conj[uniqueid].y*-1;
 	//printf("Unique ID = %d - conj = %f\n",  uniqueid,  conj[uniqueid].y*-1);
}

//__syncthreads();
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
//printf("nelem = %d\n",nelem);
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
