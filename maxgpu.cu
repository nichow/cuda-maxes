#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 512
#define NUM_BLOCKS 512
/* used for debugging */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if(code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s, %s, %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

/* function declarations */
unsigned int getmax(unsigned int *, unsigned int);
__global__ void getmaxcu(unsigned int *, unsigned int *, unsigned int);

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }    

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for( i = 0; i < size; i++)
       numbers[i] = rand()  % size;

    /* define the number of blocks, a host array for the maxes, and device arrays */
    unsigned int *maxes = (unsigned int *)malloc(NUM_BLOCKS * sizeof(unsigned int));
    unsigned int *dev_num, *dev_maxes; 

    /*allocate space on the device */
    gpuErrchk(cudaMalloc((void**)&dev_num, size * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)&dev_maxes, NUM_BLOCKS * sizeof(unsigned int)));

    /*do our copies and execute the kernel */
    gpuErrchk(cudaMemcpy(dev_num, numbers, size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    getmaxcu<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_num, dev_maxes, size);
    gpuErrchk(cudaPeekAtLastError()); //debug info
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(maxes, dev_maxes, NUM_BLOCKS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* free space on the device */
    cudaFree(dev_num);
    cudaFree(dev_maxes);

    /* the final max calculation is done on the host
     * at this point we have few enough values that using the gpu is not necessary */
    unsigned int overall_max = 0;
    for(i = 0; i < NUM_BLOCKS; ++i) {
      if(overall_max < maxes[i])
        overall_max = maxes[i];
    }
    
    printf(" The maximum number in the array is: %u\n", overall_max);

    free(numbers);
    free(maxes);
    exit(0);
}


/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmax(unsigned int num[], unsigned int size)
{
  unsigned int i;
  unsigned int max = num[0];

  for(i = 1; i < size; i++)
	if(num[i] > max)
	   max = num[i];

  return( max );

}

__global__ void getmaxcu(unsigned int * g_idata, unsigned int * g_odata, unsigned int size) {
  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  unsigned int i = bid * blockDim.x + tid;

  /* find the maximum value of each block using a reduction */
  if(i < size) {
    unsigned int stride;
    for(stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
      if(tid < stride) {
        if(g_idata[tid] < g_idata[tid + stride])
          g_idata[tid] = g_idata[tid + stride];
      }
    }
  }
  __syncthreads();

  /*write the result of each block to the output array */
  if(tid == 0)
    g_odata[bid] = g_idata[0];
}