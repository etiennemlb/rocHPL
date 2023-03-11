/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include <hip/hip_runtime.h>
#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))

#define BLOCK_SIZE 256

__launch_bounds__(BLOCK_SIZE)
__global__ void dlaswp10N(const int M,
                          const int N,
                          double* __restrict__ A,
                          const int LDA,
                          const int* __restrict__ IPIV) {


  __shared__ int s_piv[1024];

  for (int t=threadIdx.x;t<N;t+=BLOCK_SIZE) {
    s_piv[t] = IPIV[t];
  }

  __syncthreads();

  const int m = threadIdx.x + BLOCK_SIZE * blockIdx.x;

  if(m < M) {
    for(int i = 0; i < N; ++i) {

      int n = i;
      int pn = s_piv[i];

      if (pn!=i) {
        double An  = A[m + n * static_cast<size_t>(LDA)];

        while (pn!=i) {
          //swap
          A[m + n * static_cast<size_t>(LDA)] = A[m + pn * static_cast<size_t>(LDA)];

          n = pn;
          pn = s_piv[pn];
        }

        A[m + n * static_cast<size_t>(LDA)] = An;

        __syncthreads();

        if (threadIdx.x==0) { //thread 0 records pivots
          n = i;
          pn = s_piv[i];

          while (pn!=i) {
            s_piv[n] = n;
            n = pn;
            pn = s_piv[n];
          }
          s_piv[n] = n;
        }

        __syncthreads();
      }
    }
  }
}

void HPL_dlaswp10N(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   const int* IPIV) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp10N performs a sequence  of  local column interchanges on a
   * matrix A.  One column interchange is initiated  for columns 0 through
   * N-1 of A.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         __arg0__
   *
   * N       (local input)                 const int
   *         On entry,  M  specifies  the number of rows of the array A. M
   *         must be at least zero.
   *
   * A       (local input/output)          double *
   *         On entry, N specifies the number of columns of the array A. N
   *         must be at least zero.
   *
   * LDA     (local input)                 const int
   *         On entry, A  points to an  array of  dimension (LDA,N).  This
   *         array contains the columns onto which the interchanges should
   *         be applied. On exit, A contains the permuted matrix.
   *
   * IPIV    (local input)                 const int *
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * ---------------------------------------------------------------------
   */

  if((M <= 0) || (N <= 0)) return;

  assertm(N <= 1024, "NB too large in HPL_dlaswp10N");

  hipStream_t stream;
  rocblas_get_stream(handle, &stream);

  dim3 grid_size((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dlaswp10N<<<grid_size, dim3(BLOCK_SIZE), 0, stream>>>(M, N, A, LDA, IPIV);
}
