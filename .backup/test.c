#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <omp.h>

#define blocksize 16

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  float *A_r = (float*)malloc(m_a*n_a);
  float *B_c = (float*)malloc(m_a*n_a);
  if (!A_r || !B_c){
	printf("Memory Allocation failure.");
	exit(1);
  }
  register int i,j,offset,shift_a,shift_b;

  //Convert A to shift_a major
  for (i = 0; i < m_a; i++){
	for (j = 0; j < n_a; j++){
	  A_r[j+n_a*i]=A[i+m_a*j];
	}
  }
  //Convert B to shift_b major
  for (i = 0; i < n_a; i++){
	for (j = 0; j < m_a; j++){
	  B_c[i+n_a*j]=B[j+m_a*i];
	}
  }

  //Multiply the matrices
  for (i=0; i<m_a; i++){
	for (j=0; j<m_a; j++){
	  shift_a=i*n_a;
	  shift_b=j*n_a;
	  for (offset=0; offset<n_a; offset++){
	  }
	}
  }

  //Free Memory
  free(A_r);
  free(B_c);
}
