#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <omp.h>

#define blocksize 16

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  int prod=m_a*n_a;
  float *A_r = (float*)calloc(prod,sizeof(float));
  float *B_c = (float*)calloc(prod,sizeof(float));
  if (!A_r || !B_c){
	printf("Memory Allocation failure.");
	exit(1);
  }
  register int i,j,offset, shift_a, shift_b, shift_c;

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
  for (j=0; j<m_a; j++){ //B index
	for (i=0; i<m_a; i++){ //A index
	  shift_a=j*n_a;
	  shift_b=i*n_a;
	  shift_c=j*m_a;
	  for (offset=0; offset<n_a; offset++){
		C[i+shift_c]+=A_r[i+shift_a]*B_c[j+shift_b];
	  }
	}
  }

  //Free Memory
  free(A_r);
  free(B_c);
}
