#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <omp.h>

#define blocksize 16

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  omp_set_num_threads(2);
  int prod=m_a*n_a;
  float *A_r = (float*)calloc(prod,sizeof(float));
  float *B_c = (float*)calloc(prod,sizeof(float));
  float *unpacker = (float*)malloc(sizeof(float)*4);
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

  register __m128 a_vec, b_vec, c_vec;
  //Multiply the matrices
  for (j=0; j<m_a; j++){ //B column
	for (i=0; i<m_a; i++){ //A row
	  shift_a=i*n_a;
	  shift_b=j*n_a;
	  shift_c=j*m_a;
	  c_vec = _mm_setzero_ps();
	  for (offset=0; offset<n_a; offset+=4){
		a_vec=_mm_loadu_ps(A_r+offset+shift_a);
		b_vec=_mm_loadu_ps(B_c+offset+shift_b);
		{
		c_vec=_mm_add_ps(c_vec,_mm_mul_ps(a_vec,b_vec));
		}
		//C[i+shift_c]+=A_r[offset+shift_a]*B_c[offset+shift_b];
	  }
	  _mm_storeu_ps(unpacker,c_vec);
	  C[i+shift_c]=unpacker[0]+unpacker[1]+unpacker[2]+unpacker[3];
	}
  }

  //Free Memory
  free(A_r);
  free(B_c);
}
