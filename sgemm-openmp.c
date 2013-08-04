#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <omp.h>

#define blocksize 16
#define padsize 16
#define threadnum 16

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  omp_set_num_threads(threadnum);
  int m_a_padded = m_a+padsize-(m_a%padsize);
  int n_a_padded = n_a+padsize-(n_a%padsize);
  int prod=m_a_padded*n_a_padded;
  int prodC=m_a_padded*m_a_padded;
  float *A_r = (float*)calloc(prod,sizeof(float));
  float *B_c = (float*)calloc(prod,sizeof(float));
  float *C_p = (float*)calloc(prodC,sizeof(float));
  float *unpacker = (float*)malloc(threadnum*sizeof(float)*4);
  if (!(A_r && B_c && C_p)){
	printf("Memory Allocation failure.");
	exit(1);
  }
  register int i,j,k,l,offset, shift_a, shift_b, shift_c;

  //Convert A to row major and pad
#pragma omp parallel for
  for (i = 0; i < m_a; i++){
	for (j = 0; j < n_a; j++){
	  A_r[j+n_a_padded*i]=A[i+m_a*j];
	}
  }
  //Convert B to column major and pad
#pragma omp parallel for
  for (i = 0; i < n_a; i++){
	for (j = 0; j < m_a; j++){
	  B_c[i+n_a_padded*j]=B[j+m_a*i];
	}
  }

  register __m128 a_vec, b_vec, c_vec;
  float res;
  //Multiply the matrices
  for (j=0; j<m_a_padded; j++){ //B column
	for (i=0; i<m_a_padded; i+=blocksize){ //A row
		for(k=i; k<i+blocksize; k++){
		  shift_a=k*n_a_padded;
		  shift_b=j*n_a_padded;
		  shift_c=j*m_a_padded;
		  c_vec = _mm_setzero_ps();
		  for (offset=0; offset<n_a_padded; offset+=4){
			a_vec=_mm_loadu_ps(A_r+offset+shift_a);
			b_vec=_mm_loadu_ps(B_c+offset+shift_b);
			c_vec=_mm_add_ps(c_vec,_mm_mul_ps(a_vec,b_vec));
			//C[k+shift_c]+=A_r[offset+shift_a]*B_c[offset+shift_b];
		  }
		  _mm_storeu_ps(unpacker+omp_get_thread_num()*4,c_vec);
		  res=unpacker[omp_get_thread_num()*4]+unpacker[omp_get_thread_num()*4+1]+unpacker[omp_get_thread_num()*4+2]+unpacker[omp_get_thread_num()*4+3];
		  C_p[k+shift_c]=res;
		}
	}
  }

  //Trim C_p and store into C
#pragma omp parallel for
  for (i = 0; i < m_a; i++){
	for (j = 0; j < m_a; j++){
	  C[i+m_a*j]=C_p[i+m_a_padded*j];
	}
  }


  //Free Memory
  free(A_r);
  free(B_c);
  free(C_p);
  free(unpacker);
}
