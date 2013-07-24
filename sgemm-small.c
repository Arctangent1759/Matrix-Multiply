#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {

  register int a_column,a_index=0, b_index=0, max_index=m_a*n_a
  _m128 a_vec,b_vec,c_vec;

  for (; a_column < max_index; a_column+=m_a){
	for (; b_index < max_index; b_index+=1){
	  b_vec = _mm_set_ps1(B[b_index]);
	  for (a_index = 0 ; a_index < m_a; a_index+=4){
		a_vec = _mm_loadu_ps(A+a_column+a_index);
		c_vec = _mm_loadu_ps(A+a_column+a_index);
		_mm_add_ps(c_vec,_mm_mul_ps(a_vec,b_vec));
		_mm_storeu_ps(C+a_column,c_vec);
	  }
	}
  }
}
