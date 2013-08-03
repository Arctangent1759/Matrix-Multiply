#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  register int m_index, n_index, offset;
  for (n_index=0; n_index<n_a; n_index++){
	for (m_index=0; m_index<m_a; m_index++){
	  register __m128 b_vec = _mm_load_ps1(B+n_index*m_a+m_index);
	  #pragma omp for
	  for (offset=0; offset+4<=m_a; offset+=4){
		_mm_storeu_ps(
		  C+n_index*m_a+offset,
		  _mm_add_ps(
			_mm_loadu_ps(C+n_index*m_a+offset),
			_mm_mul_ps(
			  _mm_loadu_ps(A+m_index*m_a+offset),
			  b_vec)
		  )
		);
	  }
	}
  }
}
