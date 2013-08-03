#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  register int col_index=0, row_index, col_offset;
  __m128 a_vec, b_vec, c_vec;
  for (col_index; col_index < n_a; col_index++){
	for (row_index = 0; row_index < n_a; row_index++){
	  b_vec = _mm_set_ps1(B[col_index*n_a+row_index]);
	  for (col_offset = 0; col_offset < m_a; col_offset+=4){ //TODO: Edge case when m_a not a multiple of 4.
		a_vec = _mm_loadu_ps(A+col_index*m_a+col_offset);
		c_vec = _mm_loadu_ps(C+row_index*m_a+col_offset);
		c_vec=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec);
		_mm_storeu_ps(C+row_index*m_a+col_offset,c_vec);
	  }
	}
  }
}
