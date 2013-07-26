#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  register int row_index_A, col_index_B=0, iterator_index_AB, max_size=m_a*n_a;
  register __m128 a_vec, b_vec, c_vec;

  for (; col_index_B < m_a; col_index_B+=1){ //make sure index_B is the index of the beginning of each column of B
	for (row_index_A=0; row_index_A < m_a; row_index_A+=4){ //Iterate accross each 4xm_a horizontal bock of A
	  c_vec = _mm_setzero_ps(); //Load C_Vec into memory
	  for (iterator_index_AB=0; iterator_index_AB < n_a; iterator_index_AB+=3){ //Reduce each block into c_vec

		a_vec = _mm_loadu_ps(A+iterator_index_AB*m_a+row_index_A);
		b_vec = _mm_set_ps1(B[iterator_index_AB*m_a+col_index_B]);
		c_vec=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec);

		a_vec = _mm_loadu_ps(A+(iterator_index_AB+1)*m_a+row_index_A);
		b_vec = _mm_set_ps1(B[(iterator_index_AB+1)*m_a+col_index_B]);
		c_vec=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec);

		a_vec = _mm_loadu_ps(A+(iterator_index_AB+2)*m_a+row_index_A);
		b_vec = _mm_set_ps1(B[(iterator_index_AB+2)*m_a+col_index_B]);
		c_vec=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec);
	  }
	  _mm_storeu_ps(C+col_index_B*m_a+row_index_A,c_vec);
	}
  }


  /*
	 register int col_index=0, row_index, col_offset;
	 register __m128 a_vec, b_vec, c_vec;

	 for (; col_index < n_a; col_index++){
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
	 */
}
