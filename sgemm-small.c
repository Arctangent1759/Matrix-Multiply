#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  if(m_a%36==0 && n_a%36==0){
	register int row_index_A, col_index_B=0, iterator_index_AB, max_size=m_a*n_a;
	register __m128 a_vec, b_vec, c_vec1,c_vec2,c_vec3,c_vec4,c_vec5,c_vec6,c_vec7,c_vec8,c_vec9;

	for (; col_index_B < m_a; col_index_B+=9){ //make sure index_B is the index of the beginning of each column of B
	  for (row_index_A=0; row_index_A < m_a; row_index_A+=4){ //Iterate accross each 4xm_a horizontal bock of A
		c_vec1 = _mm_setzero_ps(); //Load c_vec1 into memory
		c_vec2 = _mm_setzero_ps(); //Load c_vec1 into memory
		c_vec3 = _mm_setzero_ps(); //Load c_vec1 into memory
		c_vec4 = _mm_setzero_ps(); //Load c_vec1 into memory
		c_vec5 = _mm_setzero_ps(); //Load c_vec1 into memory
		c_vec6 = _mm_setzero_ps(); //Load c_vec1 into memory
		c_vec7 = _mm_setzero_ps(); //Load c_vec1 into memory
		c_vec8 = _mm_setzero_ps(); //Load c_vec1 into memory
		c_vec9 = _mm_setzero_ps(); //Load c_vec1 into memory
		for (iterator_index_AB=0; iterator_index_AB < n_a; iterator_index_AB+=6){ //Reduce each block into c_vec1
		  a_vec = _mm_loadu_ps(A+iterator_index_AB*m_a+row_index_A);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B);
		  c_vec1=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec1);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B+1);
		  c_vec2=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec2);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B+2);
		  c_vec3=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec3);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B+3);
		  c_vec4=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec4);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B+4);
		  c_vec5=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec5);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B+5);
		  c_vec6=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec6);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B+6);
		  c_vec7=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec7);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B+7);
		  c_vec8=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec8);

		  b_vec = _mm_load_ps1(B+iterator_index_AB*m_a+col_index_B+8);
		  c_vec9=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec9);

		  //pointless comment

		  a_vec = _mm_loadu_ps(A+(iterator_index_AB+1)*m_a+row_index_A);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B);
		  c_vec1=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec1);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B+1);
		  c_vec2=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec2);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B+2);
		  c_vec3=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec3);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B+3);
		  c_vec4=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec4);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B+4);
		  c_vec5=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec5);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B+5);
		  c_vec6=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec6);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B+6);
		  c_vec7=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec7);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B+7);
		  c_vec8=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec8);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+1)*m_a+col_index_B+8);
		  c_vec9=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec9);

		  //pointless comment

		  a_vec = _mm_loadu_ps(A+(iterator_index_AB+2)*m_a+row_index_A);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B);
		  c_vec1=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec1);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B+1);
		  c_vec2=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec2);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B+2);
		  c_vec3=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec3);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B+3);
		  c_vec4=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec4);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B+4);
		  c_vec5=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec5);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B+5);
		  c_vec6=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec6);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B+6);
		  c_vec7=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec7);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B+7);
		  c_vec8=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec8);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+2)*m_a+col_index_B+8);
		  c_vec9=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec9);

		  //pointless comment

		  a_vec = _mm_loadu_ps(A+(iterator_index_AB+3)*m_a+row_index_A);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B);
		  c_vec1=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec1);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B+1);
		  c_vec2=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec2);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B+2);
		  c_vec3=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec3);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B+3);
		  c_vec4=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec4);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B+4);
		  c_vec5=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec5);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B+5);
		  c_vec6=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec6);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B+6);
		  c_vec7=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec7);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B+7);
		  c_vec8=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec8);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+3)*m_a+col_index_B+8);
		  c_vec9=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec9);

		  //pointless comment

		  a_vec = _mm_loadu_ps(A+(iterator_index_AB+4)*m_a+row_index_A);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B);
		  c_vec1=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec1);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B+1);
		  c_vec2=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec2);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B+2);
		  c_vec3=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec3);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B+3);
		  c_vec4=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec4);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B+4);
		  c_vec5=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec5);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B+5);
		  c_vec6=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec6);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B+6);
		  c_vec7=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec7);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B+7);
		  c_vec8=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec8);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+4)*m_a+col_index_B+8);
		  c_vec9=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec9);

		  //pointless comment

		  a_vec = _mm_loadu_ps(A+(iterator_index_AB+5)*m_a+row_index_A);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B);
		  c_vec1=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec1);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B+1);
		  c_vec2=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec2);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B+2);
		  c_vec3=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec3);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B+3);
		  c_vec4=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec4);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B+4);
		  c_vec5=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec5);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B+5);
		  c_vec6=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec6);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B+6);
		  c_vec7=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec7);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B+7);
		  c_vec8=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec8);

		  b_vec = _mm_load_ps1(B+(iterator_index_AB+5)*m_a+col_index_B+8);
		  c_vec9=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec9);



		}
		_mm_storeu_ps(C+col_index_B*m_a+row_index_A,c_vec1);
		_mm_storeu_ps(C+(col_index_B+1)*m_a+row_index_A,c_vec2);
		_mm_storeu_ps(C+(col_index_B+2)*m_a+row_index_A,c_vec3);
		_mm_storeu_ps(C+(col_index_B+3)*m_a+row_index_A,c_vec4);
		_mm_storeu_ps(C+(col_index_B+4)*m_a+row_index_A,c_vec5);
		_mm_storeu_ps(C+(col_index_B+5)*m_a+row_index_A,c_vec6);
		_mm_storeu_ps(C+(col_index_B+6)*m_a+row_index_A,c_vec7);
		_mm_storeu_ps(C+(col_index_B+7)*m_a+row_index_A,c_vec8);
		_mm_storeu_ps(C+(col_index_B+8)*m_a+row_index_A,c_vec9);
	  }
	}
  }else{
	register int col_index=0, row_index, col_offset, edge_offset;
	const register int edge_max = m_a%4;
	register __m128 a_vec, b_vec, c_vec1;
	for (; col_index < n_a; col_index++){
	  for (row_index = 0; row_index < n_a; row_index++){
		b_vec = _mm_set_ps1(B[col_index*n_a+row_index]);
		for (col_offset = 0; col_offset < m_a; col_offset+=4){
		  a_vec = _mm_loadu_ps(A+col_index*m_a+col_offset);
		  c_vec1 = _mm_loadu_ps(C+row_index*m_a+col_offset);
		  c_vec1=_mm_add_ps(_mm_mul_ps(a_vec,b_vec),c_vec1);
		  _mm_storeu_ps(C+row_index*m_a+col_offset,c_vec1);
		}
		for (edge_offset=edge_max; edge_offset--;){//Edge case
		  C[row_index*(m_a+1)-edge_offset]+=A[col_index*(m_a+1)-edge_offset]*B[col_index*n_a+row_index];
		}
	  }
	}
  }
}
