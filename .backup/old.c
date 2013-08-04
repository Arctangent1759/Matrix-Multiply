#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <omp.h>

#define blocksize 16

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  for (int col_index_A=0; col_index_A < n_a; col_index_A++){ //Which Column of A currently being multiplied. Also the vert offset of B.
	for (int col_index_B=0; col_index_B < m_a; col_index_B++){ //Horiz index of B currently be multiplied
	  //Multiply the element at B down the column of A.
	  __m128 b_vec = _mm_load_ps1(B+col_index_A*m_a+col_index_B);
	  //Iterate down A, multiplying each element by b_val and incrementing into C.
      #pragma omp for
	  for (int a_offset=0; a_offset<m_a; a_offset+=4){//Vert offset of A
		__m128 c_vec = _mm_loadu_ps(C+col_index_B*m_a+a_offset);
		__m128 a_vec = _mm_loadu_ps(A+col_index_A*m_a+a_offset);
		_mm_storeu_ps(C+col_index_B*m_a+a_offset,_mm_add_ps(c_vec,_mm_mul_ps(a_vec,b_vec)));
	  }
	}
  }
}

















/*
void printMatrix(int* mat,int m,int n,int rowMajor){
  printf("-------------\n");
  for (int i=0; i < m; i++){ //which row
	printf("[");
	for (int j=0; j < n; j++){ //which column
	  if (rowMajor){
		printf(" %d ",mat[i*n+j]);
	  }else{
		printf(" %d ",mat[j*m+i]);
	  }
	}
	printf("]\n");
  }
  printf("-------------\n");
}
*/
