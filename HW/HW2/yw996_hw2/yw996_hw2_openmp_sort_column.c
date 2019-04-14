/*Compile : Use the command "gcc -fopenmp yw996_hw2_openmp_sort_column.c -o sort_column -O3"
  Execute : use "./sort_column m n numTh",  
		 m : the number of rows in matrix
		 n : the number of columns in matrix
		 numTh : the number of threads
	This code includes the parallel part and the sequential parts' codes. 
  */

/******************************************
Yanling Wu
yw996
HW2. sort_column parallel and sequential
*******************************************/

#define _POSIX_C_SOURCE 199309L
#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

int *A; 
int m, n, NUM_THRDS;

/*Parallel: Swap the rows */
void swap(int a, int b) {
	int j = 0;
	#pragma omp parallel for schedule(static)
		for (j = 0; j < n; j++) {
			int temp = *(A + n * a + j); 
			*(A + n * a + j) = *(A + n * b + j);
			*(A + n * b + j) = temp;
		}
	
}

/*Parallel: Function to find the maximul value of one column and swap the rows*/
void final_A(int dim) {
	int i = 0, j = 0;
	int maxVal = -1;
	int maxRow = 0;
	for (j = 0; j < dim; j++) {
		maxRow = 0, maxVal = -1;
		#pragma parallel omp for schedule(static) shared(maxRow, maxVal) 
			for(i = j; i < m; i++) {
				if (maxVal < *(A + n * i + j)) {
					#pragma omp critical
					{
						if (maxVal < *(A + n * i + j)) {
							maxVal = *(A + n * i + j);
							maxRow = i;
						}
					}
					
				}
			}

		swap(j, maxRow);
	}
}

/*Sequencial : Swap the rows*/
void swap_seq(int a, int b) {
	int j = 0;
		for (j = 0; j < n; j++) {
			int temp = *(A + n * a + j); 
			*(A + n * a + j) = *(A + n * b + j);
			*(A + n * b + j) = temp;
		}
}

/*Sequencial : Function to find the maximul value of one column and swap the rows*/
void final_A_seq(int dim) {
	int i = 0, j = 0;
	int maxVal = -1;
	int maxRow = 0;
	for (j = 0; j < dim; j++) {
		maxRow = 0, maxVal = -1;
			for(i = j; i < m; i++) {
				if (maxVal < *(A + n * i + j)) {
					maxVal = *(A + n * i + j);
					maxRow = i;
				}
			}
		swap_seq(j, maxRow);
	}
}

int main(int argc, char * argv[]) {
	/*Judge whether the number of arguments is enough*/
	if (argc < 3 ){
        printf("\n Missing Arguments!\n");
        return 0;
     }
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	NUM_THRDS = atoi(argv[3]);

	struct timespec start,finish;
	int rs, ntime, stime;
	double ttime;
	
	int dim = MIN(m, n);
	int i, j;

	/*Generate the Matrix A*/
	A = (int *)malloc(m * n * sizeof(int));
	srand(6);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*(A + n * i + j) = rand() % 100;
		}
	}

	/*Parallel Parts*/
	clock_gettime(CLOCK_REALTIME, &start);
	/*Set number of threads*/
	omp_set_num_threads(NUM_THRDS);
	final_A(dim);

	clock_gettime(CLOCK_REALTIME, &finish);

	ntime = finish.tv_nsec - start.tv_nsec;
	stime = (int)finish.tv_sec - (int) start.tv_sec;
	ttime = ntime / 1.0e9 + stime;
	printf("Parallel Parts: Created %d threads. Time %f  sec\n", NUM_THRDS, ttime);

	/**************************************************************************/
	/*Sequencial Part*/
	srand(6);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*(A + n * i + j) = rand() % 100;
		}
	}

	clock_gettime(CLOCK_REALTIME, &start);	

	final_A_seq(dim);

	clock_gettime(CLOCK_REALTIME, &finish);
	ntime = finish.tv_nsec - start.tv_nsec;
	stime = (int)finish.tv_sec - (int) start.tv_sec;
	ttime = ntime / 1.0e9 + stime;
	printf("Sequencial Parts: Created %d threads. Time %f  sec\n", NUM_THRDS, ttime);
}