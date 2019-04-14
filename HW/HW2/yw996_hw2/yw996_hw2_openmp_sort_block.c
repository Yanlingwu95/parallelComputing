/*Compile : Use the command "gcc -fopenmp yw996_hw2_openmp_sort_block.c -o sort_block -O3"
  Execute : use "./sort_block m n numTh",  
		 m : the number of rows in matrix
		 n : the number of columns in matrix
		 numTh : the number of threads
  */

/******************************************
Yanling Wu
yw996
HW2. sort_block parallel and sequential
*******************************************/
#define _POSIX_C_SOURCE 199309L
#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CHUNK_SIZE 1
int *A; 
int m, n, NUM_THRDS;

/*Parallel: Swap the rows */
void swap(int u, int a, int b) {
	int j = 0;
	/*exchange the row*/
	#pragma omp parallel for schedule(static, CHUNK_SIZE)
		for (j = 0; j < n; j++) {
			int temp = *(A + n * a + j); 
			*(A + n * a + j) = *(A + n * u + j);
			*(A + n * u + j) = temp;
		}
	/*exchange the column*/
	int i = 0;
	#pragma omp parallel for schedule(static, CHUNK_SIZE)
		for (i = 0; i < m; i++) {
			int temp = *(A + n * i + u);
			*(A + n * i + u) = *(A + n * i + b);
			*(A + n * i + b) = temp;
		}
	
}

/*Function to find the maximul value of submatrix and swap the rows*/
void final_A(int dim) {
	int i = 0, j = 0, k = 0;
	int maxVal = -1;
	int maxRow = 0, maxCol = 0;
	for (j = 0; j < dim; j++) {
		maxRow = 0,maxCol = 0, maxVal = -1;
		#pragma parallel omp for schedule(static, CHUNK_SIZE) shared(maxRow, maxVal, maxCol) collapse(2)
			for(i = j; i < m; i++) {
				for(k = j; k < n; k++)
					if (maxVal < *(A + n * i + k)) {
						#pragma omp critical
						{
							if (maxVal < *(A + n * i + k)) {
								maxVal = *(A + n * i + k);
								maxRow = i;
								maxCol = k;
							}
						}
					}
			}

		swap(j, maxRow, maxCol);
	}
}

int main(int argc, char * argv[]) {
	if (argc < 3 ){
        printf("\n Missing Arguments: Number of Threads.\n");
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


	A = (int *)malloc(m * n * sizeof(int));
	srand(6);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*(A + n * i + j) = rand() % 100;
		}
	}

	clock_gettime(CLOCK_REALTIME, &start);
	/*Set number of threads*/
	omp_set_num_threads(NUM_THRDS);

	final_A(dim);

	clock_gettime(CLOCK_REALTIME, &finish);
	ntime = finish.tv_nsec - start.tv_nsec;
	stime = (int)finish.tv_sec - (int) start.tv_sec;
	ttime = ntime / 1.0e9 + stime;
	printf("main(): Created %d threads. Time %f  sec\n", NUM_THRDS, ttime);
}