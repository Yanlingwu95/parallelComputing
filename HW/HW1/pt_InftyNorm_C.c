/********************************************************************

pt_InftyNorm_C.c calculates the Infinity Norm of a matrix.

        Threads work on columns of the matrix.

        pthread_create()
        pthread_join()
        pthread_mutex_t
        pthread_mutex_lock()
        pthread_mutex_unlock()

Input:  row - number of rows, read from Norm_data.txt file
        col - number of columns, read from Norm_data.txt file
        NumThreads - read from the comman line

Output: Norm - the infinity norm of the matrix.

********************************************************************/

#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>

pthread_mutex_t *mutex_Row;
pthread_mutex_t mutex_col;
int Norm = 0, row, col, **Mat, *Row, CurCol=0;

void *doMyWork(int myId){

     int cRow, cCol;
     while (1){
       pthread_mutex_lock(&mutex_col); {
         cCol=CurCol;
         if (CurCol >= col){
           pthread_mutex_unlock(&mutex_col);
           return(0);
         }
         CurCol++;
       }
       pthread_mutex_unlock(&mutex_col);

       for(cRow=0; cRow<row; cRow++){
         pthread_mutex_lock(&mutex_Row[cRow]); 
           Row[cRow] += abs(Mat[cRow][cCol]);
         pthread_mutex_unlock(&mutex_Row[cRow]);
       }
     }
}
    

void main(int argc, char*argv[]){

     pthread_t *threads;
     int counter, cRow, cCol, NumThreads;

/* open a file for reading */
     FILE *fp = fopen("Norm_data.txt", "r");
        if (fp == NULL) {
         printf("Error opening file!\n");
         exit(1);
        }
     if (argc != 2 ){
        printf("\n Missing Arguments: Number of Threads.\n");
        return;
     }

/* get the number of rows, columns and threads */

     fscanf(fp, "%d", &row); fscanf(fp, "%d", &col);
     NumThreads = atoi(argv[1]);
     printf("\n %d rows, %d columns, %d threads ", row, col, NumThreads);

/* allocate memory for threads, locks and data matrix */

     threads = (pthread_t*)malloc(sizeof(pthread_t)*NumThreads);
     Mat = (int**)malloc(sizeof(int)*row);
     Row = (int*)malloc(sizeof(int)*row);
     mutex_Row = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t)*row);

     for (counter=0; counter<row; counter++){
       Mat[counter]=(int*)malloc(sizeof(int)*col);
       Row[counter]=0;
     }

/* read the matrix */
     for (cRow=0; cRow<row; cRow++)
       for (cCol=0; cCol<col; cCol++)
         fscanf(fp, "%d", &Mat[cRow][cCol]);

/* fork threads */
     for (counter=0; counter<NumThreads; counter++)
       pthread_create(&threads[counter], NULL, (void*)doMyWork, (void*) &counter);

/* join threads */
     for (counter=0; counter<NumThreads; counter++)
       pthread_join(threads[counter], NULL);

/* compute the maximum norm over all rows */
     for (cRow=0; cRow<row; cRow++)
       if (Row[cRow]>Norm)
         Norm=Row[cRow];

     printf("\n Infinity Norm: %d.\n", Norm);
  pthread_exit(0);     
}
