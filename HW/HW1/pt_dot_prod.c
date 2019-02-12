/*****************************************************************************
* FILE: dotprod_mutex.c
* DESCRIPTION:
*   This example program illustrates the use of mutex variables 
*   in a threads program. This version was obtained by modifying the
*   serial version of the program (dotprod_serial.c) which performs a 
*   dot product. The main data is made available to all threads through 
*   a globally accessible  structure. Each thread works on a different 
*   part of the data. The main thread waits for all the threads to complete 
*   their computations, and then it prints the resulting sum.
* SOURCE: Vijay Sonnad, IBM
* LAST REVISED: 01/29/09 Blaise Barney
******************************************************************************/
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <sys/times.h>
#include <sys/types.h>

#define NUM_THRDS 8
#define VEC_LEN 20000000
#define MAX_LEN NUM_THRDS*VEC_LEN

/*   
The following structure contains the necessary information  
to allow the function "dotprod" to access its input data and 
place its output into the structure.  This structure is 
unchanged from the sequential version.
*/

typedef struct 
 {
   double *a;
   double *b;
   double sum; 
   int     veclen; 
 } DOTDATA;

/* Define globally accessible variables and a mutex */

DOTDATA dotstr; 
pthread_t callThd[NUM_THRDS];
pthread_mutex_t mutexsum;

double A[MAX_LEN], B[MAX_LEN];

/*
The function dotprod is activated when the thread is created.
As before, all input to this routine is obtained from a structure 
of type DOTDATA and all output from this function is written into
this structure. The benefit of this approach is apparent for the 
multi-threaded program: when a thread is created we pass a single
argument to the activated function - typically this argument
is a thread number. All  the other information required by the 
function is accessed from the globally accessible structure. 
*/

void *dotprod(void *arg)
{

/* Define and use local variables for convenience */

   int i, start, end, len ;
   long offset;
   double mysum, *x, *y;
   offset = (long)arg;
     
   len = dotstr.veclen;
   start = offset*len;
   end   = start + len;
   x = dotstr.a;
   y = dotstr.b;

/*
Perform the dot product and assign result
to the appropriate variable in the structure. 
*/

   mysum = 0;
   for (i=start; i<end ; i++) 
    {
      mysum += (x[i] * y[i]);
    }

/*
Lock a mutex prior to updating the value in the shared
structure, and unlock it upon updating.
*/
   pthread_mutex_lock (&mutexsum);
   dotstr.sum += mysum;
   pthread_mutex_unlock (&mutexsum);

   pthread_exit((void*) 0);
}

/* 
The main program creates threads which do all the work and then 
print out result upon completion. Before creating the threads,
The input data is created. Since all threads update a shared structure, we
need a mutex for mutual exclusion. The main thread needs to wait for
all threads to complete, it waits for each one of the threads. We specify
a thread attribute value that allow the main thread to join with the
threads it creates. Note also that we free up handles  when they are
no longer needed.
*/

int main (int argc, char *argv[])
{
struct timespec start,finish;
int rc, ntime, stime;

long i,j;
/* double *a, *b; */
void *status;
pthread_attr_t attr;

/* Assign storage and initialize values */

j = (int) NULL;
printf("null pointer is %d\n", j);

/*
a = (double*) malloc (NUMTHRDS*VECLEN*sizeof(double));
if (a == NULL) printf("Cannot allocate b, NULL pointer %d \n", (int) a);
b = (double*) malloc (NUMTHRDS*VECLEN*sizeof(double));
if (b == NULL) printf("Cannot allocate b, NULL pointer %d \n", (int) b);
*/
  
for (i=0; i<VEC_LEN*NUM_THRDS; i++) {
  A[i]=1.1/((double) (i+1));
  B[i]=A[i];
  }

dotstr.veclen = VEC_LEN; 
dotstr.a = A; 
dotstr.b = B; 
dotstr.sum=0;

pthread_mutex_init(&mutexsum, NULL);
         
/* Create threads to perform the dotproduct  */
pthread_attr_init(&attr);
pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

clock_gettime(CLOCK_REALTIME,&start);
for(i=0;i<NUM_THRDS;i++) {
  /* Each thread works on a different set of data.
   * The offset is specified by 'i'. The size of
   * the data for each thread is indicated by VECLEN.
   */
   rc = pthread_create(&callThd[i], &attr, dotprod, (void *)i); 
   if (rc){
     printf("ERROR: return code from pthread_create() is %d\n", rc);
     printf("Code %d= %s\n",rc,strerror(rc));
     exit(-1);
   }

}

pthread_attr_destroy(&attr);
/* Wait on the other threads */

for(i=0;i<NUM_THRDS;i++) {
  pthread_join(callThd[i], &status);
  }
/* After joining, print out the results and cleanup */

clock_gettime(CLOCK_REALTIME,&finish);
ntime = finish.tv_nsec - start.tv_nsec;
stime = (int) finish.tv_sec - (int) start.tv_sec;
printf("main(): Created %ld threads. Time %ld, nsec %ld\n", NUM_THRDS, stime, ntime);


printf ("Sum =  %f \n", dotstr.sum);
/*free (a);
free (b); */
pthread_mutex_destroy(&mutexsum);
pthread_exit(NULL);
}   

