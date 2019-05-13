#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>
#define BLOCKSIZE 64
#define GRIDSIZE 64

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);//convolutional layer
static Layer l_s1 = Layer(4*4, 1, 6*6*6); //pooling
static Layer l_f = Layer(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

//load minist data and get its total number
static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}

	loaddata();
	learn();
	test();

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();

	clock_t start, end;
	start = clock();
	/*input data sent*/
	int datasize[1] ;
	datasize[0] = 28;
	l_input.setOutput((float *)input,(int *)datasize);
        // convolutional layer, calculate the output, bias and give them to the activation function
	fp_preact_c1<<<BLOCKSIZE, GRIDSIZE>>>((float*) l_input.output, (float*) l_c1.preact, (float*) l_c1.weight, l_c1.preactsize,28,5,6);
	fp_bias_c1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_c1.preact, l_c1.bias,l_c1.preactsize, 6);
	apply_step_function<<<BLOCKSIZE, GRIDSIZE>>>(l_c1.preact, l_c1.output, l_c1.O);
        
	// pooling layer, calculate the output
	fp_preact_s1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_c1.output, (float*)l_s1.preact, (float*)l_s1.weight,l_s1.preactsize,l_c1.preactsize,l_c1.N,4);
	fp_bias_s1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_s1.preact, l_s1.bias, l_s1.preactsize,l_c1.N);
	apply_step_function<<<BLOCKSIZE, GRIDSIZE>>>(l_s1.preact, l_s1.output, l_s1.O);

	
	// dense layer, calculate the output, bias and give them to the activation function, then output (10 nodes)
	fp_preact_f<<<BLOCKSIZE, GRIDSIZE>>>((float* )l_s1.output, l_f.preact, (float*)l_f.weight, l_s1.preactsize, l_c1.N,l_f.O);
	fp_bias_f<<<BLOCKSIZE, GRIDSIZE>>>(l_f.preact, l_f.bias,l_f.O);
	apply_step_function<<<BLOCKSIZE, GRIDSIZE>>>(l_f.preact, l_f.output, l_f.O);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();
        //there is no front information for output layer, we can just calculate the delta w and delta b
	bp_weight_f<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_f.d_weight, l_f.d_preact, (float*)l_s1.output, l_f.O, l_c1.N,l_s1.preactsize);
	bp_bias_f<<<BLOCKSIZE, GRIDSIZE>>>(l_f.bias, l_f.d_preact,l_f.O);
        
	// do what is mentioned in report
	bp_output_s1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_s1.d_output, (float*)l_f.weight, l_f.d_preact, l_f.O,l_c1.N,l_s1.preactsize);
	bp_preact_s1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_s1.d_preact, (float*)l_s1.d_output, (float*)l_s1.preact, l_c1.N,l_s1.preactsize);
	bp_weight_s1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_s1.d_weight, (float*)l_s1.d_preact, (float*)l_c1.output, l_s1.N,4,l_c1.N,l_s1.preactsize,l_c1.preactsize);
	bp_bias_s1<<<BLOCKSIZE, GRIDSIZE>>>(l_s1.bias, (float*)l_s1.d_preact, l_c1.N, l_s1.preactsize);
        // do what is mentioned in report
	bp_output_c1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_c1.d_output, (float*)l_s1.weight, (float*)l_s1.d_preact, l_s1.N,4,l_c1.N,l_s1.preactsize,l_c1.preactsize);
	bp_preact_c1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_c1.d_preact, (float*)l_c1.d_output, (float*)l_c1.preact,l_c1.N,l_c1.preactsize);
	bp_weight_c1<<<BLOCKSIZE, GRIDSIZE>>>((float*)l_c1.d_weight, (float*)l_c1.d_preact, (float*)l_input.output,l_c1.N,5,l_c1.preactsize,l_input.preactsize);
	bp_bias_c1<<<BLOCKSIZE, GRIDSIZE>>>(l_c1.bias, (float*)l_c1.d_preact, l_c1.N, l_c1.preactsize);

        //apply grade according to the delta w
	apply_grad<<<BLOCKSIZE, GRIDSIZE>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad<<<BLOCKSIZE, GRIDSIZE>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad<<<BLOCKSIZE, GRIDSIZE>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn()
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float err;
	int iter = 5;

	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;
            
			time_taken += forward_pass(train_set[i].data); //forward pass

			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10); //calculate error
			cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err); //calculate the norm2
			err += tmp_err;

			time_taken += back_pass(); //back propagation
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}

	fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}
