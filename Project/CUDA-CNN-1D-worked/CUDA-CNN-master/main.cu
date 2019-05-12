#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(11*11, 3, 18*18*3);//convolutional layer
static Layer l_c2 = Layer(7*7, 3, 12*12*3); //
static Layer l_s1 = Layer(4*4, 1, 3*3*3); //pooling
static Layer l_f = Layer(3*3*3, 10, 10);
// static Layer l_d1 = Layer(10,10,10); //my neural network
// static Layer l_d2 = Layer(3,10,10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

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
	l_c2.clear();
	l_s1.clear();
	l_f.clear();
	// l_d1.clear();

	clock_t start, end;
	start = clock();
	/*input data sent*/
	int datasize[1] ;
	datasize[0] = 28;
	l_input.setOutput((float *)input,(int *)datasize);

	fp_preact_c1<<<64, 64>>>((float*) l_input.output, (float*) l_c1.preact, (float*) l_c1.weight, l_c1.preactsize,l_input.preactsize,11,3);
	fp_bias_c1<<<64, 64>>>((float*)l_c1.preact, l_c1.bias,l_c1.preactsize, 3);
	apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

	fp_preact_c1<<<64, 64>>>((float*) l_c1.output, (float*) l_c2.preact, (float*) l_c2.weight, l_c2.preactsize,l_c1.preactsize,7,l_c2.N);
	fp_bias_c1<<<64, 64>>>((float*)l_c2.preact, l_c2.bias,l_c2.preactsize, l_c2.N);
	apply_step_function<<<64, 64>>>(l_c2.preact, l_c2.output, l_c2.O);

	fp_preact_s1<<<64, 64>>>((float*)l_c2.output, (float*)l_s1.preact, (float*)l_s1.weight,l_s1.preactsize,l_c2.preactsize,l_c2.N,4);
	fp_bias_s1<<<64, 64>>>((float*)l_s1.preact, l_s1.bias, l_s1.preactsize,l_c2.N);
	apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);

	fp_preact_f<<<64, 64>>>((float* )l_s1.output, l_f.preact, (float*)l_f.weight, l_s1.preactsize, l_c2.N,l_f.O,l_f.preactsize);
	fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias,l_f.O);
	apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);

	// fp_preact_dense<<<64, 64>>>((float*) l_f.output, (float*) l_d1.preact, (float*) l_d1.weight, l_d1.preactsize,l_f.preactsize,10,10);
	// fp_bias_dense<<<64, 64>>>((float*)l_d1.preact, l_d1.bias,l_d1.preactsize, 10);
	// apply_step_function<<<64, 64>>>(l_d1.preact, l_d1.output, l_d1.O);


	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();


	// bp_weight_dense<<<64, 64>>>((float*)l_d1.d_weight, (float*)l_d1.d_preact, (float*)l_f.output,l_d1.N,10,l_d1.preactsize,l_f.preactsize);
	// bp_bias_dense<<<64, 64>>>(l_d1.bias, (float*)l_d1.d_preact, l_d1.N, l_d1.preactsize);
	//
	// bp_output_dense<<<64, 64>>>((float*)l_f.d_output, (float*)l_d1.weight, (float*)l_d1.d_preact, l_d1.N,10,l_f.N,l_d1.preactsize,l_f.preactsize);
	// bp_preact_dense<<<64, 64>>>((float*)l_f.d_preact, (float*)l_f.d_output, (float*)l_f.preact,l_f.N,l_f.preactsize);
	bp_weight_f<<<64, 64>>>((float*)l_f.d_weight, l_f.d_preact, (float*)l_s1.output, l_f.O, l_c2.N,l_s1.preactsize);
	bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact,l_f.O);

	bp_output_s1<<<64, 64>>>((float*)l_s1.d_output, (float*)l_f.weight, l_f.d_preact, l_f.O,l_c2.N,l_s1.preactsize);
	bp_preact_s1<<<64, 64>>>((float*)l_s1.d_preact, (float*)l_s1.d_output, (float*)l_s1.preact, l_c2.N,l_s1.preactsize);
	bp_weight_s1<<<64, 64>>>((float*)l_s1.d_weight, (float*)l_s1.d_preact, (float*)l_c2.output, l_s1.N,4,l_c2.N,l_s1.preactsize,l_c2.preactsize);
	bp_bias_s1<<<64, 64>>>(l_s1.bias, (float*)l_s1.d_preact, l_c2.N, l_s1.preactsize);

	bp_output_c1<<<64, 64>>>((float*)l_c2.d_output, (float*)l_s1.weight, (float*)l_s1.d_preact, l_s1.N,4,l_c2.N,l_s1.preactsize,l_c2.preactsize);
	bp_preact_c1<<<64, 64>>>((float*)l_c2.d_preact, (float*)l_c2.d_output, (float*)l_c2.preact,l_c2.N,l_c2.preactsize);
	bp_weight_c1<<<64, 64>>>((float*)l_c2.d_weight, (float*)l_c2.d_preact, (float*)l_input.output,l_c2.N,7,l_c2.preactsize,l_input.preactsize);
	bp_bias_c1<<<64, 64>>>(l_c2.bias, (float*)l_c2.d_preact, l_c2.N, l_c2.preactsize);

	bp_output_c1<<<64, 64>>>((float*)l_c1.d_output, (float*)l_c2.weight, (float*)l_c2.d_preact, l_c2.N,7,l_c1.N,l_c2.preactsize,l_c1.preactsize);
	bp_preact_c1<<<64, 64>>>((float*)l_c1.d_preact, (float*)l_c1.d_output, (float*)l_c1.preact,l_c1.N,l_c1.preactsize);
	bp_weight_c1<<<64, 64>>>((float*)l_c1.d_weight, (float*)l_c1.d_preact, (float*)l_input.output,l_c1.N,11,l_c1.preactsize,l_input.preactsize);
	bp_bias_c1<<<64, 64>>>(l_c1.bias, (float*)l_c1.d_preact, l_c1.N, l_c1.preactsize);

	// apply_grad<<<64, 64>>>(l_d1.weight, l_d1.d_weight, l_d1.M * l_d1.N);
	apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad<<<64, 64>>>(l_c2.weight, l_c2.d_weight, l_c2.M * l_c2.N);
	apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);


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
	int iter = 20;

	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);
			// l_d1.bp_clear();
			l_f.bp_clear();
			l_s1.bp_clear();
			l_c2.bp_clear();
			l_c1.bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err); //calculate the norm2
			err += tmp_err;

			time_taken += back_pass();
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
