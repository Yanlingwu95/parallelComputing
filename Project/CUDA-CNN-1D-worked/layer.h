#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int M, N, O;

	float *output;
	float *preact;
	int *preactsize;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data,int* datasize);
	void clear();
	void bp_clear();
};


// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float* input, float* preact, float* weight, int* preactsize, int img, int kernel, int nodes);
__global__ void fp_bias_c1(float* preact, float* bias, int* preactsize, int nodes);
__global__ void fp_preact_s1(float* input, float* preact, float* weight, int* preactsize, int* res, int nodes, int kernel);
__global__ void fp_bias_s1(float* preact, float* bias, int* res, int nodes);
__global__ void fp_preact_f(float* input, float* preact, float* weight, int* res, int last_nodes, int nodes);
__global__ void fp_bias_f(float* preact, float* bias, int nodes);

// Back propagation kernels
__global__ void bp_weight_f(float* d_weight, float* d_preact, float* p_output,int nodes,int last_nodes,int* res);
__global__ void bp_bias_f(float* bias, float* d_preact,int nodes);
__global__ void bp_output_s1(float* d_output, float* n_weight, float* nd_preact,int before_nodes, int nodes, int* res);
__global__ void bp_preact_s1(float* d_preact, float* d_output, float* preact, int nodes, int* res);
__global__ void bp_weight_s1(float* d_weight, float* d_preact, float* p_output, int poolnodes, int kernel, int nodes, int* res, int* lastres);
__global__ void bp_bias_s1(float* bias, float* d_preact, int nodes, int* res);
__global__ void bp_output_c1(float* d_output, float* n_weight, float* nd_preact,int before_nodes, int before_kernel,int nodes, int* before_res, int* res);
__global__ void bp_preact_c1(float* d_preact, float* d_output, float* preact,int nodes, int* res);
__global__ void bp_weight_c1(float* d_weight, float* d_preact, float* p_output,int nodes, int kernel, int* res, int* lastres);
__global__ void bp_bias_c1(float* bias, float* d_preact, int nodes, int* res);
