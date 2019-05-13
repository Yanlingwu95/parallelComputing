/*  This file is used to implement the layer class and all kernel functions */

#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;
	//host
	float h_bias[N];
	float h_weight[N][M];
	//device
	output = NULL;
	preact = NULL;
	preactsize = NULL;
	bias   = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);
	cudaMalloc(&preactsize, sizeof(int) * 1);
	cudaMalloc(&bias, sizeof(float) * N);
	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);
	cudaFree(preactsize);
	cudaFree(bias);
	cudaFree(weight);
	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data, int* datasize)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
	cudaMemcpy(preactsize,datasize,sizeof(int)*1,cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
	cudaMemset(preactsize, 0x00, sizeof(float) * 1);
}

// Reset GPU memory between iterations after bq
void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

//Sigmoid function that can be only called by device
__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));//Sigmoid
}

__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(input[idx]);
	}
}

//Y is label of data set
__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

//update weights
__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		*(output+idx) += dt * *(grad+idx);
	}
}

//calculate convlutional output  //img = 28, kernel = 5, nodes = 6
__global__ void fp_preact_c1(float* input, float* preact, float* weight, int* preactsize, int img, int kernel, int nodes)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	// const int N = 5*5*6*24*24;
	const int res = img - kernel + 1;
	*preactsize = res;
	const int N = kernel*kernel*nodes*res*res;
	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % kernel);
		const int i2 = ((idx /= kernel	) % kernel);
		const int i3 = ((idx /= kernel	) % nodes);
		const int i4 = ((idx /= nodes	) % res);
		const int i5 = ((idx /= res	) % res);

		atomicAdd(&(*(preact+i3*res*res+i4*res+i5)), *(weight+i3*kernel*kernel+i1*kernel+i2) * *(input+(i4 + i1)*img+(i5 + i2)));

	}
}

//calculate convlutional output's bias
__global__ void fp_bias_c1(float* preact, float* bias, int* preactsize, int nodes)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int res = *preactsize;
	const int N = nodes*res*res;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) % res);
		const int i3 = ((idx /= res	) % res);

		*(preact+i1*res*res+i2*res+i3) += *(bias+i1);
	}
}

//full-stride_conv_s1 
__global__ void fp_preact_s1(float* input, float* preact, float* weight,int* preactsize, int* res, int nodes, int kernel)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	int rk = *res/kernel;
	*preactsize = rk;
	const int N = kernel*kernel*nodes*rk*rk;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % kernel);
		const int i2 = ((idx /= kernel	) % kernel);
		const int i3 = ((idx /= kernel	) % nodes);
		const int i4 = ((idx /= nodes	) % rk);
		const int i5 = ((idx /= rk	) % rk);

		atomicAdd(&(*(preact+i3*rk*rk+i4*rk+i5)), *(weight+i1*kernel+i2) * *(input+i3* *res* *res+(i4 * kernel + i1)* *res+(i5 * kernel + i2)));
	}
}

// Calculate the bias of the pooling layer
__global__ void fp_bias_s1(float* preact, float* bias, int* res, int nodes)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = nodes*(*res)*(*res);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) % (*res));
		const int i3 = ((idx /= (*res)	) % (*res));

		*(preact+i1*(*res)*(*res)+i2*(*res)+i3) += *bias;
	}
}

//Calculate the preact value of the output layer
__global__ void fp_preact_f(float* input, float* preact, float* weight, int* res, int last_nodes, int nodes)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	// const int N = 10*6*6*6;
	const int N = nodes*last_nodes*(*res)*(*res);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) % last_nodes);
		const int i3 = ((idx /= last_nodes	) % *res);
		const int i4 = ((idx /= *res	) % *res);

		atomicAdd(&(*(preact+i1)), *(weight+i1*last_nodes*(*res)*(*res)+i2*(*res)*(*res)+i3*(*res)+i4) * *(input+i2*(*res)*(*res)+i3*(*res)+i4));
	}
}

//Calculate the bias value of the output layer
__global__ void fp_bias_f(float* preact, float* bias, int nodes)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = nodes;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		*(preact+idx) += *(bias+idx);
	}
}

//Compute the gradient of weights of the output layer during back propogation
__global__ void bp_weight_f(float* d_weight, float* d_preact, float* p_output, int nodes,int last_nodes, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = nodes*last_nodes* (*res)*(*res);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) % last_nodes);
		const int i3 = ((idx /= last_nodes	) % *res);
		const int i4 = ((idx /= *res	) % *res);

		*(d_weight+i1*last_nodes*(*res)*(*res)+i2*(*res)*(*res)+i3*(*res)+i4) = *(d_preact+i1) * *(p_output+i2*(*res)*(*res)+i3*(*res)+i4);
	}
}

// Compute the bias of the output layer during back propogation
__global__ void bp_bias_f(float* bias, float* d_preact, int nodes)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = nodes;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		*(bias+idx) += dt *  *(d_preact+idx);
	}
}

// Compute the gradient of output of pooling layer
__global__ void bp_output_s1(float* d_output, float* n_weight, float* nd_preact,int before_nodes,int nodes,int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = before_nodes*nodes*(*res)*(*res);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % before_nodes);
		const int i2 = ((idx /= before_nodes	) % nodes);
		const int i3 = ((idx /= nodes	) % (*res));
		const int i4 = ((idx /= (*res)	) % (*res));

		atomicAdd(&(*(d_output+i2*(*res)*(*res)+i3*(*res)+i4)), *(n_weight+i1*nodes*(*res)*(*res)+i2*(*res)*(*res)+i3*(*res)+i4) * *(nd_preact+i1));
	}
}

// Compute the preact value of the back propogation of pooling layer
__global__ void bp_preact_s1(float* d_preact, float* d_output, float* preact, int nodes, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	//const int N = 6*6*6;
	const int N = nodes*(*res)*(*res);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) % (*res));
		const int i3 = ((idx /= (*res)	) % (*res));

		const float o = step_function(*(preact+i1*(*res)*(*res)+i2*(*res)+i3));

		*(d_preact+i1*(*res)*(*res)+i2*(*res)+i3) = *(d_output+i1*(*res)*(*res)+i2*(*res)+i3) * o * (1 - o);
	}
}

//Compute the gradient of the weights of the pooling layer
__global__ void bp_weight_s1(float* d_weight, float* d_preact, float* p_output, int poolnodes, int kernel, int nodes, int* res, int* lastres)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = poolnodes*kernel*kernel*nodes*(*res)*(*res);
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % poolnodes);
		const int i2 = ((idx /= poolnodes) % kernel);
		const int i3 = ((idx /= kernel	) % kernel);
		const int i4 = ((idx /= kernel	) % nodes);
		const int i5 = ((idx /= nodes	) % (*res));
		const int i6 = ((idx /= (*res)	) % (*res));

		atomicAdd(&(*(d_weight+i1*kernel*kernel+i2*kernel+i3)), (*(d_preact+i4*(*res)*(*res)+i5*(*res)+i6) * *(p_output+(i4*(*lastres)*(*lastres))+(i5 * kernel + i2)*(*lastres)+(i6 * kernel + i3)))/d);
	}
}

//Compute the gradient bias of the pooling layers
__global__ void bp_bias_s1(float* bias, float* d_preact, int nodes, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = nodes*(*res)*(*res);
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) % *(res));
		const int i3 = ((idx /= *(res)	) % (*res));

		atomicAdd(&(*bias), (dt *  *(d_preact+i1*(*res)*(*res)+i2*(*res)+i3)) / d);
	}
}

//Compute the gradient of the output of conv layer
__global__ void bp_output_c1(float* d_output, float* n_weight, float* nd_preact,int before_nodes, int before_kernel,int nodes, int* before_res, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	//const int N = 1*4*4*6*6*6;
	const int N = before_nodes*before_kernel*before_kernel*(*before_res)*(*before_res)*nodes;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % before_nodes);
		const int i2 = ((idx /= before_nodes	) % before_kernel);
		const int i3 = ((idx /= before_kernel	) % before_kernel);
		const int i4 = ((idx /= before_kernel	) % nodes);
		const int i5 = ((idx /= nodes	) % (*before_res));
		const int i6 = ((idx /= (*before_res)	) % (*before_res));

		atomicAdd(&(*(d_output+i4*(*res)*(*res)+(i5 * before_kernel + i2)*(*res)+(i6 * before_kernel + i3))), *(n_weight+i1*before_kernel*before_kernel+i2*before_kernel+i3) * *(nd_preact+i4*(*before_res)*(*before_res)+i5*(*before_res)+i6));
	}
}

// Compute the gradient of the preact of convo layer
__global__ void bp_preact_c1(float* d_preact, float* d_output, float* preact, int nodes, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	//const int N = 6*24*24;
	const int N = nodes*(*res)*(*res);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) % (*res));
		const int i3 = ((idx /= (*res)	) % (*res));

		const float o = step_function(*(preact+i1*(*res)*(*res)+i2*(*res)+i3));

		*(d_preact+i1*(*res)*(*res)+i2*(*res)+i3) = *(d_output+i1*(*res)*(*res)+i2*(*res)+i3) * o * (1 - o);
	}
}

//Calculate of the gradient of the weights of the convoluational layers
__global__ void bp_weight_c1(float* d_weight, float* d_preact, float* p_output, int nodes, int kernel, int* res, int* lastres)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = nodes*kernel*kernel*(*res)*(*res);
	const float d = 24.0f * 24.0f;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes); //把64*64个threads分成6个，负责6个neurons 计算
		const int i2 = ((idx /= nodes	) % kernel);
		const int i3 = ((idx /= kernel	) % kernel);
		const int i4 = ((idx /= kernel	) % (*res));
		const int i5 = ((idx /= (*res)	) % (*res));

		atomicAdd(&(*(d_weight+i1*kernel*kernel+i2*kernel+i3)), *(d_preact+i1*(*res)*(*res)+i4*(*res)+i5) * *(p_output+(i4 + i2)*(*lastres)+(i5 + i3)) / d);
	}
}

//Compute the gradient bias of the convolutional layer
__global__ void bp_bias_c1(float* bias, float* d_preact,int nodes, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = nodes*(*res)*(*res);
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) %(*res));
		const int i3 = ((idx /= (*res)	) % (*res));

		atomicAdd(&(*(bias+i1)), dt * (*(d_preact+i1*(*res)*(*res)+i2*(*res)+i3)) / d);
	}
}

//calculate dense layer output  
__global__ void fp_preact_dense(float* input, float* preact, float* weight, int* preactsize, int* img, int kernel, int nodes)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	// const int N = 5*5*6*24*24;
	const int res = 1;
	*preactsize = res;
	const int N = kernel*1*nodes*1*1;
	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % kernel);
		const int i2 = ((idx /= kernel	) % 1);
		const int i3 = ((idx /= 1	) % nodes);
		const int i4 = ((idx /= nodes	) % res);
		const int i5 = ((idx /= res	) % 1);

		atomicAdd(&(*(preact+i3*res*res+i4*res+i5)), *(weight+i3*kernel*kernel+i1*kernel+i2) * *(input+(i4 + i1)* *img+(i5 + i2)));
	}
}

//Calculate the bias of the dense layer of forward pass
__global__ void fp_bias_dense(float* preact, float* bias, int* preactsize, int nodes)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int res = *preactsize;
	const int N = nodes*res*res;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);

		*(preact+i1) += *(bias+i1);
	}
}

//Calculate the gradient of output layers
__global__ void bp_output_dense(float* d_output, float* n_weight, float* nd_preact,int before_nodes, int before_kernel,int nodes, int* before_res, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	//const int N = 1*4*4*6*6*6;
	const int N = before_nodes*before_kernel*1*(*before_res)*(*before_res)*nodes;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % before_nodes);
		const int i2 = ((idx /= before_nodes	) % before_kernel);
		const int i3 = ((idx /= before_kernel	) % 1);
		const int i4 = ((idx /= 1	) % nodes);
		const int i5 = ((idx /= nodes	) % (*before_res));
		const int i6 = ((idx /= (*before_res)	) % (*before_res));

		atomicAdd(&(*(d_output+i4*(*res)*(*res)+(i5 * before_kernel + i2)*(*res)+(i6 * 1 + i3))), *(n_weight+i1*before_kernel*1+i2*1+i3) * *(nd_preact+i4*(*before_res)*(*before_res)+i5*(*before_res)+i6));
	}
}

// calculate the back propogation of the dense layer
__global__ void bp_preact_dense(float* d_preact, float* d_output, float* preact, int nodes, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	//const int N = 6*24*24;
	const int N = nodes*(*res)*(*res);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) % (*res));
		const int i3 = ((idx /= (*res)	) % (*res));

		const float o = step_function(*(preact+i1*(*res)*(*res)+i2*(*res)+i3));

		*(d_preact+i1*(*res)*(*res)+i2*(*res)+i3) = *(d_output+i1*(*res)*(*res)+i2*(*res)+i3) * o * (1 - o);
	}
}

// calculate the back propogation weights of the dense layer
__global__ void bp_weight_dense(float* d_weight, float* d_preact, float* p_output, int nodes, int kernel, int* res, int* lastres)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

//	const int N = 6*5*5*24*24;
	const int N = nodes*kernel*1*(*res)*(*res);
	const float d = 24.0f * 24.0f;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes); 
		const int i2 = ((idx /= nodes	) % kernel);
		const int i3 = ((idx /= kernel	) % 1);
		const int i4 = ((idx /= 1	) % (*res));
		const int i5 = ((idx /= (*res)	) % (*res));

		atomicAdd(&(*(d_weight+i1*kernel*1+i2*1+i3)), *(d_preact+i1*(*res)*(*res)+i4*(*res)+i5) * *(p_output+(i4 + i2)*(*lastres)+(i5 + i3)) / d);
	}
}

// calculate the back propogation bias of the dense layer
__global__ void bp_bias_dense(float* bias, float* d_preact,int nodes, int* res)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = nodes*(*res)*(*res);
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % nodes);
		const int i2 = ((idx /= nodes	) %(*res));
		const int i3 = ((idx /= (*res)	) % (*res));

		atomicAdd(&(*(bias+i1)), dt * (*(d_preact+i1*(*res)*(*res)+i2*(*res)+i3)) / d);
	}
}
