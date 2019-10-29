//implementing gradient descent with mean squared error in order to learn a 4x4 transformation matrix
//A bit overkill for a linear problem, and nothing special needs to be done to avoid local minima.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const unsigned BLOCK_SIZE = 64;
const unsigned BATCH_SIZE = 1000000;
const unsigned EPOCHS = 1000;
const float STEP_SIZE = 0.00001;  //not really a step size, more of a constant
const unsigned VALIDATION_SIZE = 1000000;

struct mat4x4
{
	float a11, a12, a13, a14;
	float a21, a22, a23, a24;
	float a31, a32, a33, a34;
	const float a41 = 0, a42 = 0, a43 = 0, a44 = 1;
};
typedef struct mat4x4;
//shouldn't i invectigate SoA?
typedef struct vec4
{
	float x;
	float y;
	float z;
	float w;
};

//multiply a 4x4 and a vec4
__global__ void mult(const mat4x4 mat, unsigned n, const vec4* vec_in, vec4* vec_out)
{
	//should i investigate forcing mat shared here? naw, i'm good

	//our block * block size + our thread (so like flat matrix indexing)
	unsigned grid_ndx = blockIdx.x * blockDim.x + threadIdx.x;
	//there are two options for how to handle data that is larger than the number of threads we have
	//option 1: if grid_ndx > n: give up and go hom or loop:
	unsigned stride = blockDim.x * gridDim.x;
	for (unsigned i = grid_ndx; i < n; i += stride)
	{
		vec_out[i].x = (mat.a11 * vec_in[i].x) + (mat.a12 * vec_in[i].y) + (mat.a13 * vec_in[i].z) + (mat.a14 * vec_in[i].w);
		vec_out[i].y = (mat.a21 * vec_in[i].x) + (mat.a22 * vec_in[i].y) + (mat.a23 * vec_in[i].z) + (mat.a24 * vec_in[i].w);
		vec_out[i].z = (mat.a31 * vec_in[i].x) + (mat.a32 * vec_in[i].y) + (mat.a33 * vec_in[i].z) + (mat.a34 * vec_in[i].w);
		vec_out[i].w = (mat.a41 * vec_in[i].x) + (mat.a42 * vec_in[i].y) + (mat.a43 * vec_in[i].z) + (mat.a44 * vec_in[i].w);
	}
}

//calulate the squared error ouf the output (x, y, z, z) for each training sample
__global__ void calc_dSEPerOut(unsigned n, const vec4* Labels, const vec4* Out, vec4* dErrPerOut)
{
	//in this section we are doing this part: dMSE/dOut = (1/N)sum(2(Y - dot(X,W)) with out the sum
	unsigned grid_ndx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned stride = blockDim.x * gridDim.x;
	for (unsigned i = grid_ndx; i < n; i += stride)
	{
		//so dErr/dOut_i: 2(Labels[i] - dot(In[i], W))
		//so dErr/dOut_i_x = 2(Labels[i].x - dot(In[i], W.X))
		// where W.X are the weights in the first row of the transformation matrix
		dErrPerOut[i].x = 2 * (Labels[i].x - Out[i].x);
		dErrPerOut[i].y = 2 * (Labels[i].y - Out[i].y);
		dErrPerOut[i].z = 2 * (Labels[i].z - Out[i].z);
		dErrPerOut[i].w = 2 * (Labels[i].w - Out[i].w);
	}
}

//propegate squared error to the weights from the output
//if this was huge I'm guesing you couldn't do it in parallel (or batch size would be small rather)
__global__ void calc_dSEPerW(unsigned n, const vec4* dErrPerOut, const vec4* Input, mat4x4* dSEPerW)
{
	//why did I break this up?  for reusability maybe, even though I don't need to consider that now
	//in this section we are doing dErrdw = 1/N sum_N(-2(Y - dot(X, W))X^T) withouth the sum
	//allright

	unsigned grid_ndx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned stride = blockDim.x * gridDim.x;
	for (unsigned i = grid_ndx; i < n; i += stride)
	{
		//(DErr/Dout)(Dout/DW)_i = dERR/dW_i = -2(Labels[i] - dot(In[i], W))X^T
		// we already have 2(Labels[i] - dot(In[i], W))  as dErrPerOut
		//the x output
		dSEPerW[i].a11 = -(dErrPerOut[i].x * Input[i].x);
		dSEPerW[i].a12 = -(dErrPerOut[i].x * Input[i].y);
		dSEPerW[i].a13 = -(dErrPerOut[i].x * Input[i].z);
		dSEPerW[i].a14 = -(dErrPerOut[i].x * Input[i].w);
		//the y output
		dSEPerW[i].a21 = -(dErrPerOut[i].y * Input[i].x);
		dSEPerW[i].a22 = -(dErrPerOut[i].y * Input[i].y);
		dSEPerW[i].a23 = -(dErrPerOut[i].y * Input[i].z);
		dSEPerW[i].a24 = -(dErrPerOut[i].y * Input[i].w);
		//the z output
		dSEPerW[i].a31 = -(dErrPerOut[i].z * Input[i].x);
		dSEPerW[i].a32 = -(dErrPerOut[i].z * Input[i].y);
		dSEPerW[i].a33 = -(dErrPerOut[i].z * Input[i].z);
		dSEPerW[i].a34 = -(dErrPerOut[i].z * Input[i].w);
		//the w output
		//actually i'm not going to bother we no what it should be
	}
}

__global__ void calc_SE(unsigned n, const vec4* ValLabels, const vec4* ValOut, vec4* ValSE)
{
	unsigned grid_ndx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned stride = blockDim.x * gridDim.x;
	for (unsigned i = grid_ndx; i < n; i += stride)
	{
		ValSE[i].x = (ValLabels[i].x - ValOut[i].x) * (ValLabels[i].x - ValOut[i].x);
		ValSE[i].y = (ValLabels[i].y - ValOut[i].y) * (ValLabels[i].y - ValOut[i].y);
		ValSE[i].z = (ValLabels[i].z - ValOut[i].z) * (ValLabels[i].z - ValOut[i].z);
		ValSE[i].w = (ValLabels[i].w - ValOut[i].w) * (ValLabels[i].w - ValOut[i].w);
	}
}

mat4x4 dMSEPerW_cpu(const mat4x4* dSEPerW, unsigned n)
{
	mat4x4 ans = {};
	ans.a11 = 0.0f;
	ans.a12 = 0.0f;
	ans.a13 = 0.0f;
	ans.a14 = 0.0f;
	ans.a21 = 0.0f;
	ans.a22 = 0.0f;
	ans.a23 = 0.0f;
	ans.a24 = 0.0f;
	ans.a31 = 0.0f;
	ans.a32 = 0.0f;
	ans.a33 = 0.0f;
	ans.a34 = 0.0f;
	for (unsigned i = 0; i < n; i++)
	{
		ans.a11 += dSEPerW[i].a11;
		ans.a12 += dSEPerW[i].a12;
		ans.a13 += dSEPerW[i].a13;
		ans.a14 += dSEPerW[i].a14;
		ans.a21 += dSEPerW[i].a21;
		ans.a22 += dSEPerW[i].a22;
		ans.a23 += dSEPerW[i].a23;
		ans.a24 += dSEPerW[i].a24;
		ans.a31 += dSEPerW[i].a31;
		ans.a32 += dSEPerW[i].a32;
		ans.a33 += dSEPerW[i].a33;
		ans.a34 += dSEPerW[i].a34;
	}
	ans.a11 = ans.a11 / n;
	ans.a12 = ans.a12 / n;
	ans.a13 = ans.a13 / n;
	ans.a14 = ans.a14 / n;
	ans.a21 = ans.a21 / n;
	ans.a22 = ans.a22 / n;
	ans.a23 = ans.a23 / n;
	ans.a24 = ans.a24 / n;
	ans.a31 = ans.a31 / n;
	ans.a32 = ans.a32 / n;
	ans.a33 = ans.a33 / n;
	ans.a34 = ans.a34 / n;
	
	return ans;
}

void step_negMSEperW_cpu(mat4x4 dMSEPerW, mat4x4* pCurrentTransform)
{
	pCurrentTransform->a11 = pCurrentTransform->a11 - STEP_SIZE * dMSEPerW.a11;
	pCurrentTransform->a12 = pCurrentTransform->a12 - STEP_SIZE * dMSEPerW.a12;
	pCurrentTransform->a13 = pCurrentTransform->a13 - STEP_SIZE * dMSEPerW.a13;
	pCurrentTransform->a14 = pCurrentTransform->a14 - STEP_SIZE * dMSEPerW.a14;

	pCurrentTransform->a21 = pCurrentTransform->a21 - STEP_SIZE * dMSEPerW.a21;
	pCurrentTransform->a22 = pCurrentTransform->a22 - STEP_SIZE * dMSEPerW.a22;
	pCurrentTransform->a23 = pCurrentTransform->a23 - STEP_SIZE * dMSEPerW.a23;
	pCurrentTransform->a24 = pCurrentTransform->a24 - STEP_SIZE * dMSEPerW.a24;

	pCurrentTransform->a31 = pCurrentTransform->a31 - STEP_SIZE * dMSEPerW.a31;
	pCurrentTransform->a32 = pCurrentTransform->a32 - STEP_SIZE * dMSEPerW.a32;
	pCurrentTransform->a33 = pCurrentTransform->a33 - STEP_SIZE * dMSEPerW.a33;
	pCurrentTransform->a34 = pCurrentTransform->a34 - STEP_SIZE * dMSEPerW.a34;
}

vec4 MSE_cpu(unsigned n, const vec4* ValSE, float* MMSE)
{
	vec4 ans = { 0, 0, 0, 0 };
	for (unsigned i = 0; i < n; i++)
	{
		ans.x += ValSE[i].x;
		ans.y += ValSE[i].y;
		ans.z += ValSE[i].z;
		ans.w += ValSE[i].w;
	}
	ans.x = ans.x / n;
	ans.y = ans.y / n;
	ans.z = ans.z / n;
	ans.w = ans.w / n;

	*MMSE = (ans.x + ans.y + ans.z + ans.w) / 4;

	return ans;
}


int main()
{
	//the optimizing potion never sees the transform just the effects of applying the transform (labels)
	mat4x4 TransformTarget = {};
	TransformTarget.a11 = 11.0f;
	TransformTarget.a12 = 12.0f;
	TransformTarget.a13 = 13.0f;
	TransformTarget.a14 = 14.0f;
	TransformTarget.a21 = 21.0f;
	TransformTarget.a22 = 22.0f;
	TransformTarget.a23 = 23.0f;
	TransformTarget.a24 = 24.0f;
	TransformTarget.a31 = 31.0f;
	TransformTarget.a32 = 32.0f;
	TransformTarget.a33 = 33.0f;
	TransformTarget.a34 = 34.0f;

	vec4* Input;
	vec4* Labels;
	vec4* Output;
	vec4* dErrPerOut;  //to hold the error of the output
	mat4x4* dSEPerW;  //hold the 

	vec4* ValInput;
	vec4* ValLabels;
	vec4* ValOutput;
	vec4* ValSE;

	//alright

	//managed by the unified memory system...
	//pass in pointers to pointers of course
	cudaMallocManaged(&Input, sizeof(vec4) * BATCH_SIZE);
	cudaMallocManaged(&Labels, sizeof(vec4)* BATCH_SIZE);
	cudaMallocManaged(&Output, sizeof(vec4) * BATCH_SIZE);
	cudaMallocManaged(&dErrPerOut, sizeof(vec4) * BATCH_SIZE);
	cudaMallocManaged(&dSEPerW, sizeof(mat4x4) * BATCH_SIZE);

	cudaMallocManaged(&ValInput, sizeof(vec4) * BATCH_SIZE);
	cudaMallocManaged(&ValLabels, sizeof(vec4) * BATCH_SIZE);
	cudaMallocManaged(&ValOutput, sizeof(vec4) * BATCH_SIZE);
	cudaMallocManaged(&ValSE, sizeof(vec4) * BATCH_SIZE);

	//maybe srand sucks but whatever
	unsigned tid = 0;  //a thread id for the future...  whatever
	srand(time(NULL) + tid);

	unsigned Max = 50;

	mat4x4 TransformtoOptimize = {};
	int rndint = rand() % Max;
	TransformtoOptimize.a11 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a12 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a13 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a14 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a21 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a22 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a23 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a24 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a31 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a32 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a33 = (float)rndint;
	rndint = rand() % Max;
	TransformtoOptimize.a34 = (float)rndint;

	//validation data
	for (unsigned i = 0; i < BATCH_SIZE; i++)
	{
		rndint = rand() % Max;
		ValInput[i].x = (float)rndint;

		rndint = rand() % Max;
		ValInput[i].y = (float)rndint;

		rndint = rand() % Max;
		ValInput[i].z = (float)rndint;

		rndint = rand() % Max;
		ValInput[i].w = (float)rndint;
	}


	for (unsigned Epoch = 0; Epoch < EPOCHS; Epoch++)
	{
		//BEGIN THE BATCH

		//BATCH - TRAINING DATA and ATTEMPT
		//training input
		for (unsigned i = 0; i < BATCH_SIZE; i++)
		{
			rndint = rand() % Max;
			Input[i].x = (float)rndint;

			rndint = rand() % Max;
			Input[i].y = (float)rndint;

			rndint = rand() % Max;
			Input[i].z = (float)rndint;

			rndint = rand() % Max;
			Input[i].w = (float)rndint;
		}

		//perhaps we shouldn't assign so many threads but eh
		unsigned n_blocks = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

		mult << < n_blocks, BLOCK_SIZE >> > (TransformTarget, BATCH_SIZE, Input, Labels);
		cudaDeviceSynchronize(); //hold up

		mult << < n_blocks, BLOCK_SIZE >> > (TransformtoOptimize, BATCH_SIZE, Input, Output);
		cudaDeviceSynchronize();

		//now we can begin the optimizing (learning) portion (the target is not observed of course, just the labels (result))

		//BATCH - OPTIMIZATION, BACKPROPAGATION, and SECTION

		//it is somewhat pointless to do the working our for the linear combination
		//anyways, assuming the Neurons value is sum(W) where W is the weights
		//assuming, we are using MSE = (1/N)sum_N((Y - dot(X,W))^2)
		//then, dMSE/dOut = (1/N)sum(2(Y - dot(X,W))
		//and, dOut/dW = (1/N)sum(-X^T)
		//dErr/dW = (dMSE/dOut)(dOut/dW) = dErr/dW = (1/N) sum_N(-2(Y - dot(X,W))X^T)
		//dErrdw = 1/N sum_N(-2(Y - dot(X, W))X^T)

		//let's do the first part
		//get the slope of MSE with respect to the output (neurons)
		calc_dSEPerOut << < n_blocks, BLOCK_SIZE >> > (BATCH_SIZE, Labels, Output, dErrPerOut);
		cudaDeviceSynchronize();
		calc_dSEPerW << < n_blocks, BLOCK_SIZE >> > (BATCH_SIZE, dErrPerOut, Input, dSEPerW);
		cudaDeviceSynchronize();

		//BATCH - CHECK VALIDATION ERROR BEFORE STEP
		mult << < n_blocks, BLOCK_SIZE >> > (TransformTarget, BATCH_SIZE, ValInput, ValLabels);
		cudaDeviceSynchronize();
		mult << < n_blocks, BLOCK_SIZE >> > (TransformtoOptimize, BATCH_SIZE, ValInput, ValOutput);
		cudaDeviceSynchronize();
		//error portion
		calc_SE << < n_blocks, BLOCK_SIZE >> > (BATCH_SIZE, ValLabels, ValOutput, ValSE);
		cudaDeviceSynchronize();

		float ValMSE_preStep = 0;
		vec4 ValMSE4_preSetp = MSE_cpu(BATCH_SIZE, ValSE, &ValMSE_preStep);
		printf("\nValidation Erro BEFORE step of epoch %u: %f", Epoch, ValMSE_preStep);

		//BATCH - STEP
		mat4x4 dMSEPerW = dMSEPerW_cpu(dSEPerW, BATCH_SIZE);
		step_negMSEperW_cpu(dMSEPerW, &TransformtoOptimize);

		//BATCH - VALIDATION ERROR AFTER STEP
		mult << < n_blocks, BLOCK_SIZE >> > (TransformTarget, BATCH_SIZE, ValInput, ValLabels);
		cudaDeviceSynchronize();
		mult << < n_blocks, BLOCK_SIZE >> > (TransformtoOptimize, BATCH_SIZE, ValInput, ValOutput);
		cudaDeviceSynchronize();
		//error portion
		calc_SE << < n_blocks, BLOCK_SIZE >> > (BATCH_SIZE, ValLabels, ValOutput, ValSE);
		cudaDeviceSynchronize();

		float ValMSE_postStep = 0;
		vec4 ValMSE4_postSetp = MSE_cpu(BATCH_SIZE, ValSE, &ValMSE_postStep);
		printf("\nValidation Erro AFTER step of epoch  %u: %f", Epoch, ValMSE_postStep);
	}
	
	printf("\n\nTried to learn this transform: \n%f, %f, %f, %f \n%f, %f, %f, %f \n%f, %f, %f, %f \n%f, %f, %f, %f", 
		TransformTarget.a11, TransformTarget.a12, TransformTarget.a13, TransformTarget.a14,
		TransformTarget.a21, TransformTarget.a22, TransformTarget.a23, TransformTarget.a24,
		TransformTarget.a31, TransformTarget.a32, TransformTarget.a33, TransformTarget.a34,
		TransformTarget.a41, TransformTarget.a42, TransformTarget.a43, TransformTarget.a44);

	printf("\n\nAnd got: \n%f, %f, %f, %f \n%f, %f, %f, %f \n%f, %f, %f, %f \n%f, %f, %f, %f",
		TransformtoOptimize.a11, TransformtoOptimize.a12, TransformtoOptimize.a13, TransformtoOptimize.a14,
		TransformtoOptimize.a21, TransformtoOptimize.a22, TransformtoOptimize.a23, TransformtoOptimize.a24,
		TransformtoOptimize.a31, TransformtoOptimize.a32, TransformtoOptimize.a33, TransformtoOptimize.a34,
		TransformtoOptimize.a41, TransformtoOptimize.a42, TransformtoOptimize.a43, TransformtoOptimize.a44);


	cudaFree(Input);
	cudaFree(Labels);
	cudaFree(Output);
	cudaFree(dErrPerOut);
	cudaFree(dSEPerW);

	cudaFree(ValInput);
	cudaFree(ValLabels);
	cudaFree(ValOutput);
	cudaFree(ValSE);

	printf("\done");

	return 0;
}
