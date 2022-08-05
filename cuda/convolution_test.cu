#include "cuda_runtime.h"
#include <stdio.h>
#include <cudnn.h>
#include <cudnn_ops_infer.h>
using namespace std;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      printf(cudnnGetErrorString(status));                   \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

int main() {
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);
  int in_channels = 1; //每层卷积的输入通道
  int out_channels = 9; //每层卷积的输出通道
  int batch_size = 1;
  int image_height = 1024; // 图片尺寸
  int image_width =  1024;

  int kernel_height = 3;//卷积核尺寸
  int kernel_width = 3;
  int padding = 1;
  int stride = 1 ;

  bool biasflag = true;

cudnnTensorDescriptor_t input_descriptor;
checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW ,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/in_channels,
                                      /*image_height=*/image_height,
                                      /*image_width=*/image_width));

cudnnTensorDescriptor_t output_descriptor;
checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/out_channels,
                                      /*image_height=*/image_height,
                                      /*image_width=*/image_width));

cudnnTensorDescriptor_t z_descriptor;
checkCUDNN(cudnnCreateTensorDescriptor(&z_descriptor));
checkCUDNN(cudnnSetTensor4dDescriptor(z_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/out_channels,
                                      /*image_height=*/image_height,
                                      /*image_width=*/image_width));

cudnnTensorDescriptor_t bias_descriptor;
checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/out_channels,
                                      /*kernel_height=*/kernel_height,
                                      /*kernel_width=*/kernel_width));

cudnnFilterDescriptor_t kernel_descriptor;
checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/out_channels,
                                      /*in_channels=*/in_channels,
                                      /*kernel_height=*/kernel_height,
                                      /*kernel_width=*/kernel_width));

cudnnConvolutionDescriptor_t convolution_descriptor;
checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/padding,
                                           /*pad_width=*/padding,
                                           /*vertical_stride=*/stride,
                                           /*horizontal_stride=*/stride,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNN_DATA_FLOAT));

double coef = 100; //specifies the upper bound
cudnnActivationDescriptor_t activationDesc;
checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
checkCUDNN(cudnnSetActivationDescriptor(activationDesc, 
                                        /*cudnnActivationMode_t*/ CUDNN_ACTIVATION_RELU,
                                        /*cudnnNanPropagation_t*/ CUDNN_NOT_PROPAGATE_NAN,
                                        coef
                                        ));    

// cudnnConvolutionFwdAlgoPerf_t convolution_a;
// checkCUDNN(
//     cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
//                                         input_descriptor,
//                                         kernel_descriptor,
//                                         convolution_descriptor,
//                                         output_descriptor,
//                                         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
//                                         /*memoryLimitInBytes=*/0,
//                                         &convolution_a));

size_t workspace_bytes = 0;
cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));


//分配内存
void* d_workspace{nullptr};
cudaMalloc(&d_workspace, workspace_bytes);

//batch_size * channels * height * width
int in_image_bytes = batch_size * in_channels * image_height * image_width * sizeof(float); //输入图片的尺寸
int out_image_bytes = batch_size * out_channels * image_height * image_width* sizeof(float); //输出图片的尺寸
int bias_bytes = batch_size * out_channels * kernel_height * kernel_width * sizeof(float);  //偏置值矩阵

float* d_input{nullptr};
cudaMalloc(&d_input, in_image_bytes);

float* z_input{nullptr};
cudaMalloc(&z_input, out_image_bytes);

float* bias_input{nullptr};
cudaMalloc(&bias_input, bias_bytes);

float* d_output{nullptr};
cudaMalloc(&d_output, out_image_bytes);
cudaMemset(d_output, 0, out_image_bytes);

int h_kernel_bytes = out_channels * in_channels * kernel_height * kernel_width * sizeof(float); //卷积核参数尺寸

float* d_kernel{nullptr};
cudaMalloc(&d_kernel, h_kernel_bytes);

//调用函数
const float alpha = 1, beta = 0;
if(biasflag == false){
checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alpha,
                                   input_descriptor,
                                   d_input,
                                   kernel_descriptor,
                                   d_kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &beta,
                                   output_descriptor,
                                   d_output));
}else {
checkCUDNN(cudnnConvolutionBiasActivationForward(cudnn,
                                   &alpha,
                                   input_descriptor,
                                   d_input,
                                   kernel_descriptor,
                                   d_kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &beta,
                                   z_descriptor,
                                   z_input,
                                   bias_descriptor,
                                   bias_input,
                                   activationDesc,
                                   output_descriptor,
                                   d_output));
}
float* h_output = new float[out_image_bytes];
cudaMemcpy(h_output, d_output, out_image_bytes, cudaMemcpyDeviceToHost);

// Do something with h_output ...

//释放内存
delete[] h_output;
cudaFree(d_kernel);
cudaFree(d_input);
cudaFree(d_output);
cudaFree(d_workspace);
cudaFree(z_input);
cudaFree(bias_input);

cudnnDestroyTensorDescriptor(input_descriptor);
cudnnDestroyTensorDescriptor(output_descriptor);
cudnnDestroyFilterDescriptor(kernel_descriptor);
cudnnDestroyConvolutionDescriptor(convolution_descriptor);
cudnnDestroyTensorDescriptor(z_descriptor);
cudnnDestroyTensorDescriptor(bias_descriptor);
cudnnDestroyActivationDescriptor(activationDesc);
cudnnDestroy(cudnn);
};