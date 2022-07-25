#include<bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

__global__ void pre_process(float *intensity, float *resc, float *mean, float *std,const int row, const int clos) {
  // Calculate the global thread positions
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  
  if((tx<row-2) && (ty<clos-2)){
        int x = tx % 3;
        int y = ty % 3;
        resc[(tx*(clos-2)+ty)*9] = (intensity[(tx+(3-x)%3)*clos+ty+(3-y)%3] - mean[0]) / std[0];
        resc[(tx*(clos-2)+ty)*9+1] = (intensity[(tx+(3-x)%3)*clos+ty+(4-y)%3] - mean[1]) / std[1];
        resc[(tx*(clos-2)+ty)*9+2] = (intensity[(tx+(3-x)%3)*clos+ty+2-y] - mean[2]) / std[2];
        resc[(tx*(clos-2)+ty)*9+3] = (intensity[(tx+(4-x)%3)*clos+ty+(3-y)%3] - mean[3]) / std[3];
        resc[(tx*(clos-2)+ty)*9+4] = (intensity[(tx+(4-x)%3)*clos+ty+(4-y)%3] - mean[4]) / std[4];
        resc[(tx*(clos-2)+ty)*9+5] = (intensity[(tx+(4-x)%3)*clos+ty+2-y] - mean[5]) / std[5];
        resc[(tx*(clos-2)+ty)*9+6] = (intensity[(tx+2-x)*clos+ty+(3-y)%3] - mean[6]) / std[6];
        resc[(tx*(clos-2)+ty)*9+7] = (intensity[(tx+2-x)*clos+ty+(4-y)%3] - mean[7])/ std[7];
        resc[(tx*(clos-2)+ty)*9+8] = (intensity[(tx+2-x)*clos+ty+2-y] - mean[8]) / std[8]; 
        }
  }