import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import scipy.io as scio
import torch
from pycuda.compiler import SourceModule
import time
import math
import CDNN
import pycuda.gpuarray as gpuarray

dtype = torch.float
device = torch.device("cpu")
device_train = torch.device("cuda:0")
x = torch.cuda.FloatTensor(1)
class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer(self):
        return self.t.data_ptr()


mod = SourceModule(r"""
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
        //if(tx == 0 && ty ==0){
        //printf("%f\n",resc[(tx*(clos-2)+ty)*9+8]);
        //printf("%d\n",(tx*(clos-2)+ty)*9+8);
        //printf("%d\n",(tx+2-x)*clos+ty+2-y);
       // printf("%f\n",intensity[(tx+2-x)*clos+ty+2-y]);
        //}
        }

  }
""")
# 加载网络
Dnnpath = 'nets/DNN9/cnn_q7/'  #DNN9/cnn_q7/
fnet = CDNN.CDNN(9, 901).cuda()
checkpoint = torch.load(Dnnpath +'amp_checkpoint.pt')
fnet.load_state_dict(checkpoint['fnet'])
fnet.eval()
#加载数据
pic = '1'   # 1_resize
path_data = 'E:/Intensity2/'+ pic +'_Intensity.mat'  #E:/Intensity2/
data = scio.loadmat(path_data)['data'].astype(np.float32)
pre_process = mod.get_function("pre_process")
row, cols = data.shape
print('data shape :',data.shape)
data = data.reshape(-1)
resc = np.zeros(((row-2)*(cols-2),9)).astype(np.float32)

#数据标准化
scale = scio.loadmat(Dnnpath+'scale.mat')
mean = scale['mean']
std = scale['std']

d_data = torch.tensor(data,device=device_train, dtype=dtype)
d_resc = torch.tensor(resc,device=device_train, dtype=dtype)
d_mean = torch.tensor(mean, device=device_train, dtype=dtype)
d_std = torch.tensor(std, device=device_train, dtype=dtype)

# d_data = gpuarray.to_gpu(data)
# d_resc = gpuarray.to_gpu(resc)
# d_mean = gpuarray.to_gpu(mean)
# d_std = gpuarray.to_gpu(std)
del data, resc, mean, std

#配置cuda
threadsperblock = (16,16,1)
blockspergrid_x = int(math.ceil(row/threadsperblock[0]))
blockspergrid_y = int(math.ceil(cols/threadsperblock[1]))
blockspergrid = (blockspergrid_x,blockspergrid_y)
# start = time.time()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()
pre_process(
        Holder(d_data), Holder(d_resc), Holder(d_mean), Holder(d_std),np.int32(row), np.int32(cols),
        block=threadsperblock, grid=blockspergrid)
out = fnet(d_resc)
ender.record()
torch.cuda.synchronize()
stop = starter.elapsed_time(ender)
# stop = time.time() - start
# resc = resc.reshape((row-2)*(cols-2),9)
# d_resc = d_resc.reshape((row-2)*(cols-2),-1)
# resc = torch.tensor(d_resc, dtype=torch.float32, device=device_train)
# scio.savemat('E:/Intensity2/'+pic+'_resc.mat', {'data':d_resc.detach().cpu().numpy()})
print('process time:', stop / 1000)
print('resc:', d_resc)