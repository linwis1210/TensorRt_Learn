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
import pycuda.driver as cuda
import os
import tensorrt as trt

dtype = torch.float
device = torch.device("cpu")
device_train = torch.device("cuda:0")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
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
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", fp16_mode=True, int8_mode=False,
               save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:

            print(network.num_layers)
            print(network.num_inputs)
            print(network.num_outputs)
            print(network.name)

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            builder.max_batch_size = max_batch_size
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            elif int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
            else:
                config.set_flag(trt.BuilderFlag.REFIT)

            flag = builder.is_network_supported(network, config)
            print('flag', flag)

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                # print(type(model.read()))
                parser.parse(model.read())
                # parser.parse_from_file(onnx_file_path)
                assert network.num_layers > 0, 'Failed to parse ONNX model.Please check if the ONNX model is compatible '

            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

# 加载网络
Dnnpath = 'nets/CNN/'  #DNN9/cnn_q7/
fnet = CDNN.CDNN(9, 901).cuda()
checkpoint = torch.load(Dnnpath +'amp_checkpoint.pt')
fnet.load_state_dict(checkpoint['fnet'])
fnet.eval()
#加载数据
pic = '1'   # 1_resize
path_data = 'data/'+ pic +'_Intensity1.mat'  #E:/Intensity2/
# data = scio.loadmat(path_data)['data'].astype(np.float32)
data = np.random.randn(1024,1024)
pre_process = mod.get_function("pre_process")
row, cols = data.shape[0]//2, data.shape[1]
print('data shape :',data.shape)
data1 = data[512:1024,:]
data = data[0:512,:]
print('data1 shape:', data1.shape)
data = data.reshape(-1)
data1 = data1.reshape(-1)


resc = np.zeros(((row-2)*(cols-2),9)).astype(np.float32)
#数据标准化
scale = scio.loadmat(Dnnpath+'scale.mat')
mean = scale['mean']
std = scale['std']

#加载引擎
onnx_model_path = Dnnpath + 'fnet_std.onnx'
trt_engine_path = os.path.join(Dnnpath, 'fnet_onnx_engine.trt')
print('engine before ok')
engine = get_engine(1, onnx_model_path, trt_engine_path, save_engine=True)
print('engine after ok')
context = engine.create_execution_context()
# Allocate buffers for input and output

d_data = torch.tensor(data,device=device_train, dtype=dtype)
d_data1 = torch.tensor(data1, device=device_train, dtype=dtype)
d_resc = torch.tensor(resc,device=device_train, dtype=dtype)
d_resc1 = torch.tensor(resc,device=device_train, dtype=dtype)
d_mean = torch.tensor(mean, device=device_train, dtype=dtype)
d_std = torch.tensor(std, device=device_train, dtype=dtype)
d_out = torch.zeros((resc.shape[0], 901), device=device_train, dtype=dtype)
d_out1 = torch.zeros((resc.shape[0], 901), device=device_train, dtype=dtype)
# d_data = gpuarray.to_gpu(data)
# d_resc = gpuarray.to_gpu(resc)
# d_mean = gpuarray.to_gpu(mean)
# d_std = gpuarray.to_gpu(std)
del data, resc, mean, std,data1

#配置cuda
threadsperblock = (16,16,1)
blockspergrid_x = int(math.ceil(row/threadsperblock[0]))
blockspergrid_y = int(math.ceil(cols/threadsperblock[1]))
blockspergrid = (blockspergrid_x,blockspergrid_y)
# inputs, outputs, bindings, stream = allocate_buffers(engine)
# bindings = [int(d_resc.data_ptr()),int(d_out.data_ptr())]
buffers = []
# [buffers.append([d_r.data_ptr(), d_o.data_ptr()])for d_r in d_resc for d_o in d_out]
buffers.append(d_resc.data_ptr())
buffers.append(d_out.data_ptr())

buffers1 = []
buffers1.append(d_resc1.data_ptr())
buffers1.append(d_out1.data_ptr())
# [buffers.append(d_o.data_ptr()) for d_o in d_out]
stream = torch.cuda.Stream()
# stream2 = torch.cuda.Stream()
start = time.time()
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# starter.record()

# pre_process(
#         d_data, d_resc, d_mean, d_std,np.int32(row), np.int32(cols),
#         block=threadsperblock, grid=blockspergrid)
pre_process(
        Holder(d_data), Holder(d_resc), Holder(d_mean), Holder(d_std),np.int32(row), np.int32(cols),
        block=threadsperblock, grid=blockspergrid)

# out = fnet(d_resc)
# Load data to the buffer
torch.cuda.synchronize()
context.execute_async_v2(buffers,stream.cuda_stream)
stream.synchronize()
pre_process(
        Holder(d_data1), Holder(d_resc1), Holder(d_mean), Holder(d_std),np.int32(row), np.int32(cols),
        block=threadsperblock, grid=blockspergrid)
torch.cuda.synchronize()
context.execute_async_v2(buffers1,stream.cuda_stream)
# context.execute_async(batch_size=1022*1022, bindings=buffers, stream_handle=stream.cuda_stream)
torch.cuda.synchronize()
# ender.record()
stream.synchronize()
end = time.time() - start
print('time:',end)
out2 = torch.vstack((d_out, d_out1))
# stop = starter.elapsed_time(ender)
# print('process time:', stop / 1000)
# print('out:',d_out,d_out.shape)
# print('out1:',d_out1, d_out1.shape)
print('out2:', out2)
del d_out, d_out1
out = fnet(torch.vstack((d_resc,d_resc1)))
loss = torch.mean((out2 - out)**2)
print('loss:', loss)