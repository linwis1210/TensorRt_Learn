import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import scipy.io as scio
import torch
from pycuda.compiler import SourceModule
import time
import math
import CDNN1 as CDNN
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
               save_engine=True):
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
Dnnpath = 'nets/DCNN/'  #DNN9/cnn_q7/
fnet = CDNN.CDNN().cuda()
checkpoint = torch.load(Dnnpath +'480_fnet.pth')
fnet.load_state_dict(checkpoint)
fnet.eval()
#加载数据
pic = '1'   # 1_resize
path_data = 'data/'+ pic +'_Intensity1.mat'  #E:/Intensity2/
# data = scio.loadmat(path_data)['data'].astype(np.float32)
data = np.random.randn(700,700)
row, cols = data.shape
print('data shape :',data.shape)

#数据标准化
# scale = scio.loadmat(Dnnpath+'scale.mat')
# mean = scale['mean']
# std = scale['std']

#加载引擎
onnx_model_path = Dnnpath + 'fnet_700.onnx'
trt_engine_path = os.path.join(Dnnpath, 'fnet_onnx_engine.trt')
print('engine before ok')
engine = get_engine(1, onnx_model_path, trt_engine_path, save_engine=True)
print('engine after ok')
context = engine.create_execution_context()
# Allocate buffers for input and output

d_data = torch.tensor(data,device=device_train, dtype=dtype)
# d_mean = torch.tensor(mean, device=device_train, dtype=dtype)
# d_std = torch.tensor(std, device=device_train, dtype=dtype)
d_out = torch.zeros((1,901,data.shape[0],data.shape[1]), device=device_train, dtype=dtype)
# d_data = gpuarray.to_gpu(data)
# d_resc = gpuarray.to_gpu(resc)
# d_mean = gpuarray.to_gpu(mean)
# d_std = gpuarray.to_gpu(std)
del data

# inputs, outputs, bindings, stream = allocate_buffers(engine)

buffers = []
buffers.append(d_data.data_ptr())
buffers.append(d_out.data_ptr())

stream = torch.cuda.Stream()

start = time.time()

# Load data to the buffer
context.execute_async_v2(buffers,stream.cuda_stream)
#torch.cuda.synchronize()
stream.synchronize()

end = time.time() - start
print('Rt time:',end)
#print('out:',d_out,d_out.shape)

start1 = time.time()
out = fnet(d_data.view(1,1,700,700))
torch.cuda.synchronize()
end1 =time.time() - start1
print('Pytorch time:',end1)
del d_data, fnet
loss = np.mean((d_out.detach().cpu().numpy() - out.detach().cpu().numpy())**2)
print('loss:', loss)