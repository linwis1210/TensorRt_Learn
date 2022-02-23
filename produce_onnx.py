import torch
import CDNN as CDNN
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import os
import time
import scipy.io as scio
# import onnx
#
input_name = ['input']
output_name = ['output']
input = torch.randn(700,9).cuda()
model = CDNN.CDNN(9,901).cuda()
checkpoint = torch.load('nets/CNN/amp_checkpoint.pt') #nets/CNN/
model.load_state_dict(checkpoint['fnet'])
model.eval()
torch.onnx.export(model, input, 'nets/CNN/fnet_test.onnx', input_names=input_name, output_names=output_name, verbose=True)
# test = onnx.load('nets/CNN/fnet_std.onnx')
# onnx.checker.check_model(test)
print('run success')