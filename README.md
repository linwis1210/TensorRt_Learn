# TensorRT部署优化

- CDNN、CDNN1为模型结构

- produce_onnx.py 为把训练好的模型转化为onnx

- run_cuda_9.py、run_cuda_16.py为预处理优化

- run_dnn_onnx.py、run_cnn_onnx.py分别为对dnn和cnn进行推理优化

## 优化结果：

- 对于1024\*1024的光强矩阵，预处理加推理的时间为0.031s左右，与原模型的RMSE为8.2201e-05。原模型推理1024\*1024需要0.065S左右，且不包括预处理时间。


