# PP-MattingV2-onnxrun-cpp-py
使用ONNXRuntime部署PP-MattingV2人像分割，一共包含18个onnx模型，依然是C++和Python两个版本的程序
起初，我想使用opencv做部署的，但是opencv的dnn模块读取onnx文件出错， 无赖只能使用onnxruntime做部署了。
本套程序一共提供了18个onnx模型，onnx文件在百度云盘，下载链接：https://pan.baidu.com/s/1Er8-Vm7Xf3HsvMiHXrIiCg 
提取码：7rgj
images文件夹里有4张漂亮女生的图片，可以当做测试数据来使用
