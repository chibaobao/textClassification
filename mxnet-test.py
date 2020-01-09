from mxnet import nd
import functions as d2l

ctx = d2l.try_all_gpus() #探测是否支持GPU
print("mxnet使用了：",ctx)
x = nd.array([1, 2, 3])
print("通过测试")
