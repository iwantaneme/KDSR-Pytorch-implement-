import torch
from thop import profile
from model.ESRGAN import ESRGAN as Generator
import time
#加载torch模型
student =Generator(3,3,scale_factor=4,n_basic_block=10)
teacher = Generator(3,3,scale_factor=4)


# 默认使用cpu，修改为GPU
inputs_224 = torch.ones([1,3,64,64]).type(torch.float32).to(torch.device("cuda"))
student.to(torch.device("cuda"))   #10 GFLOPs and 7.30M parameters
teacher.to(torch.device("cuda"))   #45 GFLOPs and 16.66M parameters

#model为torch 模型  inputs为模型输入 batchsize设置为1即可
flops,pararms = profile(model=student,inputs=(inputs_224,))
print("Model:{:.2f} GFLOPs and {:.2f}M parameters".format(flops/1e9,pararms/1e6))


student.eval()
teacher.eval()
output1=student(inputs_224)
output2=teacher(inputs_224)
starttime=time.time()
for i in range(10):
    with torch.no_grad():
        output1 = student(inputs_224)
endtime=time.time()
print('student inference time=%.4f'%((endtime-starttime)/10))

flops,pararms = profile(model=teacher,inputs=(inputs_224,))
print("Model:{:.2f} GFLOPs and {:.2f}M parameters".format(flops/1e9,pararms/1e6))
starttime=time.time()
for i in range(10):
    with torch.no_grad():
        output1 = teacher(inputs_224)
endtime=time.time()
print('teacher inference time=%.4f'%((endtime-starttime)/10))