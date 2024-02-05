# RK3588 yolov5-face RKNPU 人脸检测

## 介绍
1. 基于[rknpu2 rknn yolov5 demo](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknpu2/examples/rknn_yolov5_demo),在RK3588上运行[Yolov5-face](https://github.com/deepcam-cn/yolov5-face)人脸检测，并带有5个点的landmark解析
2. Python版本的人脸检测，用RKNNLite API，也同样在RK3588上验证

## 环境依赖
- RK3588 Debian11 
- rknntoolit 1.6.0
- rknnnpu 1.6.0
- opencv4+

## 模型
python 使用的模型尺寸
```
输入： 1 3 640 640
输出：
1 3 80 80 16
1 3 40 40 16
1 3 20 20 16
``` 

由于RKNPU 版本的更新，有修改过尺寸，故特意罗列可能会处理的模型输出，会影响后处理的实现。
C++版本的模型尺寸
老版本模型尺寸
```
输入： 1 3 640 640
输出：
1 3 16 80 80
1 3 16 40 40
1 3 16 20 20
```

最新的版本模型输出要求
```
输入： 1 3 640 640
输出：
1 48 80 80
1 48 40 40
1 48 20 20
```

## python 代码测试
```bash
# python
python yolov5-face_rknnlite.py 
```

## C++ 代码测试
```
cd cpp 
bash build-linux_RK3588.sh # 直接在RK3588环境下编译

cd install
./rknn_yolov5_demo ./model/yolov5n-face_1x48x80x80.rknn ../../img/face.jpg
```

## 模型转换说明
在导出onnx模型时，由于输出的尺度不一样，特别说明一下，截断得到3输出，尺寸是(1, 48, 20, 20)
```
yolov5-face/yolo.py 

 45     def forward(self, x):
 46         # x = x.copy()  # for profiling
 47         z = []  # inference output
 48         if self.export_cat:
 49             for i in range(self.nl):
 50                 x[i] = self.m[i](x[i])  # conv
 51                 #bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
 52                 #x[i] = x[i].view(bs, self.na, self.no, ny, nx).contiguous() #.permute(0, 1, 2, 3, 4).contiguous()
 53             return x

```
备注：
如果需要导出其他的尺度，需要修改52行代码，将view的尺寸修改为相应的要求来适配：(bs, self.na, self.no, ny, nx)
