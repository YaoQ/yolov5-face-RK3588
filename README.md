# yolov5_face_rknnlite2

## 介绍
使用RKNNLite 实现在RK3588上运行YOLOv5_face模型，并实现人脸检测功能。

## 模型
```
输入： 1 3 640 640
输出：
1 80 80 16
1 40 40 16
1 20 20 16
``` 

## 测试
```
python yolov5-face_rknnlite.py 
```