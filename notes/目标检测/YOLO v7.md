# YOLO v7

	网络改进的两个主要方面：

* 效果：精度越高越好
* 效率：推理速度越快越好

‍

### 1 优化点

* 骨干网络：ELAN
* neck和head部分：RepBlock代替CSPBlock
* 引入MP结构做下采样：将池化与卷积的下采样合并
* 损失与yolov5相同
