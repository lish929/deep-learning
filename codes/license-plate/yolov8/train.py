# -*- coding: utf-8 -*-
# @Time    : 2024/9/6 11:30
# @Author  : Lee
# @Project ：license-plate 
# @File    : train.py


from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # https://docs.ultralytics.com/modes/train/#train-settings 选择需要的参数以及超参数
    results = model.train(
        data=r"yolov8/ccpd.yaml",       # 数据集配置文件
        epochs=100,             # 训练轮次
        patience=10,            # 无提升轮次 停止训练
        batch=-1,               # 批次
        imgsz=640,              # 图像尺寸
        save=True,              # 保存训练权重
        save_period=10,         # 保存轮次
        cache=False,             # 加载数据到内存
        device=0,               # 训练设备
        workers=2,              # 加载数据线程数
        optimizer='auto',       # 优化器选择
        verbose=False,          # 监控训练过程
        deterministic=True,     # 确定性训练
        rect=False,             # 实现批量最小填充
        cos_lr=False,           # 余弦学习率
        close_mosaic=10,        # 关闭马赛克增强轮次
        amp=True,               # 混合精度训练
        fraction=1.0,           # 使用数据集的一部分
        profile=False,          # 启用 ONNX 和 TensorRT 速度分析
        freeze=None,            # 冻结前N层或指定索引层
        lr0=0.01,               # 初始学习率
        lrf=0.01,               # 最终学习率lr0*lrf
        momentum=0.937,         # SGD动量因子 Adam beta1
        weight_decay=0.0005,    # l2正则化 防止过拟合
        warmup_epochs=3,        # warmup轮次
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,     # 偏差参数的学习率 稳定初始轮次的模型训练
        box=7.5,                # 损失比例
        cls=0.5,
        dfl=1.5,
        nbs=64,
        val=True,
        plots=True,
    )