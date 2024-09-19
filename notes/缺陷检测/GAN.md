# GAN

### 1. 什么是GAN

​![image](assets/image-20240827213423-cxuaa3g.png)​

	这里举一个例子：

* 水平不高的小偷被抓，为了生存，不断提到水平
* 为了抓高水平的校友，警察也需要不断提升自己

​![image](assets/image-20240827214007-1o09c2x.png)​

	根据上面的例子GAN（生成对抗网络）由两个重要的部分组成：

​![image](assets/image-20240827214225-gsv3atq.png)​

* 第一阶段：固定生成器，训练判别器（生成器所生成的东西需要骗过判别器，先提升判别器水平，才能更好的训练生成器）
* 第二阶段：训练生成器，固定判别器
* 循环一、二阶段

​![image](assets/image-20240827214507-1gvpbkz.png)​

​![image](assets/image-20240827214515-86vxzls.png)​

​![image](assets/image-20240827214554-nsivxox.png)​

### 2. GAN的用途

* 生成数据集

​![image](assets/image-20240827214710-ygb6zou.png)​

* 生成图像、漫画

​![image](assets/image-20240827214726-pthmtd7.png)​

* 生成指定风格的图像

​![image](assets/image-20240827214752-mqmtd1j.png)​

* 文字到图像的转换

​![image](assets/image-20240827214813-uezi71b.png)​

* 语义分割到图片的转换

​![image](assets/image-20240827214838-cgbqp5g.png)​

* 自动生成模特

​![image](assets/image-20240827214858-piq63mn.png)​

* 自动生成3D模型

​![image](assets/image-20240827215013-rfgea25.png)​

* ......

### 3. MVTec AD数据集

* 训练集只包含正常的样本
* 测试集包含正常样本与缺陷样本

	用于无监督的训练

​![image](assets/image-20240827215607-qbl51t8.png)​

‍
