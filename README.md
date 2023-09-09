# Dual-Branch network for segmentation
  构建双分支(dual-branch)网络，利用Transformer和Convolution网络的优势，针对不同特征和语义背景下的云和云影的特征实施分割(segmentation)任务,进行不同程度的尝试，构建demo供更规范化的任务实施。
- ### 1.Some Introduction    
  随着深度学习在分割任务的广泛运用，针对特殊场景的分割技术也得到了蓬勃发展。云和云影的遥感图像的研究作为气象研究领域的重要组成部分，在气候变化和土地变化分类扮演了重要角色。本Demo在充分构思关于Transformer和Convolution构建的双分支神经网络的优越特性，提高不同尺度的云和云影的分割能力，并且进一步的理解编码和解码过程中，不同层级的特征和空间信息进行融合的过程，从而得到具有较好效果的分割网络模型。

- ### 2.Content
  datasetes---用于保存标签数据的显示颜色信息
  demo---存放了300张图片（均已经过裁剪）
  demo_img---是本次demo为了更好展示效果选择的4张图片
  picture_img---是本次demo对应的4张图片经过详细标注后的$准确Label$
  rs_aug---对图像进行处理和一些基本转换工具
  Dual_branch_Network.py---是基本的双分支网络框架


- ### 3.More Details 
#### 3.1网络主体框架
该网络中可以非常直观的发现基于双分支融合网络的遥感图像云和云影分割方法，通过Transformer分支和Convolution分支分别对遥感图像进行下采样提取多尺度特征，然后基于两分支不同的特性，使用双向引导模块使得两分支能够相互指导对方进行下采样，提高了不同尺度的语义信息和空间信息提取能力。同时，图中也重点的突出了解码阶段的作用，充分利用双分支提取到的特征进行上采样，逐步引导特征图的恢复，使得云和云影的定位更加准确、分割边界更加清晰，最终生成分割结果。
![整体框架结构](pic1_Demo%E6%95%B4%E4%BD%93%E6%A1%86%E6%9E%B6%E5%9B%BE.jpg)

#### 3.2数据集处理和编码-解码思路介绍
(1) 数据集的处理
从Landsat-8号卫星和Sentinel-2号卫星上获取高清遥感图像，并且对遥感图像进行裁剪。对完成裁剪的图像进行标注，并且使用Labelme对裁剪完成的图像进行人工掩膜标注，并将宝珠类型分为：云、云影和背景三类，并且划分出训练集和验证集。
(2) 编码-解码的思路分析
将图像输入到Transformer分支和卷积分支的网络模型中，通过多次下采样获得遥感图像的不同尺度的特征信息，为特征融合做准备。
在$编码$阶段，使用Transformer和卷积网络相互引导的双分支结构去提取不同层次的特征，融合全局特征和局部特征；
在$解码$阶段，使用Transformer分支和条状卷积分支提取到的不同层次的语义信息和空间信息进行上采样，融合高级语义信息和空间位置信息，实现云和云影的精准定位和精细分割。
并且其中利用双向引导模块引导Transformer分支和卷积分支进行特征提取。

#### 3.3抽象化模块的具象化表述
![各层抽象化理解](pic2_%E5%90%84%E5%B1%82%E7%9A%84%E6%8A%BD%E8%B1%A1%E5%8C%96%E7%90%86%E8%A7%A3.jpg)


(1)Transformer分支的表达式

$$d_i=\left\{\begin{array}{l}x_0, i=0 \\ x_i^t+x_i^{m c}, i=1,3 \\ x_i^t, i=2,4\end{array}\right.$$

其中，${{d}_{i}}$表示Transformer分支第$i$层的输入矩阵($i=0,1,2,3,4$),${{x}_{0}}$表示输入到模型的矩阵，$x_{i}^{t}$和$x_{i+1}^{t}$分别表示Transformer分支的第$i$层和第$i+1$层的输出矩阵，$x_{i}^{mc}$表示卷积分支的第$i$层输出经过多级池化后的特征图。

$${{T}_{1}}=Con{{v}_{embed}}({{d}_{i}})$$,
$Con{{v}_{embed}}(\centerdot )$表示卷积的嵌入层， 
	$${{T}_{2}}=MHA\{Flatten[Con{{v}_{proj}}({{T}_{1}})+{{d}_{i}}]\}$$

$Con{{v}_{proj}}(\centerdot )$表示卷积投影层，$Flatten(\centerdot )$表示将二维数据展开的一维数据，$MHA(\centerdot )$表示多头注意力层

$$x_{i+1}^t=\operatorname{Reshape}\left\{\operatorname{MLP}\left[\operatorname{Norm}\left(T_2\right)+d_i\right]\right\}$$

$MLP(\centerdot)$表示多层感知机，$\operatorname{Re}shape(\centerdot )$表示将一维数据变成二维数据

（2）Convolution分支的表达式

$$e_i=\left\{\begin{array}{l}x_0, i=0 \\ x_i^c, i=1,3,5 \\ x_i^c+x_i^{u t}, i=2,4\end{array}\right.$$
上式中，${{e}_{i}}$表示条状卷积第$i$层的输入($i=0,1,2,3,4$),${{x}_{0}}$表示输入的原始图像，$x_{i}^{c}$和$x_{i+1}^{c}$分别表示条状卷积分支的第$i$层和第$i+1$层的输出矩阵，$x_{i}^{ut}$表示Transformer分支的第$i$层经过双线性插值上采样成为相同大小的特征图。

$$C_1=\delta\left\{\operatorname{norm}\left[\operatorname{Conv}_{1 \times 3}\left(e_i\right)\right]\right\}$$

$$C_2=\delta\left\{\operatorname{norm}\left[\operatorname{Conv}_{3 \times 1}\left(C_1\right)\right]\right\}$$
$\delta (\centerdot )$表示激活函数ReLU，$Con{{v}_{1\times 3}}$和$Con{{v}_{3\times 1}}$分别表示卷积核大小和条状卷积
	$${{C}_{3}}=Maxpooling({{C}_{2}})$$
$Maxpooling(\centerdot )$表示最大池化层
	$$x_{i+1}^{c}={{C}_{2}}+{{C}_{3}}$$

（3）解码阶段
$${{D}_{i}}=Upsample\{\delta [DWConv({{M}_{I}})]\},i=1,2,3,4$$

$$M_i=\left\{\begin{array}{l}\operatorname{Concat}\left(D_{i+1}, x_i^c\right), i=1,3 \\ \operatorname{Concat}\left(D_{i+1}, x_i^t\right), i=2 \\ \operatorname{Concat}\left(x_i^t, x_{i+1}^c\right), i=4\end{array}\right.$$
	上式中，$x_{i}^{t}$和$x_{i}^{c}$分别表示Transformer分支和卷积分支第i层的输出，${{D}_{i}}$表示解码器第i层的输出，$Upsample(\centerdot )$表示双线性插值上采样，$\delta (\centerdot )$表示记过函数GELU，DWConv表示深度可分离卷积，$Concat(\centerdot )$表示拼接操作。

#### 3.4评价指标介绍
本文的部分的评价指标如下，用于衡量构建的深度神经网络的性能。MPA( Mean Pixel Accuracy)也叫像素精度，是图像分割任务中常用的评价指标之一，其计算方法是将模型预测正确的像素数除以总像素数:
$$\text{MPA}=\frac{1}{k}\sum\limits_{i=0}^{k}{\frac{{{p}_{i,i}}}{\sum\limits_{j=0}^{k}{{{p}_{i,j}}}}}$$
其中，正确预测的像素数是指模型预测的像素标签与真实像素标签完全一致的像素数，总像素数是指图像中所有像素的数量。
MIoU(Mean Intersection over Union),也叫平均交并比，是图像分割任务中常用的另一个评价指标，其计算方法是将每个类别的交并比取平均值
$$\mathrm{MIoU}=\frac{1}{k+1} \sum_{i=0}^k \frac{p_{i, i}}{\sum_{j=0}^k p_{i, j}+\sum_{j=0}^k p_{j, i}-p_{i, i}}$$
其中，类别交集是指模型预测的像素标签与真实像素标签都为某个特定类别的像素数，类别并集是指模型预测的像素标签或真实像素标签为某个特定类别的像素数。MIoU的值越大，表示模型的分割结果越准确。

#### 3.5一些结果
不同方法的对比结果
![Alt text](result1.jpg)

训练过程
![Alt text](result2.jpg)

MIOU
![Alt text](result3_MIOU-1.jpg)








  
