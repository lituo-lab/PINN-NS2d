# 使用PINN进行流场高分辨率重建



## 问题概述

对于一个流动，可通过实验手段可获得一定时空分布的速度场，我们希望基于已有测量出的速度场以及NS方程进行速度场的高分辨率重建。此次案例以二维泰勒衰减涡为例，其公式如下：
$$
\begin{gathered}
u(x, y, t)=-\cos (x) \sin (y) e^{-2 t} \\
v(x, y, t)=\sin (x) \cos (y) e^{-2 t} \\
p(x, y, t)=-0.25(\cos (2 x)+\cos (2 y)) e^{-4 t}
\end{gathered}
$$
我们在时空中随机采样N个点，记录其时空坐标(x,y,t)和速度(u, v)，代表实验获得的速度点，基于此进行速度场高分辨率的重建，以及压力的恢复。

PINN的核心原理是基于已有数据和物理方程构建loss项，loss_data = loss(模型输出，真值)，loss_equ = 物理方程的残差。

对于流动问题，物理方程由NS方程决定。对于该二维流动问题，其残差公式为：
$$
\begin{aligned}
& e_1=u_t+u u_x+v u_y+p_x-\frac{1}{R e}\left(u_{x x}+u_{y y}\right) \\
& e_2=v_t+u v_x+v v_y+p_y-\frac{1}{R e}\left(v_{x x}+v_{y y}\right) \\
& e_4=u_x+v_y
\end{aligned}
$$
神经网络模型的输入为(x,y,t)，模型的输出为(u,v,p)。特别的，该案例由于涉及到输出对输出的二阶导u_xx, 模型的激活函数必须使用Swish或者tanh，不能出现ReLU或者PReLU等二阶倒数为0的激活函数。在训练过程中，需要用一定的系数平衡两类误差可以使得模型的收敛的更快。

由于NS方程中只涉及到压力p的导数项p_x和p_y,故恢复得到的压力场p具有和真实p的相同的分布，但数值上相差一个常量C，在真实实验中，需要考虑使用单点压力测量的手段实现常数C的校准。 



## 文件介绍

这是一个PINN高分辨率流场重建的程序。

文件包含一个output的数据集文件，以及三个py程序主体文件。

main.py：模型的训练文件，模型的data信息根据已有公式生成，真诚模型输入为(x,y,t)

predict.py：模型的预测函数，正向推理，直观展示真实uvp和模型预测的uvp对比。

output：模型在训练过程中，间隔一定epoch输出预测压力场云图。png2avi.py文件可以将图转化为一个视频文件，可以更直观的看到随着训练次数的增加，模型预测的压力场越来越接近于真实的压力场。

可以看到，网络较好的实现了速度场的高分辨率重建，基本完成了压力场的结构。



## 参考链接

https://github.com/maziarraissi/PINNs

https://zhuanlan.zhihu.com/p/363043437

https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/122934575