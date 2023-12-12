# GBDT笔记-理论推导合集

> 参考：
>
> - https://zhuanlan.zhihu.com/p/47185756
> - https://zhuanlan.zhihu.com/p/494536555
> - https://zhuanlan.zhihu.com/p/465921554
> - https://zhuanlan.zhihu.com/p/91652813
> - https://zhuanlan.zhihu.com/p/89614607

## Boosting 思想

Boosting模型可以抽象为一个前向加法模型（additive model）：$F(x;{\alpha_{m}, \theta_{m}})=\sum_{m=1}^{M}\alpha_{m}f(x;\theta_{m})$

其中， x 为输入样本，$f(x;\theta_{m})$ 为每个基学习器，$\theta_{m}$为每个基学习器的参数，treeBoost论文里面讨论的==基学习器都是CART回归树==。Boost是"提升"的意思，一般Boosting算法都是一个迭代的过程，每一次新的训练都是为了改进上一次的结果，这要求==每个基学习器的方差足够小（稳定），即足够简单(模型参数足够简单，不会导致过拟合）==（weak machine），因为Boosting的迭代过程足以让bias减小，但是不能减小方差。（Bagging的基分类器是偏差小方差大，而boosting的基分类器是偏差大方差小）

Boosting模型是通过最小化损失函数得到最优模型的，这是一个NP难问题，一般通过贪心法在每一步贪心地得到最优的基学习器。可以通过梯度下降的带每个最优基学习器的方向！

过去我们使用的梯度下降都是在参数空间的梯度下降，变量是参数，==而Boosting算法是在函数空间的梯度下降。通过不断梯度下降，我们得到很多个基学习器(函数)，将每个阶段得到的基学习器相加得到最终的学习器==。直观地说，**当前基学习器的训练需要知道所有学习器在每一个样本上的表现都差了多少**，然后当前基学习器努力地学习如何将每个样本上差了的部分给补充上来。回忆一下，之前在调模型参数的时候，我们通过梯度下降知道每个参数与下一个可能的loss最小的点都差了多少，根据这个信息去更新参数.

在梯度下降法中，我们可以看出，对于最终的最优解 $\theta^*$，是由初始值$\theta_{0}$经过T次迭代之后得到的，这里设$\theta_0=-\frac{\delta L(\theta)}{\delta \theta_0}$ ，则 $\theta^*$为：
$$
\theta^*=\sum_{t=0}^T \alpha_t *\left[-\frac{\delta L(\theta)}{\delta \theta}\right]_{\theta=\theta_{t-1}}=\theta_{T-1}+\alpha_T\left[-\frac{\delta L(\theta)}{\delta \theta}\right]_{\theta=\theta_{T-1}}
$$

其中， $\left[-\frac{\delta L(\theta)}{\delta \theta}\right]_{\theta=\theta_{t-1}}$ 表示 $\theta$ 在 $\theta-1$ 处泰勒展开式的一阶导数。

**在函数空间中**，我们也可以借鉴梯度下降的思想，进行最优函数的搜索。对于模型的损失函数 $L(y,F(x))$ ，为了能够求解出最优的函数 $F^*(x)$ ，首先设置初始值为： $F_0(x)=f_0(x)$

以函数 F(x)作为一个整体，与梯度下降法的更新过程一致，假设经过T次迭代得到最优的函数$F^*(x)$ 为：
$$
F^*(x)=\sum_{t=0}^T f_t(x)=F_{T-1}(x)+f_T(x)
$$

其中， $f_T(x)$ 为:
$$
f_t(x)=-\alpha_t g_T(x)=-\alpha_T *\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{T-1}(x)}
$$


![v2-3da39f3b5483f0b699bf72fc84e5c5d0_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-3da39f3b5483f0b699bf72fc84e5c5d0_1440w.webp)

**总结：Gradient Boosting算法在每一轮迭代中，首先计算出当前模型在所有样本上的负梯度，然后以该值为目标训练一个新的弱分类器进行拟合并计算出该弱分类器的权重，最终实现对模型的更新。**

注意：决策树是启发式算法，因为它通过递归地进行特征选择和判定，以达到最终的决策结果。在每个节点上，它选择最能区分数据的特征进行分割，通过对数据集的分层判断，最终形成一个树状结构。

> ### 函数空间
>
> **定义**：对于每一个特征x，都有一个函数数值F(x)与之对应，函数空间的指标就是这些函数数值F(x)，因此函数空间有无数个坐标，但是在训练过程，一般只有有限个训练样本（x），因此函数空间的维度也是有限的 ${F(x_{i})}^{N}_{1}$ 。
>
> 在函数空间里面做gradient descent，相当求出函数在每个样本函数值方向上的梯度，即：$g_{m}(x_{i})=-[\frac{\partial L(y_{i}, F(x_{i}))}{\partial F(x_{i})}]_{F(x)=F_{m-1}(x)}$；
>
> 但是，这个负梯度只是定义在有限的训练样本支撑起的函数空间里面，**如果泛化到其他的数据点呢， 这个基学习器在训练样本上的取值尽可能拟合负梯度**，以AdaBoost为例，负梯度为：
>
> $g_{m}(x_{i})=-[\frac{\partial L(y_{i}, F(x_{i}))}{\partial F(x_{i})}]_{F(x)=F_{m-1}(x)}=y_{i}e^{-y_{i}F_{m-1}(x)_{i}}$
>
> > 其中**$e^{-y_{i}F_{m-1}(x_{i})}$可以理解为样本的权重**，如果样本在之前几轮的分类都正确，即$y_{i} =F_{m-1}(x_{i})=1 (or -1)$,则$e^{-y_{i}F_{m-1}(x_{i})}$会比较小；反之，权重（$e^{-y_{i}F_{m-1}(x_{i})}$，上一轮迭代计算得到的系数）很大，本轮的基学习器需要在这个样本上得到较大的数值才能把这个样本从错误的泥潭拉回来。要是基学习器（$f_m(X)$）拟合好负梯度==（即$f_{m}(X) \bullet g_{m}(X)$)==，即要求基学习器在各个样本上的函数值(这里写成了向量X形式）与负梯度的点积最大，即：
> >
> > $max \sum_{i=1}^{N}f_m(x_{i})y_{i}e^{-y_{i}F_{m-1}(x_{i})}=min \sum_{i}^{N}I(h_{t}(x_{i}) \ne y_{i})e^{-y_{i}F_{m-1}(x_{i})}$
> >
> > <u>分析：</u>
> >
> > - <u>等号左边：因为要满足基学习器拟合负梯度，也就是让基分类器的结果去靠近负梯度，而左边的式子可以写成点积形式$f_{m}(X)  \bullet y^{T}e^{y^T \bullet F_{m-1}(X)},其中X=（x_{1},...,x_{n}), y=(y_{1}, ..., y_{n})$,此时好的拟合效果就是保证点积左右的向量最好是同向，也就是保证点积式子max</u>
> > - <u>而左边的式子其实可以分为$h_{t}(x_{i})=y_{i}$和$h_{t}(x_{i})\ne y_{i}$,要保证左边式子最大化，也就是最小化$h_{t}(x_{i})\ne y_{i}$（这个在$\sum_{i=1}^{N}h_{t}(x_{i})y_{i}e^{-y_{i}F_{m-1}(x_{i})}$占比最大）</u>
> >
> > 求出负梯度的目的是为**给下一个基学习器的学习指明方向**，下一个基学习器的学习目标就是拟合这些负梯度（重点关注负梯度比较大的样本）。如果loss在某个样本函数值上的负梯度大于0(则说明y=1)，也说明在下一个基学习器应该增加在这个样本上的函数值（[负梯度>0] => [Cost关于F(x)的梯度<0] => [满足$e^{-x}$单调性] => [$y_{i}F_{m-1}(x_{i})$向x正半轴移动（y=1）] 。
> >

> 对梯度提升算法的若干思考
>
> 1. **梯度提升与梯度下降的区别和联系是什么？**
>
> ![v2-88b9003102d66a6a1ed7bd71a40209dd_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-88b9003102d66a6a1ed7bd71a40209dd_1440w.webp)
>
> 可以发现，两者都是在每一轮迭代中，都是利用损失函数相对于模型的负梯度方向的信息来对当前模型进行更新，只不过在梯度下降中，模型是以参数化形式表示，从而模型的更新等价于参数的更新。而在梯度提升中，模型并不需要进行参数化表示，而是直接定义在函数空间中，从而大大扩展了可以使用的模型种类。
>
> 2. **梯度提升和提升树算法的区别和联系？**
>
> 当损失函数是平方误差损失函数和指数损失函数时，每一步优化是很简单的。但对一般损失函数而言，往往每一步优化并不那么容易。针对这一问题，Freidman提出了梯度提升（gradient boosting）算法。这是利用损失函数的负梯度在当前模型的值$-\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{t-1}(x)}$  作为提升树算法中残差的近似值，拟合一个梯度提升模型。
>
> 3. **对于一般损失函数而言，为什么可以利用损失函数的负梯度在当前模型的值作为梯度提升算法中残差的近似值呢？**
>
> 我们观察到在提升树算法中，残差 $y-F(x)$ 是损失函数 $\frac{1}{2}(y-F(x))^2$的负梯度方向，因此可以将其推广到其他不是平方误差(分类或是排序问题)的损失函数。也就是说，梯度提升算法是一种梯度下降算法，不同之处在于更改损失函数和求其负梯度就能将其推广。即:可以将结论推广为对于一般损失函数也可以利用损失函数的负梯度近似拟合残差。

## 损失函数

主要是要采用一种贪心的策略，找到使得loss最小的基学习器作为当前的基学习器：

loss function:  $argmin_{\alpha, \theta}=\sum_{i=1}^{N}L(y_{i}, F_{m-1}(x_{i})+\alpha f_{m}(x_{i}; \theta))$

*其中：$\theta$ 是第m轮的基学习器的参数， $\alpha$ 是学习的步长*

> ### 探究loss function 与拟合负梯度的联系(从三个方面）：
>
> - ==理解层面==：负梯度的方向是loss下降的最快的方向，因此拟合好负梯度其实等价于让loss下降的最多
>
> - ==一阶泰勒展开==
>
>   ~~因为boosting的目标是让$L(y_{i}, F_{m}(x_{i})) \le L(y_{i}, F_{m-1}(x_{i})), 而L(y_{i}, F_{m}(x_{i}))=L(y_{i}, F_{m-1}(x_{i})+\alpha f_{m}(x_{i}; \theta))$,所以$L(y_{i}, F_{m}(x_{i})) - L(y_{i}, F_{m-1}(x_{i})) = L(y_{i}, \alpha f_{m}(x_{i}; \theta))$（满足L（a, b+c) = L(a, b) + L(a, c)）；~~（因为对于交叉熵损失函数无法满足L（a, b+c) = L(a, b) + L(a, c)）；
>
>   ![v2-cf3b91c12a96989f758b609aa34ae472_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-cf3b91c12a96989f758b609aa34ae472_1440w.webp)
>   
>   因为
>   $$
>   \sum_{i=1}^{N}L(y_{i}, F_{m-1}(x_{i})+\alpha f_{m}(x_{i}; \theta)) \approx\sum_{i=1}^{N}L(y_{i}, F_{m-1}(x_{i}))+\sum_{i=1}^{N} \frac{\partial L(y, F(x))}{\partial F(x)}|_{F(x)=F_{m-1}(x_{i})}\alpha f_{m}(x_{i}; \theta)\\
>   \sum_{i=1}^{N}L(y_{i}, F_{m-1}(x_{i})+\alpha f_{m}(x_{i}; \theta)) - \sum_{i=1}^{N}L(y_{i}, F_{m-1}(x_{i})) \approx\sum_{i=1}^{N} \frac{\partial L(y_{i}, F(x_{i}))}{\partial F(x_{i})}|_{F(x_{i})=F_{m-1}(x_{i})}\alpha f_{m}(x_{i}; \theta)\\
>   (不等号右边可以看成两个向量的内积： \frac{\partial L(y, F(X))}{\partial F(X)}和\alpha f_{m}(X),其中X=(x_{1}, x_{2},...,x_{n})^{T}, y=(y_{1}, y_{2},...,y_{n})^{T}\\
>   当\alpha f_{m}(x_{i}; \theta) = - \sum_{i=1}^{N} \frac{\partial L(y_{i}, F(x))}{\partial F(x)}|_{F(x)=F_{m-1}(x_{i})}（梯度的负方向是局部下降最快的方向）时，\\
>   \sum_{i=1}^{N}L(y_{i}, F_{m-1}(x_{i})+\alpha f_{m}(x_{i}; \theta)) - \sum_{i=1}^{N}L(y_{i}, F_{m-1}(x_{i})) \le 0
>   $$
>   在GBDT中，会令$r_{x_{i}, y_{i}}=r_{mi}=-\frac{\partial L(y_{i}, F(x))}{\partial F(x)}|_{F(x)=F_{m-1}(x_{i})}$

> 总结GBDT的整体流程：
>
> 1. 计算当前损失函数的负梯度表达式（训练样本的权值或概率分布）；
> 2. 构造训练样本将$(x_{i}, y_{i})$带入$r_{m}(x, y)$可得到$r_{mi}$,进而得到第m轮的训练样本为$T_{m}={(x_{1},r_{m1}), ...,(x_{N}, r_{mN})}$
> 3. 让当前基学习器去拟合上述训练样本得到$\alpha f_{m}(x_{i}, \theta)$

![v2-b79bb17320e07c3c907aab843af06f8e_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-b79bb17320e07c3c907aab843af06f8e_1440w.webp)

伪代码：

![v2-2ff9038d2b798c3d36dc8e8d5d41ec4d_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-2ff9038d2b798c3d36dc8e8d5d41ec4d_1440w.webp)

注：

- 首先初始化第0个基学习器 $F_{0}(x)=argmin_{c}\sum_{i=1}^{N}L(y_{i},c)$，**即对于每一个x，第0个基学习器是使得当前loss最小的一个数值，第4行计算当前基学习器参数的时候，用了平方误差衡量基学习器与负梯度的拟合程度，实际应该根据问题选择相应的loss**

- 探究算法的第1步$F_0(\mathbf{x})=\arg \min _\rho \sum_{i=1}^N L\left(y_i, \rho\right)$

  - 当损失函数为均方误差时，$F_0(x)=\bar{y}(样本真实值的平均值)$

  - 当损失函数为MAE时，$F_0(x)=median_{y}(样本真实值的中位数)$

    > $MAE = \frac{1}{m} \sum_{i=1}^m \mid c-y_i \mid$,$gradient\ of MAE = \frac{1}{m} \sum_{i=1}^msign(c-y_i)=0$,也就是说，对于所有的样本真实值$\left \{y_i\right \}_1^N$,要满足$c>y_i和c<y_i$一样多，只能是c为中位数。（如果样本x在样本中的label取值为2, 4, 6，9，那么在 MAE 下训练的模型预估值应该是中位数5）

  - 当损失函数为logistic loss时， $\mathrm{F}_0(\mathrm{x})=\left(\frac{1}{2}\right) * \log \left(\frac{\sum \mathrm{y}_{\mathrm{i}}}{\sum\left(1-\mathrm{y}_{\mathrm{i}}\right)}\right)$

    
    
    > ![微信图片_20231127220317](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231127220317.png)
  
  
  
  - 当损失函数为指数函数$\mathrm{L}\left(\mathrm{y}_{\mathrm{i}}, \mathrm{F}\left(\mathrm{x}_{\mathrm{i}}\right)\right)=\mathrm{e}^{-\mathrm{y_i}F\left(\mathrm{x}_{\mathrm{i}}\right)},F_0(x)=\frac{1}{2}log\frac{\sum_{i=1}^NI(y_i=1)}{\sum_{i=1}^NI(y_i=-1)}$

    > ![微信图片_20231121225258](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231121225258.jpg)

- 	探究算法的第3步$\tilde{\mathrm{y}}_{\mathrm{i}}=-\left[\frac{\partial \mathrm{L}\left(\mathrm{y}_{\mathrm{i}}, \mathrm{F}\left(\mathrm{x}_{\mathrm{i}}\right)\right)}{\partial \mathrm{F}\left(\mathrm{x}_{\mathrm{i}}\right)}\right]_{\mathrm{F}(\mathrm{x})=\mathrm{F}_{\mathrm{m}-1}(\mathrm{x})}$

  - 当损失函数采用$L(y_i, F(x_i))=\frac{1}{2}(y_i-F(x_i))^2$,那么负梯度为$y_i-F(x_i)$,将$F(x_i)=F_{m-1}(x_i)$带入结果得到$y_i-F_{m-1}(x_i)$

  - 当损失函数采用$L(y_i, F(x_i))=|y_i-F(x_i)|$,那么其负梯度为

![微信图片_20231121225818](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231121225818.jpg)

- 当损失函数次啊用logistic loss时（二分类任务）$\mathrm{L}\left(\mathrm{y}_{\mathrm{i}}, \mathrm{F}\left(\mathrm{x}_{\mathrm{i}}\right)\right)=\mathrm{y}_{\mathrm{i}} \log \left(\mathrm{p}_{\mathrm{i}}\right)+\left(1-\mathrm{y}_{\mathrm{i}}\right) \log \left(1-\mathrm{p}_{\mathrm{i}}\right)$ 

  则负梯度为

  ![微信图片_20231127220821](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231127220821.jpg)
  
  

![image-20231120135648548](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/image-20231120135648548.png)

即：

- **损失函数$L(x,y)$ ：用以衡量模型预测结果与真实结果的差异**

- **弱评估器$f(x)$ ：（一般为）决策树，不同的boosting算法使用不同的建树过程**
- **综合集成结果$H(x)$：即集成算法具体如何输出集成结果**

> ​	==二阶泰勒展开==
>
> ![微信图片_20231120140148](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231120140148.jpg)
>
> 注：在分析$f_{m}(x_{i})$时可以将$\alpha_{m}$视为常数，也是将$\sqrt{\alpha_{m}}h_{m}(x_{i})$合并成一个新的constant，同时将$\alpha_{m}^{2}h_{m}(x_{i})$也视为新的constant

## 回归树

采用树模型作为GBDT基学习器的**优点是：**

1. 可解释性强； 

2. 可处理混合类型特征（既有连续型特征也有类别型特征） ；

3. 具体伸缩不变性（不用归一化特征）；

4. 有特征组合的作用；

   ![image-20231120153743186](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/image-20231120153743186.png)

5. 可自然地处理缺失值；

   ​	**基于特征的划分：** 决策树的构建是基于特征的划分的，每次选择一个特征来划分数据集。当在某个节点遇到缺失值时，树模型会考虑所有其他可用的特征进行划分。这种基于特征的划分使得树模型在训练和预测时能够灵活地处理缺失值。

   ​	**不依赖线性关系：** 决策树不依赖线性关系，而是通过树状结构递归地划分数据。对于每个节点，决策树都会选择最佳的划分方式，而当前节点的缺失值只会影响到当前节点的数据流向，并不会影响到其他节点的建立。（决策树能够根据数据的特征划分来构建决策树。这意味着，即使输入数据具有复杂的非线性关系，决策树也能够通过分层决策来模拟这种关系，从而实现对数据的分类。因为能处理分类变量和连续型变量，适合*处理非线性*的*关系*）

   > **详细解释决策树可以处理非线性关系**
   >
   > 1. **分层划分：** 决策树的核心思想是通过递归地选择最佳的特征和划分点，将数据集划分成不同的子集。**每个划分代表了对输入空间的一个非线性划分**。这种分层的划分方式使得决策树能够适应复杂的、非线性的数据结构。
   > 2. **多特征组合：** 在决策树的每个节点上，模型可以同时考虑多个特征的组合**(x1<5且x7<8，特征中除了连续变量还有离散变量)**来进行划分。这允许决策树捕捉到输入特征之间的复杂交互关系，这些交互关系可以构成非线性关系。
   > 3. **递归特征选择：** 决策树通过递归地选择最佳的特征和划分点，每次都在当前节点中选择对目标变量影响最大的特征**（适应数据的复杂结构）**。这种逐步的特征选择过程允许决策树在每个节点上根据目标变量的非线性关系进行选择，从而逐渐构建非线性模型。
   > 4. **不依赖线性关系：** 决策树不对特征之间的关系做出线性假设**(由于决策树的贪心特征，使得不需要对特征之间进行多重共线性检验）**。每个节点的划分依赖于数据在当前节点上的分布，而不受到线性关系的限制。这使得决策树能够更灵活地适应各种非线性模式。
   > 5. **非参数化结构：** 决策树是一种非参数化模型，不对数据的分布形式进行假设。这意味着决策树对于复杂的、非线性的数据结构没有先验假设，可以根据数据的实际情况进行拟合。

6. 对异常点（异常值只占数据集的一小部分）鲁棒；（由于异常值的存在不太可能导致整个分割过程的变化，再基于多数投票原则，树模型在选择分割点时相对稳健。）

7. 有特征选择作用；（基尼系数或者entropy）

8. 可扩展性强；（可以处理不同规模的数据集；处理高维特征；处理非线性关系）

9. 容易并行。（随机森林就是基于这个建立的）

**缺点是：**

1、 缺乏平滑性（回归预测时输出值只能输出有限的若干种数值，对于连续特征的划分一般是x<c, x>=c，对多棵树的输出结果【同一棵树对于新的样本每次判断结果都一样（叶子节点的均值），则最后这棵树的输出结果也是一样的】进行加权求和）

2、不适合处理高维稀疏数据（找到有效的分割点变得更加困难；当样本量相对于特征数较少时，树模型在高维稀疏数据上容易捕捉到噪声而不是信号）。



CART决策树(用于分类任务时，树的分裂准则采用基尼指数，用于回归任务时，用MSE、MAE和Friedman_mse【改进型的mse】)实际上是将空间用超平面进行划分的一种方法，每次分割的时候，都将当前的空间一分为二， 这样使得每一个叶子节点都是在空间中的一个不相交的区域，在进行决策的时候，会根据输入样本每一维feature的值，一步一步往下，最后使得样本落入N个区域中的一个（假设有N个叶子节点）。

回归树的参数主要有两个，**一个叶子节点对应的区域（如何划分特征）**，**另一个是叶子节点的取值**（图中的The Height in each segment），其intuition就是使得划到特征空间每个区域的样本的方差足够小，足以用一个数值来代表，因此每个节点的分裂标准是划分到左右子树之后能不能是样本方差变小，**即回归树使用方差衡量样本“纯度”（波动性），而分类树采用Gini系数或者熵来衡量样本纯不纯**

回归树的例子如下图所示：

![v2-50b6dc72d0d4086116832f3d252ae219_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-50b6dc72d0d4086116832f3d252ae219_1440w.webp)

![v2-87044dd8eec19cf71a0d7076d0e05728_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-87044dd8eec19cf71a0d7076d0e05728_1440w.webp)

注意：timeline从左到右分别是2010/03/20，2011/03/01；my rate over love songs(对歌曲的喜爱程度)从下到上分别是0.2， 1.0， 1.2

回归树的表示方式：$f(x;\left \{b_{j}, R_{j} \right \})=\sum_{j=1}^{J}b_jI(x \in R_{j})(J表示叶子节点个数)$

则对应于加法模型的GBDT的m轮迭代的结果是：

$F_{m}(x)=F_{m-1}(x)+\alpha_{m} \sum_{j=1}^{J}b_{jm}I(x \in R_{jm})=\sum_{k=1}^MT(x;\theta_k)$

注意：加号后面的式子对应于上面的$f(x;\left \{b_{j}, R_{j} \right \})$,因此可令$\gamma_{jm}=\alpha_{m}b_{jm },则GBDT的m轮迭代结果可以写成$$F_{m}(x)=F_{m-1}(x)+\sum_{j=1}^{J}\gamma_{jm}I(x \in R_{jm})$(此时的参数只有一个：$\gamma_{jm}$,原本只有两个参数：$\alpha_{m}和b_{jm}(I(x\in R_{jm}由b_{jm}来决定))$

**gbdt 无论用于分类还是回归一直都是使用的CART 回归树。**这是因为gbdt 每轮的训练是在上一轮的训练的残差基础之上进行训练的。这里的残差就是当前模型的负梯度值 。这个要求每轮迭代的时候，弱分类器的输出的结果相减是有意义的。**GBDT的核心就在于，每一棵树学的是之前所有树结论和的残差**，这个**残差就是一个加预测值后能得真实值的累加量**。比如A的真实年龄是18岁，但第一棵树的预测年龄是12岁，差了6岁，即残差为6岁。那么在第二棵树里我们把A的年龄设为6岁去学习，如果第二棵树真的能把A分到6岁的叶子节点，那累加两棵树的结论就是A的真实年龄。

## GBDT用于回归

- 对于回归问题，损失函数一般采用均方误差损失函数，即$Cost = \sum_{i=1}^N\left[y_i-F_{m-1}\left(x_i\right)-\sum_{j=1}^J \gamma_{jm} I\left(x_i \in R_{j m}\right)\right]^2$,因为$r_{mi}=y_i - F_{m-1}(x_{i})$,所以$Cost = \sum_{i=1}^N\left[r_{mi}-\sum_{j=1}^J \gamma_{jm} I\left(x_i \in R_{j m}\right)\right]^2(变量只有\gamma_{jm})$，

$\gamma_{jm} =\operatorname{argmin}_{\gamma_m} \sum_{i=1}^N\left[r_{mi}-\sum_{j=1}^J \gamma_{j m} 1\left(x_i \in R_{j m}\right)\right]^2,其中r_{mi}=-\frac{\partial L(y_{i}, F(x))}{\partial F(x)}|_{F(x)=F_{m-1}(x_{i})}，其中L(y_{i}, F(x))=\frac{1}{2}(y_i - F(x))^2$

![微信图片_20231121154052](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231121154052.jpg)

结论：$\gamma_{jm}=$ average $_{x_i \in R_{j m}} r_{mi}$

> 例子
>
> > 训练集：（A, 14岁）、（B，16岁）、（C, 24岁）、（D, 26岁）；
> > 训练数据的均值：20岁； （这个很重要，因为GBDT与i开始需要设置预测的均值，即第0个基学习器，这样后面才会有残差！）
> > 决策树的个数：2棵；
> > 开始GBDT学习了~
> > 首先，输入初值20岁，根据第一个特征（具体选择哪些特征可以根据信息增益来计算选择），可以把4个样本分成两类，一类是购物金额<=1K，一类是>=1K的。假如这个时候我们就停止了第一棵树的学习，这时我们就可以统计一下每个叶子中包含哪些样本，这些样本的均值是多少，因为这个时候的均值就要作为所有被分到这个叶子的样本的预测值了。比如AB被分到左叶子，CD被分到右叶子，那么预测的结果就是：AB都是15岁，CD都是25岁。和他们的实际值一看，结果发现出现的残差，ABCD的残差分别是-1, 1, -1, 1。这个残差，我们要作为后面第二棵决策树的学习样本。
> >
> > ![v2-19d611fd99d1596e7e76c5d46b278f79_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-19d611fd99d1596e7e76c5d46b278f79_1440w.webp)
> >
> > 然后学习第二棵决策树，我们把第一棵的残差样本（A, -1岁）、（B，1岁）、（C, -1岁）、（D, 1岁）输入。此时我们选择的特征是经常去百度提问还是回答。这个时候我们又可以得到两部分，一部分是AC组成了左叶子，另一部分是BD组成的右叶子。那么，经过计算可知左叶子均值为-1，右叶子均值为1. 那么第二棵数的预测结果就是AC都是-1，BD都是1. 我们再来计算一下此时的残差，发现ABCD的残差都是0！停止学习~
> >
> > ![v2-6bfe4194d60370c41a894006bde78d15_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-6bfe4194d60370c41a894006bde78d15_1440w.webp)
> >
> > 这样，我们的两棵决策树就都学习好了。进入测试环节：
> > 测试样本：请预测一个购物金额为3k，经常去百度问淘宝相关问题的女生的年龄~
> > 我们提取2个特征：购物金额3k，经常去百度上面问问题；
> > 第一棵树 —> 购物金额大于1k —> 右叶子，初步说明这个女生25岁
> > 第二棵树 —> 经常去百度提问 —> 左叶子，说明这个女生的残差为-1；
> > 叠加前面每棵树得到的结果：25-1=24岁，最终预测结果为24岁

- 当损失函数为MAE时，则$\gamma_{j m}=median_{x_i \in R_{jm}}r_{mi}$ 为梯度值。

> ![微信图片_20231121233922](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231121233922.jpg)

例子：

|  $x_i$  |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |  10  |
| :-----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| $y_{i}$ | 5.56 | 5.7  | 5.91 | 6.4  | 6.8  | 7.05 | 8.9  | 8.7  |  9.  | 9.05 |

前提：

- 选择MSE做为建树的分裂准则
- 选择MSE作为误差函数
- 树的深度设置为1

算法：

1. 我们需要初始化$F_0(x)$，因此$F_0(x)=\frac{1}{10}\sum_{i=1}^{10}r_{mi}=7.307$;

2. 拟合第一棵树,
$$ r_{mi}=-\left[\frac{\partial L(y_i,F(\mathbf{x}i))}{\partial F(\mathbf{x}i)}\right]_{F(x)=F_0(x)}=(y_i-F_0(x_i))$$,
用$\left \{x_i, r_{0i} \right \}_1^{10}$来建树



   | $x_i$|1|2|3|4|5|6|7|8|9|10|
   | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
   | $r_{0i}$ | -1.747| -1.607 |-1.397| -0.907|-0.507|-0.257 |1.593 |1.393 |1.693 | 1.743 |

   > 分析:
   >
   > $当选择x_i \le 1和x_i >1作为分割点是， mean_{left}=-1.747, MSE_{left}=[-1.747-(-1.737)]^2=0;\\mean_{right}=\frac{1}{9}\sum_{i=2}^{10}x_i\approx 0.19411,MSE_{right} = \frac{1}{9}\sum_{i=2}^{10}(x_i-mean)^2\approx1.747;$
   >
   > $当选择x_i \le 2和x_i >2作为分割点是， mean_{left}=\frac{1}{2}\sum_{i=1}^{2}x_i=-1.677, MSE_{left}=\frac{1}{2}\sum_{i=1}^{2}(x_i-mean_{left})^2=0.419;\\mean_{right}=\frac{1}{8}\sum_{i=3}^{10}x_i\approx 0.19411,MSE_{right} = \frac{1}{8}\sum_{i=2}^{10}(x_i-mean)^2\approx1.509;$
   >
   > 依次，穷尽所有取值。
   > 可以得到当选择6 66作为分裂点时$MSE_{sum}=0.3276$最小。
   >
   > <img src="https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/9c3b6b919a681ab57415143130ca3d1b.png" alt="9c3b6b919a681ab57415143130ca3d1b" style="zoom: 80%;" />
   >
   > 至此，我们完成了第一颗树的拟合，拟合完之后我们得到了$R_{jm}$以及$\gamma{jm}$:
   >
   > - $R_{11}:x_i \le 6;R_{21}:x_{i}> 6$
   >
   > - 
   >
   > - $$
   >   \begin{aligned}
   >   & \gamma_{11}=\frac{r_{01}+r_{02}+r_{03}+r_{04}+r_{05}+r_{06}}{6}=-1.0703 \\
   >   & \gamma_{21}=\frac{r_{07}+r_{08}+r_{09}+r_{0,10}}{4}=1.6055
   >   \end{aligned}
   >   $$
   >
   >   最后更新 $\mathrm{F}_1\left(\mathrm{x}_{\mathrm{i}}\right)$ 值， $\mathrm{F}_1\left(\mathrm{x}_{\mathrm{i}}\right)=\mathrm{F}_0\left(\mathrm{x}_{\mathrm{i}}\right)+\sum_{\mathrm{j}=1}^2 \gamma_{\mathrm{j} 1} \mathrm{I}\left(\mathrm{x}_{\mathrm{i}} \in \mathrm{R}_{\mathrm{j} 1}\right)$ 。
   >   比如更新其中一个样本 $x_1$ 的值:
   >   $$
   >   \mathrm{F}_1\left(\mathrm{x}_1\right)=\mathrm{F}_0\left(\mathrm{x}_1\right)+\eta * \sum_{\mathrm{j}=1}^2 \gamma_{\mathrm{j} 1} \mathrm{I}\left(\mathrm{x}_1 \in \mathrm{R}_{\mathrm{j} 1}\right)\\
   >   =7.307-0.1 * 1.0703=7.19997 (\eta表示learning\ rate=0.1)
   >   $$
3. 拟合第二颗树（m = 2)

$ \gamma_{11}=-\left[\frac{\partial L(y_i,F(\mathbf{x}i))}{\partial F(\mathbf{x}i)}\right]_{{F(x)=F_{m-1}(x)}}=(y_1-F_{1}(x_1))=(5.56-7.19997)=-1.63997$

| $x_i$ |1|2|3|4|5|6|7|8|9|10|
| - | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| $r_{1i}$ |-1.63997|-1.49997 |-1.28997 |-0.79997|-0.39997| -0.14997 | 1.43245 | 1.23245 | 1.53245 | 1.58245 |

得到两个叶子节点的值：$\gamma_{12}=−0.9633, \gamma_{22}=1.44495$

![79a56a304e2e3aa7c124450269ee0276](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/79a56a304e2e3aa7c124450269ee0276.png)

​	伪代码

![v2-3a687cddb561a00345412474793b5567_b](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-3a687cddb561a00345412474793b5567_b.webp)

>  注意：重点关注2-c的从，可以理解为训练集训练完决策树模型叶子节点的均值（每一棵树只有一个输出值，根据测试样本进入决策树过后判定的递归顺序得到的结果）

## GBDT分类树

>  GBDT用于分类的时候，并不是(像随机森林）用Gini或者熵的方式划分特征空间实现分类，由于需要拟合残差，GBDT实际上是在学习样本在每个类别上的得分

### 二分类

>  ==情况1==（$y \in \left \{-1，1\right \}$）
>
>  在论文【Friedman J H. Greedy function approximation: a gradient boosting machine[J]. Annals of statistics, 2001: 1189-1232.】中二分类$y \in \left \{-1，1\right \}$,则损失函数的表达式为$L(y, F)=\log (1+\exp (-2 y F))(系数2是作者自己加上去的)$

> **对数损失函数**
>
> 1. 对数损失函数的标准形式：$L(Y, P(Y=y \mid x))=-\log P(Y=y \mid x)$
> 2. 逻辑回归$P(Y=y \mid x)$表达式：
>    1. 当y=1时：$P(Y=y \mid x)=h_\theta(x)=g(f(x))=\frac{1}{1+e^{-f(x)}}$
>    2. 当y=-1时：$P(Y=y \mid x)=1-h_\theta(x)=1-g(f(x))=1-\frac{1}{1+e^{-f(x)}}=\frac{e^{-f(x)}}{1+e^{-f(x)}}=\frac{1}{1+e^{f(x)}}$
>
> 3. 将它带入到对数损失函数的标准形式，通过推导可以得到logistic的损失函数表达式如下：
>
>    $L(Y, P(Y=y \mid x))= \begin{cases}-\log \left(\frac{1}{1+e^{-f(x)}}\right) & y=1 \\ -\log \left(\frac{1}{1+e^{f(x)}}\right) & y=-1\end{cases}$
>
>    等价于
>
>    $L(Y, P(Y=y \mid x))= \begin{cases}\log (1+e^{-f(x)}) & y=1 \\ \log  (1+e^{f(x)}) & y=-1\end{cases}$
>
> 4. 当分的两类为 {1,−1} 时，逻辑回归的表达式$P(Y=y \mid x)$可以合并如下
>
>    $P(Y=y \mid x)=\frac{1}{1+e^{-y f(x)}}$
>
>    将它带入到对数损失函数的标准形式，通过推导可以得到logistic的损失函数表达式如下：
>
>    $L(Y, P(Y=y \mid x))=\log \left(1+e^{-y f(x)}\right)$
>
> 5. 若 $y \in Y=\{0,1\}$ ，则逻辑回归最后得到的目标式子如下:
>
> $$
> J(\theta)=-\frac{1}{m}\sum_{i=1}^m\left[ y^{(i)} \log h_\theta\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_\theta\left(x^{(i)}\right)\right)\right]
> $$
>
> 

残差可以表示为$
r_{mi}=-\left.\frac{\partial L\left(y_i, F\left(x_i\right)\right)}{\partial F\left(x_i\right)}\right|_{F\left(x_i\right)=F_{m-1}\left(x_i\right)}=\frac{2 y_i}{1+\exp \left(2 y_i F_{m-1}\left(x_i\right)\right)}$

> 推导:
>
> $\begin{aligned} & \because L(y, F)=\log (1+\exp (-2 y F)) \\ & \therefore \frac{\partial L\left(y_i, F\left(x_i\right)\right.}{\partial F\left(x_i\right)}=\frac{\exp \left(-2 y_i F\left(x_i\right)\right)\left(-2 y_i\right)}{1+\exp \left(-2 y_i F\left(x_i\right)\right)}=\left(-2 y_i\right)\left[1-\frac{1}{1+\exp \left(-2 y_i F(x_i)\right)}\right] \\ & =-2 y_i+2 y_i \cdot \frac{1}{1+\exp \left(-2 y_i F\left(x_i\right))\right.} \\ & \frac{\partial^2 L\left(y_i, F\left(x_i\right)\right)}{\partial F^2\left(x_i\right)}=2 y_i \frac{1}{1+\exp \left(-2 y_i F\left(x_i\right)\right)} \cdot \frac{\exp \left(-2 y_i F\left(x_i\right)\right)}{1+\exp \left(-2 y_i F\left(x_i\right)\right)} \cdot\left(2 y_i\right) (注意：把2y_iF(x_i)整体看成\frac{1}{1+e^{-x}}的x) \\ & \text { 而 } r_{m i}=-\left.\frac{\partial L\left(y_i, F\left(x_i\right)\right)}{\partial F_{\left(x_i\right)}}\right|_{F_{\left(x_i\right)}= F_{m-1}\left(x_i\right)}=2 y_i\left[\frac{1}{1+\exp \left(-2 y_i F_{m-1}\left(x_i\right)\right.)}-1\right] \\ & =(2 y_i) \frac{\exp \left(-2 y_i F_{m-1}\left(x_i\right)\right)}{1+\exp \left(-2 y_i F_{m-1}{\left(x_i\right)}\right)}=(2 y_i) \frac{1}{\left.\exp \left(2 y_i F_{m-1}{(} x_i\right)\right)+1} \\ & \end{aligned}$
>
> ![微信图片_20231122231956](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231122231956.jpg)
>
> $而\gamma_{l m}=\arg \min _\rho \sum_{i=1}^N \log \left(1+\exp \left(-2 y_i\left(F_{m-1}\left(\mathbf{x}_i\right)+\gamma_{lm}I(x_i\in R_{jm}\right)\right)\right)\\ (拆分到各个叶子节点）=\arg \min _\gamma \sum_{\mathbf{x}_i \in R_{j m}} \log \left(1+\exp \left(-2 y_i\left(F_{m-1}\left(\mathbf{x}_i\right)+\gamma_{lm}I(x_i\in R_{jm}\right)\right)\right)$，$\gamma_{lm}^*=\gamma_{l m} I(x_i \in R_{jm})=\arg \min _\gamma \sum_{\mathbf{x}_i \in R_{l m}} \log \left(1+\exp \left(-2 y_i\left(F_{m-1}\left(\mathbf{x}_i\right)+\gamma\right)\right)\right)\\ =\operatorname{argmin}_\gamma \sum_{x_i \in R_{j m}} L(y_i, F_{m-1}(x_i)+\gamma)(注：在F_{m-1}(x_i)处泰勒展开)\\=argmin\sum_{x_i \in R_{j m}} L\left(y_i, F_{m-1}\left(x_i\right)\right)+\partial L\left(y_i, F_{m-1}\left(x_i\right)\right) \gamma+\frac{1}{2} \partial^2 L\left(y_i, F_{m-1}\left(x_i\right)\right) \gamma^2\\ $
>
> $\gamma_{lm}^* =【注意：-\frac{b}{2a}】-\frac{\sum_{x_i \in R_{j m}} \partial L\left(y_i, F_{m-1}(x_i)\right)}{\sum_{x_i \in R_{j m}} \partial^2 L\left(y_i, F_{m-1}(x_i)\right)}=\frac{\sum_{x_i \in R_{j m}} r_{mi}}{\sum_{x_i \in R_{j m}}\left|r_{mi}\right|\left|2-r_{mi}\right|} $(证明过程见上面图片）

伪代码

注：$F_0(x)=\frac{1}{2}\frac{P(Y=1|X)}{P(Y=-1|X)}(论文中对\bar{y}未详细解释)$

![v2-42ab959ebccef1e34815c922a3848013_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-42ab959ebccef1e34815c922a3848013_1440w.webp)

==情况2==（$y \in \left \{0，1\right \}$）

> 单个样本的损失函数$J(\theta)= y_i \log h_\theta\left(x_i\right)+\left(1-y_i\right) \log \left(1-h_\theta\left(x_i\right)\right)$

对于GBDT二分类来说，其单个样本的损失函数为$L\left(y_i, F\left(x_i\right)\right)=-[y_ilog(\frac{1}{1+e^{-F(x_i)}})+(1-y_i)log(1-\frac{1}{1+e^{-F(x_i)}})]\\ =y_ilog(1+e^{-F(x_i)})-(1-y_i)log(\frac{e^{-F(x_i)}}{1+e^{-F(x_i)}})\\=y_i \log \left(1+e^{-F\left(x_i\right)}\right)+\left(1-y_i\right)\left[F\left(x_i\right)+\log \left(1+e^{-F\left(x_i\right)}\right)\right]$

- step1: 初始化第一个弱分类器$F_0(x)=log(\frac{P(Y=1|x)}{1-P(Y=1|x)})$

> 证明：
>
> ![微信图片_20231123003859](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231123003859.jpg)
>
> 其中， $P(Y=1|x)$是训练样本中 y=1 的比例，利用先验信息来初始化学习器。

- step2:建立M棵分类回归树

  - 对于建立第m棵树时，对于样本i=1,2,...,N, 计算第m棵树对应的负梯度$r_{m i}=-\left[\frac{\partial L\left(y_i, F\left(x_i\right)\right)}{\partial F(x)}\right]_{F(x)=F_{m-1}(x)}=y_i-\frac{1}{1+e^{-F\left(x_i\right)}}$

  > 证明：
  >
  > ![微信图片_20231123005408](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231123005408.jpg)

  - 对于i=1,2,…,N ，利用CART回归树拟合数据 $(x_i, r_{m,i})$ ，得到第 m 棵回归树，其对应的叶子节点区域为$R_{m,j}$,其中 j=1,2,…,$J_m$ ，且 $J_m$ 为第m棵回归树叶子节点的个数。
  
  - 对于$J_m$ 个叶子节点区域 j=1,2,…,$J_m$，计算出最佳拟合值：$c_{m, j}=\frac{\sum_{x_i \in R_{m, j}} r_{m, i}}{\sum_{x_i \in R_{m, j}}\left(y_i-r_{m, i}\right)\left(1-y_i+r_{m, i}\right)}$
  
    
  
    > 证明： 
    >
    > 补充近似值代替过程（用牛顿法迭代来求解）：
    > 假设仅有一个样本: $L\left(y_i, F(x)\right)=-\left(y_i \ln \frac{1}{1+e^{-F(x)}}+\left(1-y_i\right) \ln \left(1-\frac{1}{1+e^{-F(x)}}\right)\right)$
    >
    > 令 $P_i=\frac{1}{1+e^{-F(x)}}$ ，则 $\frac{\partial P_i}{\partial F(x)}=P_i\left(1-P_i\right)（Sigmoid激活函数求导）=-[y_ilogp_i+(1-y_i)log(1-p_i))]$
    >
    > 求一阶导:
    > $$
    > \begin{aligned}
    > \frac{\partial L\left(y_i, F(x)\right)}{\partial F(x)} & =\frac{\partial L\left(y_i, F(x)\right)}{\partial P_i} \cdot \frac{\partial P_i}{\partial F(x)} \\
    > & =-\left(\frac{y_i}{P_i}-\frac{1-y_i}{1-P_i}\right) \cdot\left(P_i \cdot\left(1-P_i\right)\right) \\
    > & =P_i-y_i
    > \end{aligned}
    > $$
    >
    > 求二阶导:
    > $$
    > \begin{aligned}
    > \frac{\partial^2 L\left(y_i, F(x)\right)}{\partial F(x)^2} & =\left(P_i-y_i\right)^{\prime} \\
    > & =P_i\left(1-P_i\right)
    > \end{aligned}
    > $$
    >
    > 对于 $L\left(y_i, F(x)+c\right)$ 的泰勒二阶展开式($c=argmin_cL\left(y_i, F(x)+c\right)$):
    > $$
    > L\left(y_i, F(x)+c\right)=L\left(y_i, F(x)\right)+\frac{\partial L\left(y_i, F(x)\right)}{\partial F(x)} \cdot c+\frac{1}{2} \frac{\partial^2 L\left(y_i, F(x)\right)}{\partial F(x)^2} c^2
    > $$
    > $L\left(y_i, F(x)+c\right)$ 取极值时，上述二阶表达式中的c为:
    > $$
    > \begin{aligned}
    > c & =-\frac{b}{2 a}=-\frac{\frac{\partial L\left(y_i, F(x)\right)}{\partial F(x)}}{2\left(\frac{1}{2} \frac{\partial^2 L\left(y_i, F(x)\right)}{\partial F(x)^2}\right)} \\
    > & =-\frac{\frac{\partial L\left(y_i, F(x)\right)}{\partial F(x)}}{\left.\frac{\partial^2 L\left(y_i, F(x)\right)}{\partial F(x)^2}\right)} \stackrel{\text { 一阶、二阶导代入 }}{\Rightarrow} \frac{y_i-P_i}{P_i\left(1-P_i\right)} \\
    > & \stackrel{r_{mi}=y_i-P_i}{\Rightarrow} \frac{r_{mi}}{(y_i-r_{mi})\left(1-y_i+r_{mi}\right)}
    > \end{aligned}
    > $$
    > 最后再在一阶导和二阶导求解时加上$\sum_{x_i \in R_{m,j}}$,得到上式结果
  
    
  
  - 更新强学习器 $F_m(x)$ :
    $$
    F_m(x)=F_{m-1}(x)+\sum_{j=1}^{J_m} c_{m, j} I\left(x \in R_{m, j}\right)
    $$
  
- step3:得到最终的强学习器 $F_M(x)$ 的表达式:
  $$
  F_M(x)=F_0(x)+\sum_{m=1}^M \sum_{j=1}^{J_m} c_{m, j} I\left(x \in R_{m, j}\right)
  $$

### **GBDT二分类算法实例**

训练集如下表所示，一组数据的特征有年龄和体重，把身高大于1.5米作为分类边界，身高大于1.5米的令标签为1，身高小于等于1.5米的令标签为0，共有4组数据。

![v2-f985b918beb9465e1f1d63d4df26653d_r](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-f985b918beb9465e1f1d63d4df26653d_r.jpg)

测试数据如下表所示，只有一组数据，年龄为25、体重为65，我们用在训练集上训练好的GBDT模型预测该组数据的身高是否大于1.5米？

![v2-b86998f87dd21433d45247347081b561_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-b86998f87dd21433d45247347081b561_1440w.png)

**模型训练阶段**

>参数设置：
>
>- 学习率learning_rate = 0.1
>- 迭代次数：n_trees = 5
>- 树的深度：max_depth = 3

算法流程：

1. 初始化弱学习器：

   $F_0(x)=\log \frac{P(Y=1 \mid x)}{1-P(Y=1 \mid x)}=\log \frac{2}{2}=0$

2. 建立M棵回归树（m=1,2,...,M):

   1. 计算负梯度$r_{m, i}=-\left[\frac{\partial L\left(y_i, F\left(x_i\right)\right)}{\partial F(x)}\right]_{F(x)=F_{m-1}(x)}=y_i-\frac{1}{1+e^{-F\left(x_i\right)}}$

      最后的计算结果如下：

      ![v2-5f8b01323172e9186389745dba59cfba_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-5f8b01323172e9186389745dba59cfba_1440w.webp)

      此时将残差作为样本的标签来训练弱学习器$F_1(x)$，即下表数据：

      ![v2-3665bd298d997938eeae08de504962a2_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-3665bd298d997938eeae08de504962a2_1440w.webp)

**接着寻找回归树的最佳划分节点**，遍历每个特征的每个可能取值。从年龄特征值为5开始，到体重特征为70结束，分别计算分裂后两组数据的平方损失（Square Error），$SSE_L$为左节点的平方损失， $SSE_R$为右节点的平方损失，找到使平方损失和$SSE_{sum}=SSE_L+SSE_R $最小的那个划分节点，即为最佳划分节点。

例如：以年龄7为划分节点，将小于7的样本划分为到左节点，大于等于7的样本划分为右节点。左节点包括 $x_0$，右节点包括样本$x_1,x_2,x_3， SSE_L=0， SSE_R=0.667，SSE_{sum}=0.667 $，所有可能的划分情况如下表所示：

![v2-1a902785ce01139828a3449e8b8f7993_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-1a902785ce01139828a3449e8b8f7993_1440w.webp)

以上划分点的总平方损失最小为**0.000，**有两个划分点：年龄21和体重60，所以随机选一个作为划分点，这里我们选**年龄21**。现在我们的第一棵树长这个样子：

![v2-46ec3591fbaafaa1ce80ae66501e9296_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-46ec3591fbaafaa1ce80ae66501e9296_1440w.webp)

我们设置的参数中树的深度max_depth=3，现在树的深度只有2，需要再进行一次划分，这次划分要对左右两个节点分别进行划分，但是我们在生成树的时候，设置了三个树继续生长的条件：

- **深度没有到达最大。树的深度设置为3，意思是需要生长成3层。**
- **点样本数 >= min_samples_split**
- ***此节点上的样本的标签值不一样（如果值一样说明已经划分得很好了，不需要再分）（本程序满足这个条件，因此树只有2层）\***

最终我们的第一棵回归树长下面这个样子：

![](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-46ec3591fbaafaa1ce80ae66501e9296_1440w.webp)

此时我们的树满足了设置，还需要做一件事情，给这棵树的每个叶子节点分别赋一个参数$c_{m,j}$，来拟合残差。

$c_{1, j}=\frac{\sum_{x_i \in R_{1, j}} r_{1, i}}{\sum_{x_i \in R_{1, j}}\left(y_i-r_{1, i}\right)\left(1-y_i+r_{1, i}\right)}$

根据上述划分结果，为了方便表示，规定从左到右为第1,2个叶子结点，其计算值过程如下：

$\begin{array}{ll}\left(x_0, x_1 \in R_{1,1}\right), & c_{1,1}=\frac{-0.5-0.5}{[(0-(-0.5))*(1-0+(-0.5))]*2}=-2.0 \\ \left(x_2, x_3 \in R_{1,2}\right), & c_{1,2}=2.0\end{array}$

此时的第一棵树长下面这个样子：

![v2-6306341d6ef4848a017c03d8496ed6b6_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-6306341d6ef4848a017c03d8496ed6b6_1440w.webp)

接着更新强学习器，需要用到学习率（这是**Shrinkage**的思想，如果每次都全部加上拟合值 ，即学习率为1，很容易一步学到位导致GBDT过拟合）：learning_rate=0.1，用lr表示。更新公式为：

$F_1(x)=F_0(x)+l r * \sum_{j=1}^2 c_{1, j} I\left(x \in R_{1, j}\right)$

3. **重复此步骤，直到m>5结束，最后生成5棵树强学习器$F_5(x)=F_0(x)+lr * \sum_{m=1}^5 \sum_{j=1}^2 c_{m, j} I\left(x \in R_{m, j}\right)$。

>第一棵树
>
>![](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-6306341d6ef4848a017c03d8496ed6b6_1440w.webp)
>
>第二棵树
>
>- 上一轮得到的$F_1(x)=F_0(x)+l r * \sum_{j=1}^2 c_{1, j} I\left(x \in R_{1, j}\right)$
>
> > $R_{1,1}:F_1(x)=F_0(x)+0.1*c_{1,1}=-0.2\\R_{1,2}:F_1(x)=F_0(x)+0.1*c_{1,2}=0.2$
>
>- 计算负梯度$r_{m, i}=-\left[\frac{\partial L\left(y_i, F\left(x_i\right)\right)}{\partial F(x)}\right]_{F(x)=F_{1}(x)}=y_i-\frac{1}{1+e^{-F_1\left(x_i\right)}}$
>
> > $R_{2,1}:r_{1, i}=0-\frac{1}{1+e^{0.2}}=-\frac{1}{1+e^{0.2}}\\R_{2,2}:F_1(x)=0-\frac{1}{1+e^{-0.2}}=-\frac{1}{1+e^{-0.2}}$
>
>- 给这棵树的每个叶子节点分别赋一个参数$c_{m,j}$，来拟合残差。
>
> $c_{2, j}=\frac{\sum_{x_i \in R_{2, j}} r_{1, i}}{\sum_{x_i \in R_{2, j}}\left(y_i-r_{2, i}\right)\left(1-y_i+r_{2, i}\right)}$
>
> $\begin{array}{ll}\left(x_0, x_1 \in R_{2,1}\right), & c_{2,1}=\frac{-\frac{1}{1+e^{-0.2}}-\frac{1}{1+e^{-0.2}}}{[(0-(-\frac{1}{1+e^{-0.2}}))*(1-0+(-\frac{1}{1+e^{-0.2}}))]*2}=\frac{1}{\frac{1}{1+e^2}-1}=-1.8187 \\ \left(x_2, x_3 \in R_{2,2}\right), & c_{2,2}=1.8187\end{array}$
>
>![v2-73b99303ccecdc78f59d9da5e7523de4_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-73b99303ccecdc78f59d9da5e7523de4_1440w.webp)
>
> 第三棵树
>
> ![v2-c24b8c302fdb469ab169bc2425d336a5_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-c24b8c302fdb469ab169bc2425d336a5_1440w.webp)
>
>
>
> 第四棵树
>
> ![v2-7e26f3b72816524240155e8b672e63b9_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-7e26f3b72816524240155e8b672e63b9_1440w.webp)
>
> 第五棵树
>
> ![v2-9640af653aa523ef5cc2d5ffcf8a9249_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-9640af653aa523ef5cc2d5ffcf8a9249_1440w.webp)

4. 模型预测阶段

   

- $F_0(x)=0$
- 在$F_1(x)$中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**2.0000**。
- 在$F_2(x)$ 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**1.8187**。
- 在 $F_3(x)$中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**1.6826**。
- 在 $F_4(x)$ 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**1.5769**。
- 在 $F_5(x)$ 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**1.4927**。
- **最终预测结果为：**$\begin{aligned} & F(x)=0.0000+0.1 *(2.0000+1.8187+1.6826+1.5769+1.4927)=0.8571 \\ & P(Y=1 \mid x)=\frac{1}{1+e^{-F(x)}}=\frac{1}{1+e^{-0.8571}}=0.7021\end{aligned}$

## GBDT多分类

### softmax损失函数

当使用逻辑回归处理多标签的分类问题时，如果一个样本只对应于一个标签，我们可以假设每个样本属于不同标签的概率服从于几何分布，使用多项逻辑回归（Softmax Regression）来进行分类：
$$
\begin{aligned}
P\left(Y=y_i \mid x\right)=h_\theta(x)\left[\begin{array}{c}
P(Y=1 \mid x ; \theta) \\
P(Y=2 \mid x ; \theta) \\
\cdot \\
\cdot \\
\cdot \\
P(Y=k \mid x ; \theta)
\end{array}\right] \\
=\frac{1}{\sum_{j=1}^k e^{\theta_j^T x}}\left[\begin{array}{c}
e^{\theta_1^T x} \\
e^{\theta_2^T x} \\
\cdot \\
\cdot \\
\cdot \\
e^{\theta_k^T x}
\end{array}\right]
\end{aligned}
$$
其中， $\theta_1, \theta_2, \ldots, \theta_k \in \mathfrak{R}^n$ 为模型的参数，而 $\frac{1}{\sum_{j=1}^k e^{\theta_j^T x}}$ 可以看作是对概率的归一化。一般来说，多项逻辑回归具有参数冗余的特点，即将 $\theta_1, \theta_2, \ldots, \theta_k$ 同时加减一个向量后预测结果不变，因为 $P(Y=1 \mid x)+P(Y=2 \mid x)+\ldots+P(Y=k \mid x)=1$ ，所以 $P(Y=1 \mid x)=1-P(Y=2 \mid x)-\ldots-P(Y=k \mid x)$ 。

假设从参数向量 $\theta_j^T$ 中减去向量 $\psi$ ，这时每一个 $\theta_j^T$ 都变成了 $\theta_j^T-\psi(j=1,2, \ldots, k)$ 。此时假设函数变成了以下公式:

$\begin{aligned} P\left(Y=y_j \mid x ; \theta\right) & =\frac{e^{\theta_j^T x}}{\sum_{i=1}^k e^{\theta_i^T x}} \\ & =\frac{e^{\left(\theta_j^T-\psi\right) x}}{\sum_{i=1}^k e^{\left(\theta_i^T-\psi\right) x}} \\ & =\frac{e^{\theta_j^T x} \times e^{-\psi x}}{\sum_{i=1}^k e^{\theta_i^T x} \times e^{-\psi x}} \\ & =\frac{e^{\theta_j^T x}}{\sum_{i=1}^k e^{\theta_i^T x}}\end{aligned}$

从上式可以看出，从 $\theta_j^T$ 中减去 $\psi$ 完全不影响假设函数的预测结果，这表明前面的Softmax回归模型中存在冗余的参数。特别地，当类别数为 2 时，
$$
h_\theta(x)=\frac{1}{e^{\theta_1^T x}+e^{\theta_2^T x}}\left[\begin{array}{l}
e^{\theta_1^T x} \\
e^{\theta_2^T x}
\end{array}\right]
$$

利用参数冗余的特点，我们将所有的参数减去 $\theta_1$ ，上式变为:
$$
\begin{aligned}
h_\theta(x) & =\frac{1}{e^{0 \cdot x}+e^{\left(\theta_2^T-\theta_1^T\right) x}}\left[\begin{array}{c}
e^{0 \cdot x} \\
e^{\left(\theta_2^T-\theta_1^T\right) x}
\end{array}\right] \\
& =\left[\begin{array}{c}
\frac{1}{1+e^{\theta^T x}} \\
1-\frac{1}{1+e^{\theta^T x}}
\end{array}\right]
\end{aligned}
$$

其中 $\theta=\theta_2-\theta_1$ 。而整理后的式子与逻辑回归一致。因此，多项逻辑回归实际上是二分类逻辑回归在多标签分类下的一种拓展。

当存在样本可能属于多个标签的情况时，我们可以训练 $k$ 个二分类的逻辑回归分类器。第 $i$ 个分类器用以区分每个样本是否可以归为第 $i$ 类，训练该分类器时，需要把标签重新整理为 “第 $i$ 类标签” 与 “非第 $i$ 类标签” 两类。通过这样的办法，我们就解决了每个样本可能拥有多个标签的情况。

在二分类的逻辑回归中，对输入样本 $x$ 分类结果为类别1和 0 的概率可以写成下列形式:
$$
P(Y=y \mid x ; \theta)=\left(h_\theta(x)\right)^y\left(1-h_\theta(x)\right)^{1-y}
$$

其中， $h_\theta(x)=\frac{1}{1+e^{-\theta^T x}}$ 是模型预测的概率值， $y$ 是样本对应的类标签。

将问题泛化为更一般的多分类情况:
$$
P\left(Y=y_i \mid x ; \theta\right)=\prod_{i=1}^K P\left(y_i \mid x\right)^{y_i}=\prod_{i=1}^K h_\theta(x)^{y_i}
$$

由于*连乘可能导致最终结果接近 0* 的问题，一般对似然函数取对数的负数，变成最小化对数似然函数。
$$
-\log P\left(Y=y_i \mid x ; \theta\right)=-\log \prod_{i=1}^K P\left(y_i \mid x\right)^{y_i}=-\sum_{i=1}^K y_i \log \left(h_\theta(x)\right)
$$

### GBDT多分类原理

将GBDT应用于二分类问题需要考虑逻辑回归模型，同理，对于GBDT多分类问题则需要考虑以下 Softmax模型:
$$
\begin{gathered}
P(y=1 \mid x)=\frac{e^{F_1(x)}}{\sum_{i=1}^k e^{F_i(x)}} \\
P(y=2 \mid x)=\frac{e^{F_2(x)}}{\sum_{i=1}^k e^{F_i(x)}} \\
\ldots \\
\cdots \\
P(y=k \mid x)=\frac{e^{F_k(x)}}{\sum_{i=1}^k e^{F_i(x)}}
\end{gathered}
$$

其中 $F_1 \ldots F_k$ 是 $k$ 个不同的CART回归树集成。每一轮的训练实际上是训练了 $k$ 棵树去拟合 softmax的每一个分支模型的负梯度。softmax模型的单样本损失函数为:
$$
\text { loss }=-\sum_{i=1}^k y_i \log P\left(y_i \mid x\right)=-\sum_{i=1}^k y_i \log \frac{e^{F_i(x)}}{\sum_{j=1}^k e^{F_j(x)}}
$$
伪代码

![v2-2ff9038d2b798c3d36dc8e8d5d41ec4d_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-2ff9038d2b798c3d36dc8e8d5d41ec4d_1440w.webp)

对于训练过程的解释：

第一步我们在训练的时候，是针对样本 $x$ 每个可能的类都训练一个分类回归树。举例说明，目前样本有三类，也就是 $K=3$ ，样本 $x$ 属于第二类。那么针对该样本的分类标签，其实可以用一个三维向量 $[0,1,0]$ 来表示。 0 表示样本不属于该类， 1 表示样本属于该类。由于样本已经属于第二类了，所以第二类对应的向量维度为 1 ，其它位置为 0 。

针对样本有三类的情况，我们实质上在每轮训练的时候是同时训练三颗树。第一颗树针对样本 $x$的第一类，输入为 $(x, 0)$ 。第二颗树输入针对样本 $x$ 的第二类，输入为 $(x, 1)$ 。第三颗树针对样本 $x$ 的第三类，输入为 $(x, 0)$ 。这里每颗树的训练过程其实就CART树的生成过程。在此我们参照CART生成树的步骤即可解出三颗树，以及三颗树对 $x$ 类别的预测值 $F_1(x), F_2(x), F_3(x)$ ，那么在此类训练中，我们仿照多分类的逻辑回归，使用Softmax 来产生概率，则属于类别 1 的概率为:
$$
p_1(x)=\frac{\exp \left(F_1(x)\right)}{\sum_{k=1}^3 \exp \left(F_k(x)\right)}
$$

并且我们可以针对类别 1 求出残差 $\tilde{y}_1=0-p_1(x)$ ；类别 2 求出残差 $\tilde{y}_2=1-p_2(x)$ ；类别 3 求出残差 $\tilde{y}_3=0-p_3(x)$ 。

然后开始第二轮训练，针对第一类输入为 $\left(x, \tilde{y}_1\right)$ ，针对第二类输入为 $\left(x, \tilde{y}_2\right)$ ，针对第三类输入为 $\left(x, \tilde{y}_3\right)$ 。继续训练出三颗树。一直迭代M轮。每轮构建 3 颗树。

当 K=3 时，我们其实应该有三个式子:
$$
\begin{aligned}
& F_{1 M}(x)=\sum_{m=1}^M c_{1 m} I\left(x \epsilon R_{1 m}\right) \\
& F_{2 M}(x)=\sum_{m=1}^M c_{2 m} I\left(x \epsilon R_{2 m}\right) \\
& F_{3 M}(x)=\sum_{m=1}^M c_{3 m} I\left(x \epsilon R_{3 m}\right)
\end{aligned}
$$
当训练完以后，对于新样本 ，我们要预测该样本类别的时候，便可以有这三个式子产生三个值 $F_{1M},F_{2M},F_{3M}$。样本属于某个类别的概率为：$p_i(x)=\frac{\exp \left(F_{i M}(x)\right)}{\sum_{k=1}^3 \exp \left(F_{k M}(x)\right)}$

> 推导$\tilde{y}_{ik}$
>
> ![微信图片_20231126103544](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231126103544.jpg)

> 推导$\gamma_{jkm}$
>
> ![微信图片_20231126110046](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231126110046.png)
>
> ![](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231126110055.png))
>
> ![微信图片_20231126110059](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231126110059.png)
>
> ![微信图片_20231126110102](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231126110102.png)
>
> ![微信图片_20231126110106](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231126110106.png)
>
> ![微信图片_20231126121820](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20231126121820.png)

### GBDT多分类实例

- 数据集

$$
\begin{array}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline x_i & 6 & 12 & 14 & 18 & 20 & 65 & 31 & 40 & 1 & 2 & 100 & 101 & 65 & 54 \\
\hline y_i & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 2 & 2 & 2 & 2 \\
\hline
\end{array}
$$

- 模型训练阶段

- 首先，由于我们需要转化3个二分类的问题，所以需要先做一步one-hot：
  $$
  \begin{array}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
  \hline x_i & 6 & 12 & 14 & 18 & 20 & 65 & 31 & 40 & 1 & 2 & 100 & 101 & 65 & 54 \\
  \hline y_i & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 2 & 2 & 2 & 2 \\
  \hline y_{i, 0} & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
  \hline y_{i, 1} & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
  \hline y_{i, 2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 \\
  \hline
  \end{array}
  $$

  > 参数设置：
  > 学习率：learning_rate = 1
  > 树的深度：max_depth = 2
  > 迭代次数：n_trees = 5

     1. 先对所有的样本，进行初始化$F_{k0}(x_i)=\frac{count(k)}{count(n)}$，就是各类别在总样本集中的占比，结果如下表。
  
        ![v2-0c9535cd24219b07ad0893f820f3bd76_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-0c9535cd24219b07ad0893f820f3bd76_1440w.webp)
  
     2. 对第一个类别 ($y_i$=0) 拟合第一颗树 m=1) 
  
  ![v2-ec002ab2961e0db569a3fae02913d3d1_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-ec002ab2961e0db569a3fae02913d3d1_1440w.png)
  
  - 利用公式 $p_{k, m}(x)=\frac{e^{F_{k, m}(x)}}{\sum_{l=1}^K e^{F_{l, m}(x)}}$ 计算概率。
  
  - 计算负梯度值，以 $x_1$ 为例 (k=0,i=1) ：
  
  $\begin{aligned} & \tilde{y}_{i k}=y_{i, k}-p_{k, m-1} \\ & \tilde{y}_{10}=y_{1,0}-p_{0,0}=1-\frac{e^{F_{0,0}(x_1)}}{e^{F_{0,0}(x_1)}+e^{F_{1,0}(x_1)}+e^{F_{2,0}(x_1)}}\approx0.6588\end{aligned}$
  
  同样地，计算其它样本可以有下表：
  
  ![v2-788cfbbb2ca1ea019b5b15e075442533_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-788cfbbb2ca1ea019b5b15e075442533_1440w.png)
  
     3. 寻找回归树的最佳划分节点。在GBDT的建树中，可以采用如MSE、MAE等作为分裂准则来确定分裂点。本文采用的分裂准则是MSE，具体计算过程如下。遍历所有特征的取值，将每个特征值依次作为分裂点，然后计算左子结点与右子结点上的MSE，寻找两者加和最小的一个.
  
        > 比如，选择 $x_8=1$ 作为分裂点时 $(x<1)$ 。
        > 左子结点上的集合的MSE为:
        > $$
        > M S E_{l e f t}=0
        > $$
        >
        > 右子节点上的集合的MSE为:
        > $$
        > \begin{aligned}
        > M S E_{\text {right }} & =(0.6588-0.04342)^2+\ldots+(-0.3412-0.04342)^2 \\
        > & =3.2142
        > \end{aligned}
        > $$
        >
        > 对所有特征计算完后可以发现，当选择 $x_6=31$ 做为分裂点时，可以得到最小的MSE， $M S E=1.42857$ 。
  
     4. 对$x_6=31$ 拟合第一棵回归树
  
        >  ![v2-1a36bddbd939f693afc07cb12024afe3_1440w](https://tpora-alex.oss-cn-guangzhou.aliyuncs.com/v2-1a36bddbd939f693afc07cb12024afe3_1440w.webp)
  
     5. 给这棵树的每个叶子节点分别赋一个参数 $\gamma_{jkm}$，来拟合残差。
  
        $\gamma_{101}=\frac{0.6588*5+(-0.3412)*2}{0.6588*(1-0.6588)*5+0.3412*(1-0.3412)*2}*\frac{2}{3}\approx1.1066$
  
        $\gamma_{201}=-1.0119$
  
  
  
  
  最后，更新 $F_{k m}\left(x_i\right)$ 可得下表:
  $$
  F_{k m}\left(x_i\right)=F_{k, m-1}\left(x_i\right)+\eta * \sum_{x_i \in R_{j k m}} \gamma_{j k m} * I\left(x_i \in R_{j k m}\right)
  $$
  |    $x_(i)$     |   6    |   12   |   14   |   18   |   20   |   65    |   31    |   40    |   1    |   2    |   100   |   101   |   65    |   54    |
  | :------------: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: | :-----: | :----: | :----: | :-----: | :-----: | :-----: | :-----: |
  | $F_{0,1}(x_i)$ | 1.4638 | 1.4638 | 1.4638 | 1.4638 | 1.4638 | -0.6548 | -0.6548 | -0.6548 | 1.4638 | 1.4638 | -0.6548 | -0.6548 | -0.6548 | -0.6548 |
  
  至此第一个类别 (类别0) 的第一颗树拟合完毕，下面开始拟合第二个类别（类别 1 ) 的第一棵树，按照上述过程建立其他两个类别。反复进行，直到训练了M轮。

## GBDT与AdaBoost的不同

- **弱评估器**

> GBDT的弱评估器输出类型不再与整体集成算法输出类型一致。**对于AdaBoost或随机森林算法来说，当集成算法执行的是回归任务时，弱评估器也是回归器，当集成算法执行分类任务时，弱评估器也是分类器**。但对于GBDT而言，**无论GBDT整体在执行回归/分类/排序任务，弱评估器一定是回归器。GBDT通过sigmoid或softmax函数输出具体的分类结果**，但实际弱评估器一定是回归器。

- **损失函数𝐿(𝑥,𝑦)**

> 在GBDT当中，损失函数范围不再局限于固定或单一的某个损失函数，而从数学原理上推广到了任意可微的函数。因此GBDT算法中可选的损失函数非常多，GBDT实际计算的数学过程也与损失函数的表达式无关。

- **拟合残差**

> GBDT依然自适应调整弱评估器的构建，**但却不像AdaBoost一样通过调整数据分布来**间接**影响后续弱评估器。相对的，GBDT通过修改后续弱评估器的拟合目标来直接影响后续弱评估器的结构**。
>
> 具体地来说，**在AdaBoost当中，每次建立弱评估器之前需要修改样本权重，且用于建立弱评估器的是样本𝑋以及对应的𝑦，在GBDT当中，我们不修改样本权重，但每次用于建立弱评估器的是样本𝑋以及当下集成输出𝐻(𝑥𝑖)与真实标签𝑦的差异（𝑦−𝐻(𝑥𝑖))）**。这个差异在数学上被称之为残差（Residual），因此**GBDT不修改样本权重，而是通过拟合残差来影响后续弱评估器结构**。

- **抽样思想**

> GBDT加入了随机森林中随机抽样的思想，在每次建树之前，允许对样本和特征进行抽样来增大弱评估器之间的独立性（也因此可以有袋外数据集）。虽然Boosting算法不会大规模地依赖于类似于Bagging的方式来降低方差，但由于Boosting算法的输出结果是弱评估器结果的加权求和，因此Boosting原则上也可以获得由“平均”带来的小方差红利。当弱评估器表现不太稳定时，采用与随机森林相似的方式可以进一步增加Boosting算法的稳定性。

