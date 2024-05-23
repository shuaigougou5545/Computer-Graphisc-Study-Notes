# 采样

#### （1）基本概念

概率密度函数(Probability Density Function, PDF)：描述连续随机变量在某个取值范围内的概率分布的函数 - 某个位置处出现的概率，通常用f(x)、p(x)或pdf表示

<img src="https://latex.codecogs.com/svg.latex? \int f(x)\text{d}x=1" />

累积分布函数(Cumulative Distribution Function, CDF)：描述随机变量小于等于某个值的概率的函数 - 对PDF的累积，通常用F(x)或cdf表示

<img src="https://latex.codecogs.com/svg.latex?F(x)=\int^x_{-\infty}f(t)\text{d}t=P(X \le x) "/>

#### （2）重要性采样

重要性采样的目标：通过改变采样分布来减少方差，从而提高估计的精度

- 期望的等价转换

  当计算某个函数的期望时，假设变量x符合某个分布p(x)，但分布p(x)往往难以直接采样；我们可以选取另一个容易进行采样的分布q(x)：

  <img src="https://latex.codecogs.com/svg.latex? \begin{array}{c}
  E_{x\sim p}[f(x)]=\int_x f(x)p(x)\text{d}x=\int_x f(x)\frac{p(x)}{q(x)}q(x)\text{d}x=E_{x\sim q}[f(x)\frac{p(x)}{q(x)}]
  \\ \therefore E_{x\sim p}[f(x)]=E_{x\sim q}[f(x)\frac{p(x)}{q(x)}]
  \end{array} "/>

  这一步骤说明在不同分布下采样，通过加权可以得到相同的期望，权值/重要性即为"p(x)/q(x)"

- 蒙特卡洛估计

  通过从q(x)中采样N个点，期望值的蒙特卡洛估计为：

  <img src="https://latex.codecogs.com/svg.latex? E_{x\sim p}[f(x)]\approx \frac{1}{N}\sum_{i=1}^N \frac{p(x_i)}{q(x_i)}f(x_i) "/>

  🏆**关键在于不等式≈在什么情况下接近于等式！**⇒ 为了确保估计的准确性，我们需要最小化估计值的方差：

  <img src="https://latex.codecogs.com/svg.latex? \min \ Var(\frac{f(x)p(x)}{q(x)}) "/>

  为什么方差较小时，采样的估计值越接近真实的期望值？有两种理解方式：(1)当方差较小时，意味着估计值在不同的样本集上波动较小，因此更接近其期望值 (2)利用切比雪夫不等式证明：

  <img src="https://latex.codecogs.com/svg.latex? \begin{array}{c}
  \text{Pr}(|X-\mu|\ge k\sigma)\le \frac{1}{k^2}
  \\ \text{Pr}(|\hat{I}_{IS}-I|\ge k\sqrt{\frac{Var(h(x))}{N}})\le \frac{1}{k^2}
  \end{array} "/>

  当N增加，或者方差减少，估计值越接近于期望值

- 最小化方差

  我们知道常数的方差为0，当q(x)与|f(x)p(x)|成正比关系时，方差最小
  
  <img src="https://latex.codecogs.com/svg.latex? \\ q(x)\propto |f(x)p(x)| \ \ \ \rightarrow\ \ \ \min Var(c) "/>

总结：重要性采样阐述了如何选择合适的采样方式，使得估计期望值更加准确

#### （3）蒙特卡洛积分

- 求解定积分

  在实际应用中，我们往往需要求解函数的定积分(蒙特卡洛积分)：

  <img src="https://latex.codecogs.com/svg.latex? \int f(x)dx=E_{x\sim p}[\frac{f(x)}{p(x)}]\approx \frac{1}{N}\sum_{i=1}^N\frac{f(x_i)}{q(x_i)} "/>

  此时，我们只需要选择合适的分布q(x)，使得q(x)与|f(x)|成正比

- 通过期望证明
  
  <img src="https://latex.codecogs.com/svg.latex? \begin{array}{c}
  E(\frac{f(x)}{p(x)})=\int\frac{f(x)}{p(x)}p(x)\text{d}x=\int f(x)\text{d}x
  \\ \because E(\frac{f(x)}{p(x)})\approx \frac{1}{N} \sum_{i=1}^n \frac{f(x_i)}
  {p(x_i)} \\ \therefore \int f(x)\text{d}x\approx \frac{1}{N} \sum_{i=1}^n \frac{f(x_i)}
  {p(x_i)}\end{array} "/>
  
  在上述过程中，通过选择另一个容易进行采样的分布q(x)，假设了q(x)与|p(x)|存在正比关系，即可将q(x)等价替换上述过程中的p(x)
  
- 蒙特卡洛积分的理解

  在求解定积分时，不同区间段对积分值的贡献应是均等的。然而，由于概率分布(采样)的影响，选取的采样点往往集中在概率密度较大的区域，使得积分值偏向该区域。为了消除这种偏差，需要将f(x)除以p(x)，即调整每个采样点的权重{💡个人认为，这是一种快速理解和记忆蒙特卡洛积分的好方法}

**综上：我们根据目标函数的分布选择相似的采样分布，以提高估计的精度**

#### （4）逆变换采样

逆变换采样：是一种**为任意概率分布生成随机样本**的技术 → 将均匀分布的随机数u映射到目标分布f(x)上，其中映射函数为累积分布函数(cdf)的反函数F<sup>-1</sup>。也就是说，新构造的随机变量X可以表示为F<sup>-1</sup>(u)，认为这个新变量X的分布与目标分布一致

- 证明过程：

  我们假设变量u在[0,1]上均匀线性随机分布，通过映射函数F<sup>-1</sup>得到新的变量X，即

  <img src="https://latex.codecogs.com/svg.latex? \begin{array}{c}
  u\sim U(0,1)
  \\ X=F^{-1}(u)
  \\ \text{}\ F(x)=\int_{-\infty}^x f(t)\text{d}t
  \end{array} "/>

  我们需要验证新变量X与原始变量x具有相同的分布，这可以通过累积分布函数(CDF)进行推导，新变量X的累积分布函数为：

  <img src="https://latex.codecogs.com/svg.latex? \begin{array}{c}
  F_X(x)=P(X\le x)
  \\ = P(F^{-1}(u)\le x)
  \end{array} "/>

  利用累积分布函数F的单调递增特性(反函数F<sup>-1</sup>也必定单调递增)，因此：

  <img src="https://latex.codecogs.com/svg.latex? =P(u\le F(x)) "/>

  又利用u的均匀分布性质，u的累积分布函数为y=x，有：

  <img src="https://latex.codecogs.com/svg.latex? \because P(u\le u)=u "/>

  所以：

  <img src="https://latex.codecogs.com/svg.latex? \begin{array}{c}
  =P(u\le F(x))=F(x)
  \\ \therefore F_X(x)=F(x)
  \end{array} "/>

  综上，证明了随机变量X=F<sup>-1</sup>(u)的累积分布函数F<sub>X</sub>(x)与目标累积分布函数F(x)一致。因此，利用逆变换采样生成的随机变量X确实服从目标概率分布f(x)

- 使用步骤：

  1. 已知概率密度函数(PDF) → p(x) 或者 f(x)

  2. 求累积分布函数(CDF) → F(x) 
  
     <img src="https://latex.codecogs.com/svg.latex? F(x)=\int_{-\infty}^x f(t)\text{d}t "/>
  
3. 求累积分布函数的反函数 → F<sup>-1</sup>(u)
  
4. 生成均匀分布的随机数 → u ~ U(0, 1)
  
5. 通过逆函数映射生成样本 → X = F<sup>-1</sup>(u)

综上所述：逆变换采样是一种非常通用的随机样本生成算法。在已知PDF或CDF的情况下，通过线性随机数和逆函数映射的方法，可以生成指定分布的随机样本。

#### （5）常见采样方式

在图形学中，采样技术的应用主要集中在求解渲染方程过程中。渲染方程通过对着色点的入射半球进行采样。采样分布的选择往往依赖于积分的函数f(x)，而渲染方程中的f(x)包括光源项、BRDF项、余弦项等。根据不同情况，衍生出了如下常见的采样方式：

- 半球均匀采样
- 余弦加权的半球采样 → 基于朗伯余弦定理
- GGX采样 → 基于GGX法线分布
- ...

#### （6）半球均匀采样

> 参考链接 - 半球均匀采样：https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
>
> 参考博客 - 逆变换采样+半球采样：https://zhuanlan.zhihu.com/p/622443806

本节将详细讨论如何**在圆面(2D)和球面(3D)上采样**，并对常见的采样思路进行分析

**2D圆面采样**

- [**简单圆面采样**]圆面上坐标可以用极坐标系表示，即r、θ，一种简单的思路是分别对其取随机小数，使r∈[0, R], θ∈[0, 2𝜋]

  ```python
  def simple_circle_sampling(r, num_samples):
      # 简单圆面采样
      x_list = []
      y_list = []
      for _ in range(num_samples):
          random_theta = random.random() * math.pi * 2 # U(0,1) -> [0,2𝜋]
          random_r = random.random() * r # U(0,1) -> [0,r]
          x = math.cos(random_theta) * random_r
          y = math.sin(random_theta) * random_r
          x_list.append(x)
          y_list.append(y)
      return x_list, y_list
  ```

  可视化采样点如图，可以发现采样点集中在圆心区域：

  <img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405211135739.png" alt="简单圆面采样" style="zoom:20%;" />

  为什么采样点集中在圆心区域，而不是均匀分布？<font color='green'>想要"**均匀**"分布，面积微元dA在圆的不同位置就得保持一致</font>。我们知道dA=rdrdθ，面积微元受到半径r的影响，因此在离圆心较远的位置面积微元较大，点与点之间的距离就更大，显得采样点更稀疏。
  $$
  \text{d}A=\text{弧长}\times \text{半径}=(r\cdot \text{d}\theta) \cdot \text{d}r
  $$
  这种简单圆面采样不能保证均匀分布，想要均匀分布，需要使用一个在中心更加稀疏外围更加密集的采样函数，需要运用到逆变换采样推导

- [**均匀圆面采样**] 均匀分布的核心是面积，我们可以从面积入手。既然假设了是均匀采样，那么PDF已知，为1 / 总圆面面积 [接下来的步骤-逆变换采样]：
  $$
  p(A)=\frac{1}{\text{圆面总面积}}=\frac{1}{\pi R^2}
  \\ \therefore\int_{circle} \frac{1}{\pi R^2} \text{d}A=1
  $$
  将dA转换到极坐标系drdθ：
  $$
  \int_{circle} \frac{1}{\pi R^2} \text{d}A=1
  \\ \therefore\int_{circle} \frac{r}{\pi R^2} \text{d}r\text{d}\theta=1
  \\ \therefore p(r, \theta)=\frac{r}{\pi R^2}
  $$
  得到联合概率密度p(r,θ)，因此可以计算边缘概率密度[联合概率密度→边缘概率密度 - 在另一个变量上求积分]
  $$
  p_r(r,\theta)=\int_0^{2\pi}\frac{r}{\pi R^2}\text{d}\theta=\frac{2r}{R^2}
  \\ p_\theta(r,\theta)=\int_0^{R}\frac{r}{\pi R^2}\text{d}r=\frac{1}{2\pi}
  $$
  求边缘累积分布CDF：
  $$
  F_r(r,\theta)=\int_0^r \frac{2r}{R^2}\text{d}r=\frac{r^2}{R^2}
  \\ F_\theta(r,\theta)= \int_0^\theta \frac{1}{2\pi}\text{d}\theta=\frac{\theta}{2\pi}
  $$
  最后求反函数[求反函数的方法 - 交换x与y]：
  $$
  F_r^{-1}(r,\theta)= R\sqrt{r}
  \\ F_\theta^{-1}(r,\theta)= 2\pi\theta
  $$
  根据逆变换采样的原理，为r和θ分别生成随机数U(0,1)，再经过F<sup>-1</sup>映射到对应变量

  ```python
  def uniform_circle_sampling(R, num_samples):
      #均匀圆面采样
      x_list = []
      y_list = []
      for _ in range(num_samples):
          random_theta = random.random() * math.pi * 2 # 2𝜋θ
          random_r = math.sqrt(random.random()) * R # R·sqrt{r}
          x = math.cos(random_theta) * random_r
          y = math.sin(random_theta) * random_r
          x_list.append(x)
          y_list.append(y)
      return x_list, y_list
  ```

  可视化采样点如图，可以发现采样点均匀分布在圆面上：

  <img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405211218611.png" alt="均匀球面采样" style="zoom:20%;" />

**3D球面采样**

- [简单球面采样]：球面上坐标可以用俯仰角和方位角表示，即θ、𝜑，一种简单的思路是分别对其取随机小数，使θ∈[0, 𝜋], 𝜑∈[0, 2𝜋]

  ```python
  def simple_sphere_sampling(R, num_samples):
      # 简单球面采样
      x_list = []
      y_list = []
      z_list = []
      for _ in range(num_samples):
          random_theta = random.random() * math.pi # U(0,1) -> [0,𝜋]
          random_phi = random.random() * math.pi * 2 # U(0,1) -> [0,2𝜋]
          x = math.sin(random_theta) * math.cos(random_phi) * R
          y = math.sin(random_theta) * math.sin(random_phi) * R
          z = math.cos(random_theta) * R
          x_list.append(x)
          y_list.append(y)
          z_list.append(z)
      return x_list, y_list, z_list
  ```

  可视化采样点如图，可以发现采样点集中在极点区域：

  <img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405211235547.png" alt="简单球面采样" style="zoom:20%;" />

  不均匀的原因很清楚，通过单位立体角分析：
  $$
  \text{d}\omega=\sin\theta\ \text{d}\theta\ \text{d}\phi
  $$
  极点处sinθ最小，故采样点之间拉不开差距

- [**均匀球面采样**] 均匀分布的核心是面积，我们可以从面积入手，球面中与面积相关的概念是单位立体角[单位立体角的定义即面积微元除以半径的平方]。既然假设了是均匀采样，那么PDF已知，为1 / 总立体角 [接下来的步骤-逆变换采样]：
  $$
  p(\omega)=\frac{1}{\text{圆面总立体角}}=\frac{1}{4\pi}
  \\ \therefore\int_{sphere} \frac{1}{4\pi} \text{d}\omega=1
  $$
  将dω转换到单位俯仰角和方位角：
  $$
  \int_{sphere} \frac{1}{4\pi} \text{d}\omega=1
  \\ \therefore \int_{sphere} \frac{\sin\theta}{4\pi} \text{d}\theta \text{d}\phi=1
  \\ \therefore p(\theta,\phi)=\frac{\sin\theta}{4\pi}
  $$
  根据联合概率密度，计算边缘概率密度：
  $$
  p_\theta(\theta,\phi)=\int_0^{2\pi}\frac{\sin\theta}{4\pi}\text{d}\phi=\frac{\sin\theta}{2}
  \\ p_\phi(\theta,\phi)=\int_0^{\pi}\frac{\sin\theta}{4\pi}\text{d}\theta=\frac{1}{2\pi}
  $$
  对边缘概率密度进行积分，得到边缘累积分布CDF：
  $$
  F_\theta(\theta,\phi)=\int_0^\theta p_\theta(\theta,\phi) \text{d}\theta=\frac{1-\cos\theta}{2}
  \\ F_\phi(\theta,\phi)=\int_0^\phi p_\phi(\theta,\phi) \text{d}\phi =\frac{\phi}{2\pi}
  $$
  计算反函数：
  $$
  F^{-1}_\theta(\theta,\phi)=\cos^{-1}(2\theta -1)=\arccos(2\theta-1)
  \\ F^{-1}_\phi(\theta,\phi)=2\pi \phi
  $$
  可视化如下：

  <img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405211256854.png" alt="均匀球面采样" style="zoom:20%;" />

  要想求均匀半球面，上述推导中PDF=1/(2𝜋)，且对应积分区域范围砍半即可，推导过程略，结果为：
  $$
  \theta':F^{-1}_\theta(u)=\arccos(1-u)
  \\ \phi':F^{-1}_\phi(v)=2\pi v
  $$

```cpp
参考博客中涉及圆环(圆)、阿基米德理论(球)进行推导，个人觉得直接用逆变换采样推导来的直观简单
```

#### （7）余弦加权的半球采样

余弦加权的半球采样：一种特殊的球面采样方式，基于朗伯余弦定理，即渲染方程中的cosθ项。由于这种方法会使采样点集中在极点附近，因此它不属于均匀分布，需要单独讨论。

- 推导：我们假设PDF与函数f(x)成正比关系，函数是cosθ，即PDF = c · cosθ，其中c为常数，根据PDF的积分为1求得c：
  $$
  \int_{hemisphere} c\cdot \cos\theta \ \text{d}\omega=1
  \\ \int^{2\pi}_{0}\int^{\pi/2}_0 c \sin\theta\cos\theta \ \text{d}\theta \text{d}\phi=1
  \\ \therefore c = \frac{1}{\pi}
  $$
  概率密度：
  $$
  \\ \therefore p(\omega)=\frac{\cos\theta}{\pi}
  \\ \therefore p(\theta,\phi)=\frac{\sin\theta \cos\theta}{\pi}
  $$
  利用逆变换采样，具体步骤略～，得到如下：
  $$
  F^{-1}_\theta(\theta,\phi)=\frac{\arccos(1-2\theta)}{2}
  \\ F^{-1}_\phi(\theta,\phi)=2\pi\phi
  $$
  代码如下：

  ```cpp
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> rng(0.0, 1.0);
  for (int t = 0; t < sample_side; t++) {
      for (int p = 0; p < sample_side; p++) {
          double samplex = (t + rng(gen)) / sample_side;
          double sampley = (p + rng(gen)) / sample_side;
          
          double theta = 0.5f * acos(1 - 2*samplex); 
          double phi =  2 * M_PI * sampley; 
          Vec3f wi = Vec3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
          float pdf = wi.z / PI; // p(ω)
          
          samlpeList.directions.push_back(wi);
          samlpeList.PDFs.push_back(pdf);
      }
  }
  ```

  <img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405211325491.png" alt="余弦加权的半球采样" style="zoom:20%;" />

- ※算法的特殊之处：当转动图像，从上向下看，可以发现这些点的投影似乎是均匀的 → 将点从球面投影到圆面上，在圆面上均匀分布 → 我们是否可以先生成均匀的圆面采样，再投影到球面之上呢？

  球面上采样点的θ选择，将θ投影到圆面上，得到对应的半径r：
  $$
  F^{-1}_\theta(\theta,\phi)=\frac{\arccos(1-2\theta)}{2}
  \\r_{projection}=\sin(\frac{\arccos(1-2\theta)}{2})
  \\ \because F_r^{-1}(r,\theta)= \sqrt{r}
  $$
  会发现投影的半径，和均匀采样圆面时的半径，值是一致的

#### （8）GGX采样

GGX采样

- 如何设计PDF？

  Microfacet Specular BRDF + cosine项为被积函数，即为光照传输部分，如下：
  $$
  f_r(i,o,h)\cos\theta
  $$
  需要设计一个类似于这个的采样PDF，GGX采样进行如下设计，这里借助了NDF的性质：
  $$
  pdf_m(\textbf{m})=D(\textbf{m})(\textbf{m}\cdot \textbf{n})
  \\ \because \int D(\textbf{m})(\textbf{m}\cdot \textbf{n})\text{d}\omega_h=1
  $$
  其中NDF项使用GGX法线分布：{式子中的m即为半程向量，即微表面的法线朝向}
  $$
  D_{ggx}(m)=\frac{\alpha^2}{\pi ((m\cdot n)^2 (\alpha^2-1) + 1)^2}
  $$
  GGX法线分布中往往假设法线n为(0,0,1)，半程向量m用θ/𝜑表示：
  $$
  \textbf{n}=(0,0,1)
  \\ \textbf{m}=(\sin\theta\cos\phi, \sin\theta\sin\phi,\cos\theta)
  \\ \therefore(\textbf{m}\cdot \textbf{n})=\cos\theta
  $$
  找到了PDF，即可通过逆采样法来构造对应采样点

- [题外话]NDF的理解

  > NDF的相关理解：https://www.reedbeta.com/blog/hows-the-ndf-really-defined/

  NDF(D项)并不是在半球上积分为1，而需额外增加一个cos项，即m点乘n，是因为：
  $$
  \text{d}A_h=D(h)\text{d}\omega_h \cdot A
  \\ \frac{1}{A}\int (n\cdot h)\text{d}A_h=\int D(h)(n\cdot h)\text{d}\omega
  $$
  微观面积dA<sub>h</sub>通过投影(n·h)被投影到宏观表面A之上，所以D(h)不单单是关于单位立体角的概率分布，而且还关于面积

- 设计采样点[逆变换采样法]

  为了从已知PDF<sub>m</sub>(m)构造随机采样方向m，通常将立体角转换成球面坐标系θ和𝜑来处理，即PDF<sub>m</sub>(θ,𝜑)

  在这个过程中，需要一个雅可比行列式(Jacobian)，因为PDF<sub>m</sub>(m)的单位是逆立体角(sr<sup>-1</sup>)，以保证转换是正确的
  $$
  pdf_m(\theta,\phi)=pdf_m(m) \cdot \frac{\text{d}m}{\text{d}\theta\  \text{d}\phi}
  \\=\frac{\alpha^2\cos\theta}{\pi (\cos^2\theta(\alpha^2-1) + 1)^2} \sin\theta
  $$
  很明显PDF与𝜑无关，可以认为𝜑可以在[0,2𝜋]上均匀生成：
  $$
  \phi=2\pi\xi_2
  $$
  对于θ来说，需要走逆变换采样中那一套PDF→CDF→CDF<sup>-1</sup>转换：
  $$
  CDF(\theta)=\int_0^\theta pdf_m(\theta',\phi)\text{d}\theta'
  $$
  令x=tanθ能简化积分的推导：

  <img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405221318508.png" alt="截屏2024-05-22 13.18.14" style="zoom:50%;" />

  因此得到采样点：
  $$
  \theta=\arctan (\sqrt{\frac{\xi_1}{1-\xi_1}})
  \\ \phi = 2\pi \xi_2
  $$

- 蒙特卡洛积分：

  在上述已经推导出PDF<sub>m</sub>，接着需要计算蒙特卡洛积分中的每一项：
  $$
  \frac{1}{N}\sum\frac{f_r(i,o,h)(i,n)}{pdf_i(i)}
  $$
  因为之前在m上积分，可以通过镜面反射关系推导出入射方向i：
  $$
  \textbf{i}=2(\textbf{m}\cdot \textbf{o})\textbf{m}-\textbf{o}
  $$
  还需要将PDF<sub>m</sub>转换为PDF<sub>i</sub>，此处需要一个Jacobian项进行转换：
  $$
  pdf_i(i)=pdf_m(m)\cdot ||\frac{\partial\omega_m}{\partial\omega_i}||
  $$
  对上述i/m/o之间的关系左右两边同时微分进行推导[涉及微分运算法则]：
  $$
  \text{d}\textbf{i}=4(\textbf{i}\cdot \textbf{m})\text{d}\textbf{m}
  \\ ||\frac{\partial\omega_m}{\partial\omega_i}||=\frac{1}{4(\textbf{i}\cdot \textbf{m})}
  $$
  最终的权重为：这里的权重是指蒙特卡洛积分的每一项，即f(xi)/p(xi)
  $$
  weight(\mathbf{i}) = \frac{F(\mathbf{i}, \mathbf{o}, \mathbf{h}) G(\mathbf{i}, \mathbf{o}, \mathbf{h}) (\mathbf{m} \cdot \mathbf{o})}{(\mathbf{m} \cdot \mathbf{n}) (\mathbf{n} \cdot \mathbf{o})}
  $$
  Kulla Conty近似中用到了GGX采样，因为Kulla Conty单独考虑菲涅尔项F，即计算上述权重时认为F=1

- TBN：

  在通过逆变换采样法推导采样微表面法线向量m时，通常假设法线向量n为(0,0,1)，因此推导出的半程向量m是在这个假设下计算的相对值。当有更具体的法线向量n时，需要通过 TBN 矩阵将这个相对的半程向量m转换到实际的世界坐标系中。可以理解为：局部空间 → TBN空间

  ```cpp
  glm::vec3 up = std::abs(N.z) < 0.999 ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
  glm::vec3 T = glm::normalize(glm::cross(up, N));
  glm::vec3 B = glm::cross(N, T); 
  
  return glm::normalize(T * H.x + B * H.y + N * H.z); 
  ```

  一种常见的定义TBN矩阵的方式，构造up向量时防止N与up(往往是z轴)平行，这是为了防止叉乘计算T时结果接近0而不稳定


#### （9）低差异序列

低差异序列(low discrepancy sequences)：用于解决随机采样时采样点分布较为杂乱的问题。如下图，右侧的随机样本更为均匀，我们认为质量更高

<img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405211757398.png" alt="截屏2024-05-21 17.57.17" style="zoom:33%;" />

```
在此不讨论Discrepancy的定义，其大体描述了任意区域内实际样本数与应有样本数之间的差异，这种差异反映着样本在不同区域内的聚集程度
```

常见的低差异序列：

- Halton序列
- Harmmersley序列
- Sobol序列
- Stratified序列

<img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405211802209.png" alt="截屏2024-05-21 18.02.40" style="zoom:35%;" />

**Harmmersley序列**

- 生成方法

  结合了均匀分布和Van der Corput序列，通过在一个维度上使用均匀分布，其他维度使用Van der Corput序列

- Van der Corput序列

  通过基数b进行数字反转生成的低差异序列，其中基数b用于确定反转过程所用的进制数，b常常选择质数，比如2(二进制)、3(三进制)、5(五进制)

  假设b=2，具体步骤：(1)将整数n转换为2进制表示 (2)将2进制表示的数字反转 (3)再次转换回十进制小数

  [比如 1 → 0001 → 1000 → 0.5；2 → 0010 → 0100 → 0.25；3 → 0011 → 1100 → 0.75 ...]

  ```python
  # base = 2 -> 二进制
  def van_der_corput(bits : int) -> float:
    # 分治算法 - 按位反转一个无符号32位整数
    bits = (bits >> 16) | (bits << 16)
    bits = ((bits & 0xFF00FF00) >> 8) | ((bits & 0x00FF00FF) << 8)
    bits = ((bits & 0xF0F0F0F0) >> 4) | ((bits & 0x0F0F0F0F) << 4)
    bits = ((bits & 0xCCCCCCCC) >> 2) | ((bits & 0x33333333) << 2)
    bits = ((bits & 0xAAAAAAAA) >> 1) | ((bits & 0x55555555) << 1)
    return float(bits) * 2.3283064365386963e-10  # 除以2^32,将32位整数转换成0~1浮点数
  ```

- 均匀球面采样 + Harmmersley序列

  ```python
  # base = 2 -> 二进制
  def van_der_corput(bits : int) -> float:
      # 分治算法 - 按位反转一个无符号32位整数
      bits = (bits >> 16) | (bits << 16)
      bits = ((bits & 0xFF00FF00) >> 8) | ((bits & 0x00FF00FF) << 8)
      bits = ((bits & 0xF0F0F0F0) >> 4) | ((bits & 0x0F0F0F0F) << 4)
      bits = ((bits & 0xCCCCCCCC) >> 2) | ((bits & 0x33333333) << 2)
      bits = ((bits & 0xAAAAAAAA) >> 1) | ((bits & 0x55555555) << 1)
      return float(bits) * 2.3283064365386963e-10  # 除以2^32,将32位整数转换成0~1浮点数
  
  def harmmersley(i : int, N : int) -> tuple[float, float]:
      x = float(i) / float(N) 
      y = van_der_corput(i)
      return x, y
  
  def harmmersley_uniform_sphere_sampling(R, num_samples):
      # 均匀球面采样 + 低差异序列
      x_list = []
      y_list = []
      z_list = []
      for index in range(num_samples):
          random_1, random_2 = harmmersley(index, num_samples)
          random_theta = math.acos(2 * random_1 - 1) # acos(2θ-1)
          random_phi = random_2 * math.pi * 2 # 2𝜋𝜑
          x = math.sin(random_theta) * math.cos(random_phi) * R
          y = math.sin(random_theta) * math.sin(random_phi) * R
          z = math.cos(random_theta) * R
          x_list.append(x)
          y_list.append(y)
          z_list.append(z)
      return x_list, y_list, z_list
  ```

  <img src="https://cdn.jsdelivr.net/gh/shuaigougou5545/blog-image/img/202405212230233.png" alt="均匀球面采样_低差异序列" style="zoom:20%;" />

**综上：低差异序列能对给定的整数索引生成均匀分布的小数，可以代替传统的伪随机数用于均匀采样**

-----

总结：<u>蒙特卡洛积分</u>告诉我们可以通过采样和概率密度函数 (PDF) 来计算定积分；<u>重要性采样</u>告诉我们如何选择最优的采样方式/PDF；<u>逆变换采样</u>告诉我们如何在已知 PDF 的前提下生成采样点 → 这一套组合拳就能推理图形学中大部分的采样方式