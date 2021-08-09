# jha2-answer

## 1

### (a)

![IMG_0472](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/cs224n-a2-p1.PNG)

经过softmax之后，
$$
\hat{y}_{0} = \frac{e^{u^{\top}_{o}v_{c}}}{\sum_{w}e^{u^{\top}_{w}v_{c}}}
$$
以上。

### (b)

![IMG_0473](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/cs224n-a2-p2.PNG)
$$
\frac{\partial \boldsymbol J_{naive-softmax}(\boldsymbol{v}_{c},o,\boldsymbol{U})}{\partial \boldsymbol{v}_{c}} = \boldsymbol{U}^{\top}(\boldsymbol{\hat{y}-\boldsymbol{y}})
$$

### (c)

![IMG_0474](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/cs224n-a2-p3.PNG)
$$
\frac{\partial \boldsymbol J_{naive-softmax}(\boldsymbol{v}_{c},o,\boldsymbol{U})}{\partial \boldsymbol{u}_{w}} = \begin{cases}\boldsymbol v_{c}(\boldsymbol y^{\top}\hat{\boldsymbol y}-1), &\boldsymbol u_w=\boldsymbol u_o\\ \boldsymbol v_c \hat{\boldsymbol y}_w, &\boldsymbol u_w\neq \boldsymbol u_o\end{cases}
$$

### (d)

![IMG_0475](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/cs224n-a2-p4.PNG)
$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

### (e)

![IMG_0476](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/cs224n-a2-p5.PNG)

![IMG_0477](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/cs224n-a2-p6.PNG)

更正：最后一行为+号。
$$
\frac{\partial \boldsymbol J_{neg-sample}(\boldsymbol v_c,o, \boldsymbol U)}{\partial \boldsymbol v_c} = -\sigma(-\boldsymbol u^\top_o \boldsymbol v_c) \boldsymbol u_o - \sum_k \sigma(\boldsymbol u^\top_k \boldsymbol v_c)\boldsymbol u_k
$$

$$
\frac{\partial \boldsymbol J_{neg-sample}(\boldsymbol v_c,o, \boldsymbol U)}{\partial \boldsymbol u_o} = -\sigma(-\boldsymbol u^\top_o \boldsymbol v_c) \boldsymbol v_c
$$

$$
\frac{\partial \boldsymbol J_{neg-sample}(\boldsymbol v_c,o, \boldsymbol U)}{\partial \boldsymbol u_k} = \sigma(\boldsymbol u^\top_k \boldsymbol v_c) \boldsymbol v_c
$$

### (f)

略。

## 2

![word_vectors](https://raw.githubusercontent.com/WinterCyan/imagebed/main/img/cs224n-a2-word_vectors.png)

